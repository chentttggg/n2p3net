import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.List.FinRange
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.FinCases

/-!
# Personal waveform LDA: field recovery, feature coordinates and missing channels

Source: calibrated.py:644-659,764-809; binary.py:472-474;
baselines/features.py:29-38; calibration_candidates.py:311-332.

This typed exact-real layer starts after tensor rank/shape/dtype decoding.
Mean, scale and coefficient are separate rank-one vectors of the same length;
intercept is the scalar obtained by `.item()` from any tensor with one element.
The source permits zero-length feature vectors at construction. Only positive
scale is added to the typed constructor; real values exclude NaN/infinity.
The waveform evaluator is restricted to feature width selected.length*bins.
The source constructor does not verify this width against the feature map:
constructor acceptance alone therefore does not guarantee successful scoring.
The four named field readers are independent. Evaluator preservation follows
from their individual recovery and constructor success, never from an assumed
prediction equality or whole-state roundtrip.

`TraceMap` is an explicit trusted numerical boundary for the real source's
np.asarray(dtype=float), rounded target length, SciPy Fourier resample on axis
2 when shorter, and final float32 cast. Its output length is a parameter whose
binding to Python round/SciPy is not proved here. Each output coordinate is a
resampled time sample, not an averaging bin. Channel selection and default
C-order reshape are defined outside this primitive and proved below.

Both current LDA producers save the four fields as float64. This same-saved-
representation proof does not equate the sklearn pipeline bitwise with saved
inference: feature extraction casts to float32, sklearn StandardScaler can
round each transform operation in float32, while saved state promotes the
explicit subtraction/division to float64. No IEEE rounding or Torch/NumPy
serialization refinement, QC transformation, participant/time authorization,
ERP/trunk branch or training correctness is asserted.
-/

namespace N2P3.PersonalLDA

open Matrix
open scoped BigOperators

noncomputable section

abbrev Vector (d : ℕ) := Fin d → ℝ

structure State (d : ℕ) where
  featureMean : Vector d
  featureScale : Vector d
  coefficient : Vector d
  intercept : ℝ

def ValidState {d : ℕ} (s : State d) : Prop := ∀ j, 0 < s.featureScale j

inductive Field where
  | featureMean | featureScale | coefficient | intercept
  deriving DecidableEq

def Field.key : Field → String
  | .featureMean => "feature_mean"
  | .featureScale => "feature_scale"
  | .coefficient => "coefficient"
  | .intercept => "intercept"

inductive Payload (d : ℕ) where
  | vector (values : Vector d)
  | scalar (value : ℝ)

abbrev Record (d : ℕ) := Field → Option (Payload d)

def writeState {d : ℕ} (s : State d) : Record d
  | .featureMean => some (.vector s.featureMean)
  | .featureScale => some (.vector s.featureScale)
  | .coefficient => some (.vector s.coefficient)
  | .intercept => some (.scalar s.intercept)

def readVector {d : ℕ} (r : Record d) (f : Field) : Option (Vector d) :=
  match r f with
  | some (.vector values) => some values
  | _ => none

def readScalar {d : ℕ} (r : Record d) (f : Field) : Option ℝ :=
  match r f with
  | some (.scalar value) => some value
  | _ => none

def readState {d : ℕ} (r : Record d) : Option (State d) := do
  let mean ← readVector r .featureMean
  let scale ← readVector r .featureScale
  let coefficient ← readVector r .coefficient
  let intercept ← readScalar r .intercept
  pure ⟨mean, scale, coefficient, intercept⟩

def loadState {d : ℕ} (r : Record d) : Option (State d) := by
  classical
  exact do
    let s ← readState r
    if ValidState s then some s else none

theorem read_written_mean {d : ℕ} (s : State d) :
    readVector (writeState s) .featureMean = some s.featureMean := rfl

theorem read_written_scale {d : ℕ} (s : State d) :
    readVector (writeState s) .featureScale = some s.featureScale := rfl

theorem read_written_coefficient {d : ℕ} (s : State d) :
    readVector (writeState s) .coefficient = some s.coefficient := rfl

theorem read_written_intercept {d : ℕ} (s : State d) :
    readScalar (writeState s) .intercept = some s.intercept := rfl

theorem readState_recovers_each_field {d : ℕ} (r : Record d) (s : State d)
    (h : readState r = some s) :
    readVector r .featureMean = some s.featureMean ∧
    readVector r .featureScale = some s.featureScale ∧
    readVector r .coefficient = some s.coefficient ∧
    readScalar r .intercept = some s.intercept := by
  cases hm : readVector r .featureMean with
  | none => simp [readState, hm] at h
  | some mean =>
    cases hs : readVector r .featureScale with
    | none => simp [readState, hm, hs] at h
    | some scale =>
      cases hc : readVector r .coefficient with
      | none => simp [readState, hm, hs, hc] at h
      | some coefficient =>
        cases hi : readScalar r .intercept with
        | none => simp [readState, hm, hs, hc, hi] at h
        | some intercept =>
          have he : State.mk mean scale coefficient intercept = s := by
            simpa [readState, hm, hs, hc, hi] using h
          subst s
          exact ⟨rfl, rfl, rfl, rfl⟩

theorem loadState_recovers_fields_and_guard {d : ℕ} (r : Record d) (s : State d)
    (h : loadState r = some s) : readState r = some s ∧ ValidState s := by
  classical
  cases hr : readState r with
  | none => simp [loadState, hr] at h
  | some t =>
    by_cases ht : ValidState t
    · have he : t = s := by simpa [loadState, hr, ht] using h
      subst s
      exact ⟨rfl, ht⟩
    · simp [loadState, hr, ht] at h

def standardized {d : ℕ} (s : State d) (features : Vector d) : Vector d :=
  fun j => (features j - s.featureMean j) / s.featureScale j

def evaluate {d : ℕ} (s : State d) (features : Vector d) : ℝ :=
  dotProduct (standardized s features) s.coefficient + s.intercept

theorem evaluate_explicit {d : ℕ} (s : State d) (features : Vector d) :
    evaluate s features =
      (∑ j, ((features j - s.featureMean j) / s.featureScale j) * s.coefficient j) + s.intercept := rfl

/-- Matrix `@ coefficient + intercept` on every supplied feature row. -/
theorem batch_evaluation_is_matrix_formula {n d : ℕ} (s : State d)
    (features : Matrix (Fin n) (Fin d) ℝ) :
    (fun i => evaluate s (features i)) =
      fun i => ((fun i j => (features i j - s.featureMean j) / s.featureScale j) *ᵥ
        s.coefficient) i + s.intercept := rfl

theorem evaluator_preserved_from_four_field_recovery {d : ℕ}
    (source loaded : State d) (r : Record d)
    (hm : readVector r .featureMean = some source.featureMean)
    (hs : readVector r .featureScale = some source.featureScale)
    (hc : readVector r .coefficient = some source.coefficient)
    (hi : readScalar r .intercept = some source.intercept)
    (hload : loadState r = some loaded) (features : Vector d) :
    evaluate loaded features = evaluate source features := by
  have hf := readState_recovers_each_field r loaded
    (loadState_recovers_fields_and_guard r loaded hload).1
  have hmean : loaded.featureMean = source.featureMean := Option.some.inj (hf.1.symm.trans hm)
  have hscale : loaded.featureScale = source.featureScale := Option.some.inj (hf.2.1.symm.trans hs)
  have hcoef : loaded.coefficient = source.coefficient := Option.some.inj (hf.2.2.1.symm.trans hc)
  have hintercept : loaded.intercept = source.intercept := Option.some.inj (hf.2.2.2.symm.trans hi)
  unfold evaluate standardized
  rw [hmean, hscale, hcoef, hintercept]

theorem valid_written_state_loads {d : ℕ} (s : State d) (h : ValidState s) :
    loadState (writeState s) = some s := by
  simp [loadState, readState, readVector, readScalar, writeState, h]

theorem saved_evaluator_preserved {d : ℕ} (s loaded : State d)
    (hload : loadState (writeState s) = some loaded) (features : Vector d) :
    evaluate loaded features = evaluate s features :=
  evaluator_preserved_from_four_field_recovery s loaded (writeState s)
    (read_written_mean s) (read_written_scale s) (read_written_coefficient s)
    (read_written_intercept s) hload features

abbrev Selection (channels : ℕ) := Option (List (Fin channels))

def selectedChannels {channels : ℕ} (selection : Selection channels) : List (Fin channels) :=
  selection.getD (List.finRange channels)

def ValidSelection {channels : ℕ} (selection : Selection channels) : Prop :=
  match selection with
  | none => True
  | some selected => selected ≠ [] ∧ selected.Nodup

theorem absent_selection_uses_every_channel (channels : ℕ) :
    selectedChannels (none : Selection channels) = List.finRange channels := rfl

theorem supplied_selection_keeps_its_order {channels : ℕ} (selected : List (Fin channels)) :
    selectedChannels (some selected) = selected := rfl

def observedMask {epochs channels : ℕ} (channelMask : Fin channels → Bool)
    (trialMask : Option (Fin epochs → Fin channels → Bool)) : Fin epochs → Fin channels → Bool :=
  fun i c => channelMask c && (trialMask.map (fun mask => mask i c)).getD true

theorem observed_without_trial_mask {epochs channels : ℕ} (mask : Fin channels → Bool) :
    observedMask (epochs := epochs) mask none = fun _ c => mask c := by
  funext i c
  simp [observedMask]

def selectedObserved {epochs channels : ℕ} (observed : Fin epochs → Fin channels → Bool)
    (selected : List (Fin channels)) (i : Fin epochs) : Bool :=
  selected.all (fun c => observed i c)

theorem selectedObserved_iff {epochs channels : ℕ}
    (observed : Fin epochs → Fin channels → Bool) (selected : List (Fin channels)) (i : Fin epochs) :
    selectedObserved observed selected i = true ↔ ∀ c ∈ selected, observed i c = true := by
  simp [selectedObserved]

theorem default_selection_requires_all_channels {epochs channels : ℕ}
    (observed : Fin epochs → Fin channels → Bool) (i : Fin epochs) :
    selectedObserved observed (selectedChannels none) i = true ↔ ∀ c, observed i c = true := by
  simp [selectedObserved, selectedChannels]

inductive MissingPolicy where
  | maskRows | rejectBatch
  deriving DecidableEq

/-- The caller computes this from epoch-QC presence OR the exact
continuous `status=applied` and `channel_local is True` metadata check. -/
def policyFor (hasEpochQC channelLocalApplied : Bool) : MissingPolicy :=
  if hasEpochQC || channelLocalApplied then .maskRows else .rejectBatch

/-- `base` already includes epoch-QC drops/all-bad and minimum usable-channel
count. Hard mode checks all selected observations BEFORE testing base.any. -/
def availabilityGate {epochs channels : ℕ} (policy : MissingPolicy)
    (base : Fin epochs → Bool) (observed : Fin epochs → Fin channels → Bool)
    (selection : Selection channels) : Except Unit (Fin epochs → Bool) := by
  classical
  exact match policy with
  | .maskRows => .ok (fun i => base i && selectedObserved observed (selectedChannels selection) i)
  | .rejectBatch =>
    if ∀ i, selectedObserved observed (selectedChannels selection) i = true then .ok base
    else .error ()

theorem soft_availability_is_rowwise {epochs channels : ℕ} (base : Fin epochs → Bool)
    (observed : Fin epochs → Fin channels → Bool) (selection : Selection channels) :
    availabilityGate .maskRows base observed selection =
      .ok (fun i => base i && selectedObserved observed (selectedChannels selection) i) := rfl

theorem soft_available_iff {epochs channels : ℕ} (base : Fin epochs → Bool)
    (observed : Fin epochs → Fin channels → Bool) (selection : Selection channels) (i : Fin epochs) :
    (base i && selectedObserved observed (selectedChannels selection) i) = true ↔
      base i = true ∧ ∀ c ∈ selectedChannels selection, observed i c = true := by
  simp [selectedObserved]

theorem hard_availability_iff_every_row {epochs channels : ℕ} (base : Fin epochs → Bool)
    (observed : Fin epochs → Fin channels → Bool) (selection : Selection channels) :
    availabilityGate .rejectBatch base observed selection = .ok base ↔
      ∀ i, ∀ c ∈ selectedChannels selection, observed i c = true := by
  classical
  simp only [availabilityGate, selectedObserved_iff]
  split_ifs <;> simp_all

theorem hard_rejection_from_missing_selected {epochs channels : ℕ}
    (base : Fin epochs → Bool) (observed : Fin epochs → Fin channels → Bool)
    (selection : Selection channels) (i : Fin epochs) (c : Fin channels)
    (hc : c ∈ selectedChannels selection) (hmissing : observed i c = false) :
    availabilityGate .rejectBatch base observed selection = .error () := by
  classical
  have hnot : ¬∀ i, selectedObserved observed (selectedChannels selection) i = true := by
    intro h
    have he := (selectedObserved_iff observed _ i).mp (h i) c hc
    simp [hmissing] at he
  simp [availabilityGate, hnot]

abbrev TraceMap (times bins : ℕ) := (Fin times → ℝ) → Fin bins → ℝ

def waveformFeatures {channels times bins : ℕ} (traceMap : TraceMap times bins)
    (selected : List (Fin channels)) (epoch : Fin channels → Fin times → ℝ) :
    Vector (selected.length * bins) := fun j =>
  let pair := finProdFinEquiv.symm j
  traceMap (epoch (selected.get pair.1)) pair.2

theorem flattened_coordinate_is_channel_major {selected bins : ℕ}
    (c : Fin selected) (b : Fin bins) :
    (finProdFinEquiv (c, b)).val = bins * c.val + b.val := by
  simp [finProdFinEquiv]
  omega

theorem waveform_feature_at_coordinate {channels times bins : ℕ}
    (traceMap : TraceMap times bins) (selected : List (Fin channels))
    (epoch : Fin channels → Fin times → ℝ) (c : Fin selected.length) (b : Fin bins) :
    waveformFeatures traceMap selected epoch (finProdFinEquiv (c, b)) =
      traceMap (epoch (selected.get c)) b := by
  simp [waveformFeatures]

theorem waveform_saved_score_formula {channels times bins : ℕ}
    (traceMap : TraceMap times bins) (selected : List (Fin channels))
    (epoch : Fin channels → Fin times → ℝ) (state : State (selected.length * bins)) :
    evaluate state (waveformFeatures traceMap selected epoch) =
      (∑ c, ∑ b, ((traceMap (epoch (selected.get c)) b -
        state.featureMean (finProdFinEquiv (c, b))) /
        state.featureScale (finProdFinEquiv (c, b))) *
        state.coefficient (finProdFinEquiv (c, b))) + state.intercept := by
  rw [evaluate_explicit]
  congr 1
  rw [← Equiv.sum_comp finProdFinEquiv]
  simp [Fintype.sum_prod_type, waveform_feature_at_coordinate]

theorem waveform_evaluator_preserved_from_four_fields {channels times bins : ℕ}
    (traceMap : TraceMap times bins) (selected : List (Fin channels))
    (epoch : Fin channels → Fin times → ℝ)
    (source loaded : State (selected.length * bins)) (r : Record (selected.length * bins))
    (hm : readVector r .featureMean = some source.featureMean)
    (hs : readVector r .featureScale = some source.featureScale)
    (hc : readVector r .coefficient = some source.coefficient)
    (hi : readScalar r .intercept = some source.intercept)
    (hload : loadState r = some loaded) :
    evaluate loaded (waveformFeatures traceMap selected epoch) =
      evaluate source (waveformFeatures traceMap selected epoch) :=
  evaluator_preserved_from_four_field_recovery source loaded r hm hs hc hi hload _

/-- `none` denotes the source's unavailable NaN slot; it is not a real NaN. -/
def scatterScores {epochs d : ℕ} (available : Fin epochs → Bool)
    (state : State d) (features : Fin epochs → Vector d) : Fin epochs → Option ℝ :=
  fun i => if available i then some (evaluate state (features i)) else none

theorem unavailable_row_has_no_score {epochs d : ℕ} (available : Fin epochs → Bool)
    (state : State d) (features : Fin epochs → Vector d) (i : Fin epochs)
    (h : available i = false) : scatterScores available state features i = none := by
  simp [scatterScores, h]

/-- Scoped to the no-trunk/no-ERP waveform-LDA branch, after input/QC gates.
No feature or LDA value is used for a row whose final availability is false. -/
def predictWaveform {epochs channels times bins : ℕ} (traceMap : TraceMap times bins)
    (selection : Selection channels)
    (state : State ((selectedChannels selection).length * bins))
    (policy : MissingPolicy) (base : Fin epochs → Bool)
    (observed : Fin epochs → Fin channels → Bool)
    (epochsX : Fin epochs → Fin channels → Fin times → ℝ) :
    Except Unit (Fin epochs → Option ℝ) := do
  let available ← availabilityGate policy base observed selection
  pure (scatterScores available state (fun i =>
    waveformFeatures traceMap (selectedChannels selection) (epochsX i)))

theorem waveform_hard_missing_rejected {epochs channels times bins : ℕ}
    (traceMap : TraceMap times bins) (selection : Selection channels)
    (state : State ((selectedChannels selection).length * bins))
    (base : Fin epochs → Bool) (observed : Fin epochs → Fin channels → Bool)
    (epochsX : Fin epochs → Fin channels → Fin times → ℝ)
    (i : Fin epochs) (c : Fin channels) (hc : c ∈ selectedChannels selection)
    (hmissing : observed i c = false) :
    predictWaveform traceMap selection state .rejectBatch base observed epochsX = .error () := by
  simp [predictWaveform, hard_rejection_from_missing_selected base observed selection i c hc hmissing]

theorem waveform_predictor_preserved_from_four_fields {epochs channels times bins : ℕ}
    (traceMap : TraceMap times bins) (selection : Selection channels)
    (source loaded : State ((selectedChannels selection).length * bins))
    (r : Record ((selectedChannels selection).length * bins))
    (hm : readVector r .featureMean = some source.featureMean)
    (hs : readVector r .featureScale = some source.featureScale)
    (hc : readVector r .coefficient = some source.coefficient)
    (hi : readScalar r .intercept = some source.intercept)
    (hload : loadState r = some loaded) (policy : MissingPolicy)
    (base : Fin epochs → Bool) (observed : Fin epochs → Fin channels → Bool)
    (epochsX : Fin epochs → Fin channels → Fin times → ℝ) :
    predictWaveform traceMap selection loaded policy base observed epochsX =
      predictWaveform traceMap selection source policy base observed epochsX := by
  have hscores : ∀ features, evaluate loaded features = evaluate source features :=
    evaluator_preserved_from_four_field_recovery source loaded r hm hs hc hi hload
  unfold predictWaveform
  cases hg : availabilityGate policy base observed selection with
  | error e => rfl
  | ok available =>
    change Except.ok (scatterScores available loaded _) = Except.ok (scatterScores available source _)
    apply congrArg Except.ok
    funext i
    unfold scatterScores
    split_ifs
    · rw [hscores]
    · rfl

def legalState : State 2 := ⟨fun _ => 0, fun _ => 1, ![1, 10], 3⟩
def legalFeatures : Vector 2 := ![2, 4]
def changedCoefficient : State 2 := { legalState with coefficient := ![2, 10] }
def zeroScaleState : State 2 := { legalState with featureScale := fun _ => 0 }

theorem legal_state_valid : ValidState legalState := by
  intro j
  norm_num [legalState]

theorem legal_state_constructs : loadState (writeState legalState) = some legalState :=
  valid_written_state_loads legalState legal_state_valid

theorem legal_saved_score : evaluate legalState legalFeatures = 45 := by
  norm_num [evaluate, standardized, legalState, legalFeatures, dotProduct, Fin.sum_univ_two]

theorem changed_coefficient_changes_score : evaluate changedCoefficient legalFeatures = 47 := by
  norm_num [evaluate, standardized, changedCoefficient, legalState, legalFeatures,
    dotProduct, Fin.sum_univ_two]

theorem zero_scale_rejected : loadState (writeState zeroScaleState) = none := by
  have hn : ¬ValidState zeroScaleState := by
    intro h
    have hf := h 0
    norm_num [zeroScaleState, legalState] at hf
  simp [loadState, readState, readVector, readScalar, writeState, hn]

def partialObserved : Fin 1 → Fin 2 → Bool := fun _ c => c == 0
def baseAvailable : Fin 1 → Bool := fun _ => true
def baseUnavailable : Fin 1 → Bool := fun _ => false

theorem default_missing_channel_rejects :
    availabilityGate .rejectBatch baseAvailable partialObserved none = .error () := by
  have hn : ¬∀ i, selectedObserved partialObserved (selectedChannels none) i = true := by
    intro h
    have hf := (default_selection_requires_all_channels partialObserved 0).mp (h 0) 1
    norm_num [partialObserved] at hf
  simp [availabilityGate, hn]

theorem selected_subset_accepts_missing_unselected :
    availabilityGate .rejectBatch baseAvailable partialObserved (some [0]) = .ok baseAvailable := by
  rw [hard_availability_iff_every_row]
  intro i c hc
  have he : c = 0 := by simpa [selectedChannels] using hc
  subst c
  rfl

theorem soft_missing_channel_masks_row :
    availabilityGate .maskRows baseAvailable partialObserved none = .ok baseUnavailable := by
  rw [soft_availability_is_rowwise]
  congr 1

def twoRowObserved : Fin 2 → Fin 2 → Bool := fun i c => i == 0 || c == 0
def twoRowBase : Fin 2 → Bool := fun _ => true

theorem first_row_is_fully_observed : ∀ c, twoRowObserved 0 c = true := by
  intro c
  simp [twoRowObserved]

theorem hard_partial_batch_rejects_fully_observed_row :
    availabilityGate .rejectBatch twoRowBase twoRowObserved none = .error () := by
  have hn : ¬∀ i, selectedObserved twoRowObserved (selectedChannels none) i = true := by
    intro h
    have hf := (default_selection_requires_all_channels twoRowObserved 1).mp (h 1) 1
    norm_num [twoRowObserved] at hf
  simp [availabilityGate, hn]

def orderedEpoch : Fin 2 → Fin 2 → ℝ := ![![1, 2], ![3, 4]]
def identityTrace : TraceMap 2 2 := id
abbrev orderedChannels : List (Fin 2) := [0, 1]
def secondCoordinate : Fin (orderedChannels.length * 2) := ⟨1, by decide⟩

theorem channel_major_second_feature :
    waveformFeatures identityTrace orderedChannels orderedEpoch secondCoordinate = 2 := by
  have hi : secondCoordinate = finProdFinEquiv
      ((⟨0, by decide⟩ : Fin orderedChannels.length), (1 : Fin 2)) := rfl
  rw [hi, waveform_feature_at_coordinate]
  norm_num [identityTrace, orderedEpoch, orderedChannels]

end

end N2P3.PersonalLDA
