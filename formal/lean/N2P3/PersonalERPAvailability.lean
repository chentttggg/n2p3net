import N2P3.ERPState
import Mathlib.Data.List.FinRange
import Mathlib.Data.List.Nodup
import Mathlib.Data.List.Basic

/-!
# Actual personal ERP observation, selection, ordered views and scatter

Current source: calibrated.py:493-498,768-810; calibration_candidates.py:89-94,
116-133,555-578,589-592,650-682; continuous_qc.py:543-549. Version identity is
science_contract_v2.json. This finite exact-real layer starts
after the actual dataset/input/participant/recipe/shape guards. QC outputs
are aligned and FIXED, not an ideal-QC or physical threshold assumption.
Changing raw data in ways that alter those earlier checks/QC is outside the
noninterference results. Selected channel order is retained, never sorted.

The small `CeilProduct` primitive is the source float multiplication followed
by np.ceil/int. It may fail; no exact-real ceil is substituted for IEEE
rounding. Its signed integer result is retained without an invented clamp.
Continuous metadata is a parsed unique-key mapping with exact bool tags.
Training union fallback uses channel_local is True without status=applied;
inference local mode and minimum_count require status=applied.

NumericOutcome preserves nonfinite/exception behavior. Option Real is a
logical output slot (none represents NaN). Successful backend vectors may
themselves contain NaN slots; availability never adds a finite-output gate.
The `nanBackend` witness is a typed routing countermodel only. It does not
exhibit a current accepted xDAWN state: current structured_erp.py checks
covariance finiteness and numerical Cholesky before tangent transformation,
and its reader rejects the historical single-bin/degenerate witness.
The generic batch callback is only the already-decoded ERP numerical backend;
view equality is proved before calling it, and no whole-fit or score-equality
premise is used. Its source-shaped finite evaluator instance is given below.

QC evidence-to-original-row reconstruction is a separate small indexed-map
boundary: actual pandas/timeline sorting/physical event authority is not
proved by an arbitrary map. Restricted training masks are explicitly composed
with that map before selection. The training digest explicitly sequences the
three selected array encodings. Per-array contiguous byte encoding and the
hash update/finalize operations remain small representation boundaries; no
collision freedom, concrete NumPy bytes or genuine estimator fit is assumed.
No fitting, training, serialization, dtype, byte/aliasing, or physical
raw-producer correctness is claimed.
-/

namespace N2P3.PersonalERPAvailability

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState

noncomputable section

attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

abbrev Observed (N C : ℕ) := Fin N → Fin C → Bool
abbrev Mask (N : ℕ) := Fin N → Bool

variable {N C T K : ℕ}

def observed (channelMask : Fin C → Bool) (trialMask : Option (Observed N C)) : Observed N C :=
  fun i c => channelMask c && (trialMask.map (fun mask => mask i c)).getD true

theorem observed_without_trial_mask (channels : Fin C → Bool) :
    observed (N:=N) channels none=fun _ c => channels c := by
  funext i c
  simp [observed]

theorem observed_with_trial_mask (channels : Fin C → Bool) (trials : Observed N C) (i : Fin N) (c : Fin C) :
    observed channels (some trials) i c=true ↔ channels c=true ∧ trials i c=true := by simp [observed]

structure QCResult (N C T : ℕ) where
  values : ERPFeatures.Batch N C T
  mask : Observed N C
  drop : Mask N
  allBad : Mask N

structure Prepared (N C T : ℕ) where
  values : ERPFeatures.Batch N C T
  mask : Observed N C
  qualityAvailable : Mask N

def prepare (values : ERPFeatures.Batch N C T) (mask : Observed N C)
    (quality : Option (QCResult N C T)) (epochStatus : Value) : Check (Prepared N C T) := do
  match quality with
  | none => pure ⟨values,mask,fun _ => true⟩
  | some result =>
    ensure (epochStatus ≠ .text "applied") "epoch QC applied twice"
    pure ⟨result.values,result.mask,fun i => !result.drop i && !result.allBad i⟩

theorem prepare_without_quality (values : ERPFeatures.Batch N C T) (mask : Observed N C) (status : Value) :
    prepare values mask none status=.ok ⟨values,mask,fun _ => true⟩ := rfl

theorem prepare_fixed_quality_replaces_values_and_mask (values : ERPFeatures.Batch N C T)
    (mask : Observed N C) (quality : QCResult N C T) (status : Value) (h : status ≠ .text "applied") :
    prepare values mask (some quality) status=.ok
      ⟨quality.values,quality.mask,fun i => !quality.drop i && !quality.allBad i⟩ := by simp [prepare,ensure,h]

theorem prepare_refuses_double_quality (values : ERPFeatures.Batch N C T)
    (mask : Observed N C) (quality : QCResult N C T) :
    prepare values mask (some quality) (.text "applied")=.error (.sourceReject "epoch QC applied twice") := by
  simp [prepare,ensure]

def dictionaryDefault (f : Fields) (key : String) : Value := (f.get key).getD (.object .nil)

def continuousSpec (processing : Fields) : Check Fields := do
  let model ← objectAt (dictionaryDefault processing "model") "continuous model.get"
  objectAt (dictionaryDefault model "spec") "continuous spec.get"

def channelLocal (processing : Fields) : Check Bool := do
  let spec ← continuousSpec processing
  pure (decide (fieldDefault spec "channel_local"=.boolean true))

def inferenceLocal (processing : Fields) : Check Bool :=
  if fieldDefault processing "status"=.text "applied" then channelLocal processing else .ok false

abbrev CeilProduct := ℕ → ℝ → Check ℤ

def minimumCount (ceilProduct : CeilProduct) (processing : Fields) (channels : ℕ) : Check ℤ := do
  if fieldDefault processing "status" ≠ .text "applied" then pure 1
  else
    let localFlag ← channelLocal processing
    if !localFlag then pure 1
    else
      let spec ← continuousSpec processing
      let fraction ← required spec "minimum_channel_fraction"
      let fraction ← match numericScalar fraction with
        | some fraction => .ok fraction
        | none => .error (.fieldFailure "minimum fraction numeric scalar")
      ceilProduct channels fraction

theorem inactive_minimum_is_one (ceilProduct : CeilProduct) (processing : Fields)
    (channels : ℕ) (h : fieldDefault processing "status" ≠ .text "applied") :
    minimumCount ceilProduct processing channels=.ok 1 := by simp [minimumCount,h]

theorem inactive_inference_is_not_local (processing : Fields)
    (h : fieldDefault processing "status" ≠ .text "applied") : inferenceLocal processing=.ok false := by
  simp [inferenceLocal,h]

def localProcessing (status localFlag fraction : Value) : Fields := Fields.ofList
  [("status",status),("model",.object (Fields.ofList [("spec",.object (Fields.ofList
    [("channel_local",localFlag),("minimum_channel_fraction",fraction)]))]))]

theorem local_flag_is_exact_bool (status localFlag fraction : Value) :
    channelLocal (localProcessing status localFlag fraction)=.ok (decide (localFlag=.boolean true)) := by
  simp [channelLocal,continuousSpec,localProcessing,dictionaryDefault,fieldDefault,
    objectAt,Fields.ofList,Fields.get]

theorem numeric_one_is_not_strict_local (status fraction : Value) :
    channelLocal (localProcessing status (.integer 1) fraction)=.ok false := by
  rw [local_flag_is_exact_bool]
  simp

def observedCount (mask : Observed N C) (i : Fin N) : ℕ :=
  ((List.finRange C).filter (mask i)).length

def countAvailable (minimum : ℤ) (mask : Observed N C) (i : Fin N) : Bool :=
  decide (minimum ≤ (observedCount mask i : ℤ))

def baseAvailability (minimum : ℤ) (view : Prepared N C T) : Mask N :=
  fun i => view.qualityAvailable i && countAvailable minimum view.mask i

def allSelected (mask : Observed N C) (selected : List (Fin C)) (i : Fin N) : Bool :=
  selected.all (mask i)

theorem all_selected_iff (mask : Observed N C) (selected : List (Fin C)) (i : Fin N) :
    allSelected mask selected i=true ↔ ∀ c ∈ selected, mask i c=true := by simp [allSelected]

def selectionGate (hasQuality localFlag : Bool) (base : Mask N)
    (mask : Observed N C) (selected : List (Fin C)) : Check (Mask N) :=
  if hasQuality || localFlag then .ok (fun i => base i && allSelected mask selected i)
  else if ∀ i, allSelected mask selected i=true then .ok base
    else .error (.sourceReject "selected channels missing")

theorem soft_available_exact (base : Mask N) (mask : Observed N C) (selected : List (Fin C)) (i : Fin N) :
    (base i && allSelected mask selected i)=true ↔ base i=true ∧ ∀ c ∈ selected, mask i c=true := by
  simp [allSelected]

theorem hard_checks_every_requested_row (base : Mask N) (mask : Observed N C) (selected : List (Fin C)) :
    selectionGate false false base mask selected=.ok base ↔
      ∀ i, ∀ c ∈ selected, mask i c=true := by
  simp only [selectionGate,Bool.false_or,Bool.false_eq_true,↓reduceIte,all_selected_iff]
  split_ifs <;> simp_all

theorem hard_refuses_even_previously_unavailable (base : Mask N) (mask : Observed N C)
    (selected : List (Fin C)) (i : Fin N) (c : Fin C) (hc : c ∈ selected) (hm : mask i c=false) :
    selectionGate false false base mask selected=.error (.sourceReject "selected channels missing") := by
  have hn : ¬∀ j, allSelected mask selected j=true := by
    intro h
    have hx := (all_selected_iff mask selected i).mp (h i) c hc
    simp [hm] at hx
  simp [selectionGate,hn]

def availableRows (availability : Mask N) : List (Fin N) := (List.finRange N).filter availability

theorem available_row_membership (availability : Mask N) (i : Fin N) :
    i ∈ availableRows availability ↔ availability i=true := by simp [availableRows]

theorem available_rows_nodup (availability : Mask N) : (availableRows availability).Nodup :=
  (List.nodup_finRange N).filter _

theorem available_rows_preserve_requested_order (availability : Mask N) :
    (availableRows availability).Sublist (List.finRange N) := List.filter_sublist

def orderedView (values : ERPFeatures.Batch N C T) (rows : List (Fin N)) (selected : List (Fin C)) :
    ERPFeatures.Batch rows.length selected.length T :=
  fun i c t => values (rows.get i) (selected.get c) t

theorem ordered_view_coordinate (values : ERPFeatures.Batch N C T) (rows : List (Fin N))
    (selected : List (Fin C)) (i : Fin rows.length) (c : Fin selected.length) (t : Fin T) :
    orderedView values rows selected i c t=values (rows.get i) (selected.get c) t := rfl

def scatter (rows : List (Fin N)) (scores : Fin rows.length → Option ℝ) : Fin N → Option ℝ :=
  fun i => if h : i ∈ rows then scores ⟨rows.idxOf i,List.idxOf_lt_length_iff.mpr h⟩ else none

theorem scatter_unselected_row (rows : List (Fin N)) (scores : Fin rows.length → Option ℝ)
    (i : Fin N) (h : i ∉ rows) : scatter rows scores i=none := by simp [scatter,h]

theorem scatter_ordered_row (rows : List (Fin N)) (scores : Fin rows.length → Option ℝ)
    (h : rows.Nodup) (i : Fin rows.length) : scatter rows scores (rows.get i)=scores i := by
  unfold scatter
  rw [dif_pos (List.get_mem rows i)]
  congr 1
  apply Fin.ext
  exact List.get_idxOf h i

abbrev Backend (selected times : ℕ) :=
  (rows : ℕ) → ERPFeatures.Batch rows selected times → NumericOutcome (Fin rows → Option ℝ)

def dispatch (selected : List (Fin C)) (backend : Backend selected.length T)
    (values : ERPFeatures.Batch N C T) (availability : Mask N) : NumericOutcome (Fin N → Option ℝ) :=
  let rows := availableRows availability
  if rows=[] then .finite (fun _ => none)
  else numericMap (scatter rows) (backend rows.length (orderedView values rows selected))

theorem no_available_rows_returns_all_nan_without_backend (selected : List (Fin C))
    (backend : Backend selected.length T) (values : ERPFeatures.Batch N C T) (availability : Mask N)
    (h : ∀ i, availability i=false) :
    dispatch selected backend values availability=.finite (fun _ => none) := by
  have he : availableRows availability=[] := by simp [availableRows,h]
  simp [dispatch,he]

theorem dispatch_numerical_exception_is_not_mask_refusal (selected : List (Fin C))
    (backend : Backend selected.length T) (values : ERPFeatures.Batch N C T) (availability : Mask N)
    (h : availableRows availability ≠ []) (op : String)
    (hb : backend (availableRows availability).length
      (orderedView values (availableRows availability) selected)=.exception op) :
    dispatch selected backend values availability=.exception op := by simp [dispatch,h,hb,numericMap,numericThen]

theorem ordered_view_noninterference (x y : ERPFeatures.Batch N C T)
    (rows : List (Fin N)) (selected : List (Fin C))
    (h : ∀ i ∈ rows, ∀ c ∈ selected, ∀ t, x i c t=y i c t) :
    orderedView x rows selected=orderedView y rows selected := by
  funext i c t
  exact h _ (List.get_mem rows i) _ (List.get_mem selected c) t

theorem fixed_qc_consumed_view_noninterference (selected : List (Fin C))
    (backend : Backend selected.length T) (x y : ERPFeatures.Batch N C T) (availability : Mask N)
    (h : ∀ i, availability i=true → ∀ c ∈ selected, ∀ t, x i c t=y i c t) :
    dispatch selected backend x availability=dispatch selected backend y availability := by
  have he : orderedView x (availableRows availability) selected=
      orderedView y (availableRows availability) selected := by
    apply ordered_view_noninterference
    intro i hi c hc t
    exact h i ((available_row_membership _ _).mp hi) c hc t
  simp only [dispatch,he]

def run (ceilProduct : CeilProduct) (processing : Fields) (epochStatus : Value)
    (quality : Option (QCResult N C T)) (values : ERPFeatures.Batch N C T) (mask : Observed N C)
    (selected : List (Fin C)) (backend : Backend selected.length T) :
    Check (NumericOutcome (Fin N → Option ℝ)) := do
  let view ← prepare values mask quality epochStatus
  let minimum ← minimumCount ceilProduct processing C
  let localFlag ← inferenceLocal processing
  let availability ← selectionGate quality.isSome localFlag (baseAvailability minimum view) view.mask selected
  pure (dispatch selected backend view.values availability)

def trainingIntersection (training : Mask N) (mask : Observed N C) : List (Fin C) :=
  (List.finRange C).filter (fun c => decide (∀ i, training i=true → mask i c=true))

def trainingUnion (training : Mask N) (mask : Observed N C) : List (Fin C) :=
  (List.finRange C).filter (fun c => decide (∃ i, training i=true ∧ mask i c=true))

structure TrainingSelection (N C : ℕ) where
  channels : List (Fin C)
  rows : Mask N
  usedUnion : Bool

def trainingSelection (ceilProduct : CeilProduct) (processing : Fields) (training : Mask N)
    (mask : Observed N C) : Check (TrainingSelection N C) := do
  ensure (∃ i, training i=true) "nonempty aligned training mask"
  let common := trainingIntersection training mask
  let minimum ← minimumCount ceilProduct processing C
  if (common.length : ℤ) < minimum then
    let localFlag ← channelLocal processing
    if localFlag then
      let channels := trainingUnion training mask
      pure ⟨channels,fun i => training i && allSelected mask channels i,true⟩
    else pure ⟨common,training,false⟩
  else pure ⟨common,training,false⟩

theorem training_intersection_exact (training : Mask N) (mask : Observed N C) (c : Fin C) :
    c ∈ trainingIntersection training mask ↔ ∀ i, training i=true → mask i c=true := by
  simp [trainingIntersection]

theorem training_union_exact (training : Mask N) (mask : Observed N C) (c : Fin C) :
    c ∈ trainingUnion training mask ↔ ∃ i, training i=true ∧ mask i c=true := by
  simp [trainingUnion]

theorem fallback_rows_never_add_training (training : Mask N) (mask : Observed N C) (i : Fin N) :
    (training i && allSelected mask (trainingUnion training mask) i)=true → training i=true := by
  intro h
  cases ht : training i <;> simp_all

theorem fallback_rows_require_every_union_channel (training : Mask N) (mask : Observed N C) (i : Fin N) :
    (training i && allSelected mask (trainingUnion training mask) i)=true ↔
      training i=true ∧ ∀ c ∈ trainingUnion training mask, mask i c=true := by simp [allSelected]

def remapTraining {M : ℕ} (originalRows : Fin N → Fin M) (training : Mask M) : Mask N :=
  fun i => training (originalRows i)

theorem post_qc_training_uses_original_row {M : ℕ} (originalRows : Fin N → Fin M)
    (training : Mask M) (i : Fin N) : remapTraining originalRows training i=training (originalRows i) := rfl

theorem applied_true_local_routes (fraction : Value) :
    inferenceLocal (localProcessing (.text "applied") (.boolean true) fraction)=.ok true := by
  simp [inferenceLocal,channelLocal,continuousSpec,localProcessing,dictionaryDefault,fieldDefault,
    objectAt,Fields.ofList,Fields.get]

theorem applied_true_minimum_calls_only_scalar_ceil (ceilProduct : CeilProduct) (channels : ℕ) (fraction : ℝ) :
    minimumCount ceilProduct (localProcessing (.text "applied") (.boolean true) (.number fraction)) channels =
      ceilProduct channels fraction := by
  simp [minimumCount,localProcessing,channelLocal,continuousSpec,dictionaryDefault,fieldDefault,
    Fields.ofList,Fields.get,objectAt,required,numericScalar]

theorem applied_numeric_one_keeps_default_minimum (ceilProduct : CeilProduct) (channels : ℕ) (fraction : Value) :
    minimumCount ceilProduct (localProcessing (.text "applied") (.integer 1) fraction) channels=.ok 1 := by
  simp [minimumCount,channelLocal,continuousSpec,localProcessing,dictionaryDefault,fieldDefault,
    Fields.ofList,Fields.get,objectAt]

theorem base_availability_uses_fixed_post_qc_mask (minimum : ℤ) (q : QCResult N C T) (i : Fin N) :
    baseAvailability minimum ⟨q.values,q.mask,fun j => !q.drop j && !q.allBad j⟩ i=true ↔
      q.drop i=false ∧ q.allBad i=false ∧ minimum ≤ (observedCount q.mask i : ℤ) := by
  simp [baseAvailability,countAvailable,and_assoc]

theorem scatter_requested_denominator (rows : List (Fin N)) (scores : Fin rows.length → Option ℝ) :
    (List.ofFn (scatter rows scores)).length=N := by simp

theorem scatter_all_nan (rows : List (Fin N)) : scatter rows (fun _ => none)=(fun _ => none) := by
  funext i
  simp [scatter]

/-- A routing-interface countermodel, not a current xDAWN state witness. -/
def nanBackend : Backend K T := fun _ _ => .finite (fun _ => none)

theorem availability_does_not_imply_finite_output (selected : List (Fin C))
    (values : ERPFeatures.Batch N C T) (availability : Mask N) :
    dispatch selected nanBackend values availability=.finite (fun _ => none) := by
  dsimp only [dispatch]
  split_ifs <;> simp [nanBackend,numericMap,numericThen,scatter_all_nan]

theorem dispatch_nonfinite_is_not_a_source_refusal (selected : List (Fin C))
    (backend : Backend selected.length T) (values : ERPFeatures.Batch N C T) (availability : Mask N)
    (h : availableRows availability ≠ [])
    (hb : backend (availableRows availability).length
      (orderedView values (availableRows availability) selected)=.nonfinite) :
    dispatch selected backend values availability=.nonfinite := by
  simp [dispatch,h,hb,numericMap,numericThen]

def callsBackend (availability : Mask N) : Bool := !(availableRows availability).isEmpty

theorem no_rows_no_call (availability : Mask N) (h : ∀ i, availability i=false) :
    callsBackend availability=false := by simp [callsBackend,availableRows,h]

theorem inference_with_fixed_quality_and_actual_minimum
    (ceilProduct : CeilProduct) (fraction : ℝ) (minimum : ℤ)
    (q : QCResult N C T) (raw : ERPFeatures.Batch N C T) (rawMask : Observed N C)
    (epochStatus : Value) (hs : epochStatus ≠ .text "applied")
    (selected : List (Fin C)) (backend : Backend selected.length T)
    (hceil : ceilProduct C fraction=.ok minimum) :
    run ceilProduct (localProcessing (.text "applied") (.boolean true) (.number fraction))
      epochStatus (some q) raw rawMask selected backend = .ok
        (dispatch selected backend q.values (fun i => (!q.drop i && !q.allBad i) &&
          countAvailable minimum q.mask i && allSelected q.mask selected i)) := by
  simp [run,prepare_fixed_quality_replaces_values_and_mask _ _ _ _ hs,
    applied_true_minimum_calls_only_scalar_ceil,hceil,applied_true_local_routes,selectionGate,
    baseAvailability]

theorem training_empty_mask_refused (ceilProduct : CeilProduct) (processing : Fields) (mask : Observed N C) :
    trainingSelection ceilProduct processing (fun _ => false) mask=
      .error (.sourceReject "nonempty aligned training mask") := by
  simp [trainingSelection,ensure]

theorem training_selection_never_adds_rows (ceilProduct : CeilProduct) (processing : Fields)
    (training : Mask N) (mask : Observed N C) (selection : TrainingSelection N C)
    (h : trainingSelection ceilProduct processing training mask=.ok selection)
    (i : Fin N) (hi : selection.rows i=true) : training i=true := by
  by_cases hne : ∃ i, training i=true
  · cases hm : minimumCount ceilProduct processing C with
    | error err => simp [trainingSelection,ensure,hne,hm] at h
    | ok minimum =>
      by_cases hc : ((trainingIntersection training mask).length : ℤ) < minimum
      · cases hl : channelLocal processing with
        | error err => simp [trainingSelection,ensure,hne,hm,hc,hl] at h
        | ok flag =>
          cases flag with
          | false =>
            have hs : (⟨trainingIntersection training mask,training,false⟩ : TrainingSelection N C)=selection := by
              simpa [trainingSelection,ensure,hne,hm,hc,hl] using h
            subst selection
            exact hi
          | true =>
            have hs : (⟨trainingUnion training mask,
                fun j => training j && allSelected mask (trainingUnion training mask) j,true⟩ :
                TrainingSelection N C)=selection := by
              simpa [trainingSelection,ensure,hne,hm,hc,hl] using h
            subst selection
            exact fallback_rows_never_add_training training mask i hi
      · have hs : (⟨trainingIntersection training mask,training,false⟩ : TrainingSelection N C)=selection := by
          simpa [trainingSelection,ensure,hne,hm,hc] using h
        subst selection
        exact hi
  · simp [trainingSelection,ensure,hne] at h

theorem training_intersection_subset_union (training : Mask N) (mask : Observed N C)
    (hne : ∃ i, training i=true) : (trainingIntersection training mask) ⊆ trainingUnion training mask := by
  intro c hc
  obtain ⟨i,hi⟩ := hne
  rw [training_intersection_exact] at hc
  rw [training_union_exact]
  exact ⟨i,hi,hc i hi⟩

theorem training_channel_order_is_source_order (training : Mask N) (mask : Observed N C) :
    (trainingIntersection training mask).Sublist (List.finRange C) ∧
    (trainingUnion training mask).Sublist (List.finRange C) := ⟨List.filter_sublist,List.filter_sublist⟩

theorem remapped_training_never_creates_original_authorization {M : ℕ}
    (originalRows : Fin N → Fin M) (training : Mask M) (i : Fin N)
    (h : remapTraining originalRows training i=true) : training (originalRows i)=true := h

structure FitView (rows channels times : ℕ) where
  values : ERPFeatures.Batch rows channels times
  labels : Fin rows → Bool
  observed : Observed rows channels

def fitView (values : ERPFeatures.Batch N C T) (labels : Fin N → Bool) (mask : Observed N C)
    (selection : TrainingSelection N C) : FitView (availableRows selection.rows).length selection.channels.length T :=
  ⟨orderedView values (availableRows selection.rows) selection.channels,
    fun i => labels ((availableRows selection.rows).get i),
    fun i c => mask ((availableRows selection.rows).get i) (selection.channels.get c)⟩

theorem fit_view_coordinate (values : ERPFeatures.Batch N C T) (labels : Fin N → Bool)
    (mask : Observed N C) (selection : TrainingSelection N C)
    (i : Fin (availableRows selection.rows).length) (c : Fin selection.channels.length) (t : Fin T) :
    (fitView values labels mask selection).values i c t=
      values ((availableRows selection.rows).get i) (selection.channels.get c) t := rfl

theorem fit_view_label (values : ERPFeatures.Batch N C T) (labels : Fin N → Bool)
    (mask : Observed N C) (selection : TrainingSelection N C)
    (i : Fin (availableRows selection.rows).length) :
    (fitView values labels mask selection).labels i=labels ((availableRows selection.rows).get i) := rfl

theorem fit_view_observed (values : ERPFeatures.Batch N C T) (labels : Fin N → Bool)
    (mask : Observed N C) (selection : TrainingSelection N C)
    (i : Fin (availableRows selection.rows).length) (c : Fin selection.channels.length) :
    (fitView values labels mask selection).observed i c=
      mask ((availableRows selection.rows).get i) (selection.channels.get c) := rfl

/-- Source ascontiguousarray(array).tobytes() at each array boundary. This is
not an assumed whole-fit or whole-digest primitive. -/
structure ArrayEncoders where
  values : (rows channels times : ℕ) → ERPFeatures.Batch rows channels times → List UInt8
  labels : (rows : ℕ) → (Fin rows → Bool) → List UInt8
  observed : (rows channels : ℕ) → Observed rows channels → List UInt8

structure HashOperations (State Digest : Type*) where
  initial : State
  update : State → List UInt8 → State
  finish : State → Digest

def trainingDigest {State Digest : Type*} (hash : HashOperations State Digest) (encode : ArrayEncoders)
    (view : FitView N C T) : Digest :=
  hash.finish (hash.update (hash.update (hash.update hash.initial
    (encode.values N C T view.values)) (encode.labels N view.labels))
    (encode.observed N C view.observed))

theorem training_digest_update_order {State Digest : Type*}
    (hash : HashOperations State Digest) (encode : ArrayEncoders) (view : FitView N C T) :
    trainingDigest hash encode view=hash.finish (hash.update (hash.update (hash.update hash.initial
      (encode.values N C T view.values)) (encode.labels N view.labels))
      (encode.observed N C view.observed)) := rfl

def requestedFitView (values : ERPFeatures.Batch N C T) (labels : Fin N → Bool)
    (mask : Observed N C) (training : Mask N) : FitView (availableRows training).length C T :=
  ⟨fun i c t => values ((availableRows training).get i) c t,
   fun i => labels ((availableRows training).get i),
   fun i c => mask ((availableRows training).get i) c⟩

/-- Literal ERP fit arguments and the two source history digests. This records
the call operands; it never asserts that an estimator was genuinely fitted. -/
structure ERPFitInvocation (rows channels times : ℕ) (Digest : Type*) where
  values : ERPFeatures.Batch rows channels times
  labels : Fin rows → Bool
  requestedDataDigest : Digest
  executedDataDigest : Digest

def erpFitInvocation {State Digest : Type*} (hash : HashOperations State Digest) (encode : ArrayEncoders)
    (values : ERPFeatures.Batch N C T) (labels : Fin N → Bool) (mask : Observed N C)
    (training : Mask N) (selection : TrainingSelection N C) :
    ERPFitInvocation (availableRows selection.rows).length selection.channels.length T Digest :=
  let view := fitView values labels mask selection
  ⟨view.values,view.labels,trainingDigest hash encode (requestedFitView values labels mask training),
    trainingDigest hash encode view⟩

theorem erp_fit_input_uses_selected_ordered_rows {State Digest : Type*}
    (hash : HashOperations State Digest) (encode : ArrayEncoders)
    (values : ERPFeatures.Batch N C T) (labels : Fin N → Bool) (mask : Observed N C)
    (training : Mask N) (selection : TrainingSelection N C)
    (i : Fin (availableRows selection.rows).length) (c : Fin selection.channels.length) (t : Fin T) :
    (erpFitInvocation hash encode values labels mask training selection).values i c t=
      values ((availableRows selection.rows).get i) (selection.channels.get c) t := rfl

theorem erp_fit_row_retains_training_authority
    (ceilProduct : CeilProduct) (processing : Fields) (training : Mask N) (mask : Observed N C)
    (selection : TrainingSelection N C)
    (h : trainingSelection ceilProduct processing training mask=.ok selection)
    (i : Fin (availableRows selection.rows).length) :
    training ((availableRows selection.rows).get i)=true := by
  apply training_selection_never_adds_rows ceilProduct processing training mask selection h
  exact (available_row_membership _ _).mp (List.get_mem _ i)

theorem erp_fit_labels_use_same_ordered_rows {State Digest : Type*}
    (hash : HashOperations State Digest) (encode : ArrayEncoders)
    (values : ERPFeatures.Batch N C T) (labels : Fin N → Bool) (mask : Observed N C)
    (training : Mask N) (selection : TrainingSelection N C)
    (i : Fin (availableRows selection.rows).length) :
    (erpFitInvocation hash encode values labels mask training selection).labels i=
      labels ((availableRows selection.rows).get i) := rfl

theorem erp_executed_digest_uses_actual_fit_view {State Digest : Type*}
    (hash : HashOperations State Digest) (encode : ArrayEncoders)
    (values : ERPFeatures.Batch N C T) (labels : Fin N → Bool) (mask : Observed N C)
    (training : Mask N) (selection : TrainingSelection N C) :
    (erpFitInvocation hash encode values labels mask training selection).executedDataDigest=
      trainingDigest hash encode (fitView values labels mask selection) := rfl

theorem erp_requested_digest_keeps_requested_full_channel_view {State Digest : Type*}
    (hash : HashOperations State Digest) (encode : ArrayEncoders)
    (values : ERPFeatures.Batch N C T) (labels : Fin N → Bool) (mask : Observed N C)
    (training : Mask N) (selection : TrainingSelection N C) :
    (erpFitInvocation hash encode values labels mask training selection).requestedDataDigest=
      trainingDigest hash encode (requestedFitView values labels mask training) := rfl

def affineBackend (width : ℕ) (state : ERPFeatures.Affine (K*(T/width))) : Backend K T :=
  fun _ values => .finite (fun i => some (ERPFeatures.evaluate state (ERPFeatures.binFeatures width (values i))))

def fixtureSelected : List (Fin 3) := [0,2]
def fixtureRaw : ERPFeatures.Batch 3 3 5 := fun _ =>
  ![![1,3,5,7,999],![0,0,0,0,0],![2,4,6,8,-999]]
def fixturePostMask : Observed 3 3 := ![![true,true,true],![true,true,false],![false,true,true]]
def fixtureQC : QCResult 3 3 5 :=
  ⟨fixtureRaw,fixturePostMask,![false,false,true],fun _ => false⟩
def fixtureMinimum : CeilProduct := fun channels fraction =>
  if channels=3 ∧ fraction=(1/2 : ℝ) then .ok 2 else .error (.fieldFailure "outside half-channel fixture")
def fixtureAvailability : Mask 3 := ![true,false,false]

theorem half_of_three_has_ceiling_two : (1 : ℝ) < 3*(1/2 : ℝ) ∧ 3*(1/2 : ℝ) ≤ 2 := by norm_num

theorem fixture_post_observed_counts :
    ∀ i : Fin 3, observedCount fixturePostMask i = (![3,2,2] : Fin 3 → ℕ) i := by
  intro i
  fin_cases i <;> decide

theorem fixture_quality_availability_is_true_false_false :
    (fun i => (!fixtureQC.drop i && !fixtureQC.allBad i) && countAvailable 2 fixtureQC.mask i &&
      allSelected fixtureQC.mask fixtureSelected i)=fixtureAvailability := by
  simp only [fixtureQC,countAvailable,fixture_post_observed_counts]
  funext i
  fin_cases i <;> simp [fixturePostMask,fixtureSelected,fixtureAvailability,allSelected]

theorem fixture_available_rows_in_order : availableRows fixtureAvailability=[0] := by decide

theorem fixture_selected_channel_view_is_original_erp_fixture :
    orderedView fixtureRaw ([0] : List (Fin 3)) fixtureSelected ⟨0,by simp⟩=
      N2P3.ERPFeatures.fixtureEpoch := by
  funext c t
  fin_cases c <;> fin_cases t <;> rfl

theorem fixture_ordered_score_53 :
    dispatch fixtureSelected (affineBackend 2 N2P3.ERPFeatures.fixtureState)
      fixtureRaw fixtureAvailability = .finite ![some 53,none,none] := by
  dsimp only [dispatch]
  rw [fixture_available_rows_in_order]
  simp only [List.cons_ne_self,↓reduceIte]
  simp only [affineBackend,numericMap,numericThen]
  congr 1
  funext i
  fin_cases i
  · simp only [scatter,List.mem_cons,List.not_mem_nil,or_false]
    change some (ERPFeatures.evaluate ERPFeatures.fixtureState (ERPFeatures.binFeatures 2
      (orderedView fixtureRaw ([0] : List (Fin 3)) fixtureSelected ⟨0,by simp⟩)))=some 53
    rw [fixture_selected_channel_view_is_original_erp_fixture,ERPFeatures.fixture_saved_score_53]
  · simp [scatter]
  · simp [scatter]

theorem fixture_full_quality_run :
    run fixtureMinimum (localProcessing (.text "applied") (.boolean true) (.number (1/2)))
      (.text "not_applied") (some fixtureQC) fixtureRaw (fun _ _ => true)
      fixtureSelected (affineBackend 2 N2P3.ERPFeatures.fixtureState)=.ok (.finite ![some 53,none,none]) := by
  rw [inference_with_fixed_quality_and_actual_minimum _ _ 2 _ _ _ _ (by simp) _ _
    (by simp [fixtureMinimum])]
  rw [fixture_quality_availability_is_true_false_false]
  exact congrArg Except.ok fixture_ordered_score_53

def fixtureChangedQC : QCResult 3 3 5 := { fixtureQC with drop := fun _ => true }

theorem fixture_changed_qc_availability_is_empty :
    (fun i => (!fixtureChangedQC.drop i && !fixtureChangedQC.allBad i) &&
      countAvailable 2 fixtureChangedQC.mask i && allSelected fixtureChangedQC.mask fixtureSelected i)=
      (fun _ => false) := by
  funext i
  simp [fixtureChangedQC]

theorem fixture_changed_qc_full_run :
    run fixtureMinimum (localProcessing (.text "applied") (.boolean true) (.number (1/2)))
      (.text "not_applied") (some fixtureChangedQC) fixtureRaw (fun _ _ => true)
      fixtureSelected (affineBackend 2 ERPFeatures.fixtureState)=.ok (.finite (fun _ => none)) := by
  rw [inference_with_fixed_quality_and_actual_minimum _ _ 2 _ _ _ _ (by simp) _ _
    (by simp [fixtureMinimum])]
  rw [fixture_changed_qc_availability_is_empty]
  exact congrArg Except.ok (no_available_rows_returns_all_nan_without_backend _ _ _ _ (by intro i; rfl))

def fixtureReversedEpoch : ERPFeatures.Epoch 2 5 :=
  ![![2,4,6,8,-999],![1,3,5,7,999]]

theorem fixture_reversed_selected_view :
    orderedView fixtureRaw ([0] : List (Fin 3)) [2,0] ⟨0,by simp⟩=fixtureReversedEpoch := by
  funext c t
  fin_cases c <;> fin_cases t <;> rfl

theorem fixture_reversed_bins : ERPFeatures.completeBins 2 fixtureReversedEpoch=![![3,7],![2,6]] := by
  funext c b
  fin_cases c <;> fin_cases b <;>
    norm_num [ERPFeatures.completeBins,ERPFeatures.loopMean,ERPFeatures.sumLoop,ERPFeatures.reshapeBins,
      ERPFeatures.truncated,fixtureReversedEpoch,finProdFinEquiv,Fin.castLE,ERPFeatures.retainedSamples]

theorem fixture_reversed_features : ERPFeatures.binFeatures 2 fixtureReversedEpoch=![3,2,7,6] := by
  funext j
  fin_cases j <;>
    norm_num [ERPFeatures.binFeatures,fixture_reversed_bins,ERPFeatures.flatten,
      ERPFeatures.sourceColumn,finProdFinEquiv,Fin.divNat,Fin.modNat]

theorem fixture_reversed_selected_score_51 :
    ERPFeatures.evaluate ERPFeatures.fixtureState (ERPFeatures.binFeatures 2
      (orderedView fixtureRaw ([0] : List (Fin 3)) [2,0] ⟨0,by simp⟩))=51 := by
  rw [fixture_reversed_selected_view,fixture_reversed_features]
  norm_num [ERPFeatures.evaluate,ERPFeatures.fixtureState,Fin.sum_univ_succ]

def trainingFixtureMask : Observed 6 2 := fun i c =>
  if i.val<4 then true else if i.val=4 then decide (c=0) else decide (c=1)

theorem training_fixture_intersection_empty : trainingIntersection (N:=6) (fun _ => true) trainingFixtureMask=[] := by
  simp [trainingIntersection,trainingFixtureMask]
  decide

theorem training_fixture_union_all : trainingUnion (N:=6) (fun _ => true) trainingFixtureMask=[0,1] := by
  simp [trainingUnion,trainingFixtureMask]
  decide

theorem fallback_without_applied_status_is_executed :
    trainingSelection fixtureMinimum (localProcessing (.text "not_applied") (.boolean true) (.number (1/2)))
      (N:=6) (fun _ => true) trainingFixtureMask = .ok
        ⟨[0,1],fun i => decide (i.val<4),true⟩ := by
  have hm : (fun i : Fin 6 => allSelected trainingFixtureMask [0,1] i)=
      fun i => decide (i.val<4) := by
    funext i
    fin_cases i <;> simp [trainingFixtureMask,allSelected]
  have hmin : minimumCount fixtureMinimum
      (localProcessing (.text "not_applied") (.boolean true) (.number (1/2))) 2=.ok 1 := by
    apply inactive_minimum_is_one
    simp [localProcessing,fieldDefault,Fields.ofList,Fields.get]
  unfold trainingSelection
  rw [hmin]
  simp [ensure,training_fixture_intersection_empty,local_flag_is_exact_bool,training_fixture_union_all,hm]

def emptyFallbackMask : Observed 2 2 := fun i c => decide (i=c)

theorem training_union_fallback_may_return_no_rows :
    trainingSelection fixtureMinimum (localProcessing (.text "not_applied") (.boolean true) (.number (1/2)))
      (N:=2) (fun _ => true) emptyFallbackMask=.ok ⟨[0,1],fun _ => false,true⟩ := by
  have hi : trainingIntersection (N:=2) (fun _ => true) emptyFallbackMask=[] := by decide
  have hu : trainingUnion (N:=2) (fun _ => true) emptyFallbackMask=[0,1] := by decide
  have hm : (fun i : Fin 2 => allSelected emptyFallbackMask [0,1] i)=(fun _ => false) := by
    funext i
    fin_cases i <;> decide
  have hmin : minimumCount fixtureMinimum
      (localProcessing (.text "not_applied") (.boolean true) (.number (1/2))) 2=.ok 1 := by
    apply inactive_minimum_is_one
    simp [localProcessing,fieldDefault,Fields.ofList,Fields.get]
  unfold trainingSelection
  rw [hmin]
  simp [ensure,hi,local_flag_is_exact_bool,hu,hm]

theorem original_row_mapping_is_not_cleaned_position :
    remapTraining (![2,0] : Fin 2 → Fin 3) (![true,false,false] : Mask 3)=![false,true] := by
  funext i
  fin_cases i <;> rfl

end

end N2P3.PersonalERPAvailability
