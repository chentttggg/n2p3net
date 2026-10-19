import N2P3.WaveformFeatureWidth
import N2P3.ERPFeatures
import Mathlib.Tactic.Linarith

/-! SelectionERP's actual physical sampling seam. Contract:
lean_selection_erp_20261018/source_contract_01.json, fixed before this file.
The full selection filter and ONE per-trace shared FFT operation are explicit;
the event window, transpose, first178 crop and mean-only baseline are literal.
This does not replace a complete preprocessing/decoder by a success oracle.
Native FFT, rounding and cast operations retain their individual interfaces. -/
namespace N2P3.SelectionERPSupport
open Matrix
open scoped BigOperators
noncomputable section

def sourceRate : ℝ := 256
def outputRate : ℝ := 128
def firstOffset : ℤ := -51
def stopExclusive : ℤ := 307

theorem closed_source_endpoints_give_358_raw_samples :
    stopExclusive-firstOffset=358 ∧ stopExclusive-1=306 := by
  norm_num [stopExclusive,firstOffset]

theorem the_two_output_count_formulas_are_distinct :
    (358*128/256 : ℚ)=179 ∧ (306-(-51):ℤ)=357 ∧ (357*128/256:ℚ)=178+1/2 := by norm_num

structure InputGeometry (samples eventsMinusOne : ℕ) where
  onset : Fin (eventsMinusOne+1) → ℕ
  ordered : StrictMono onset
  first : onset 0=51
  last : onset (Fin.last eventsMinusOne)+307=samples

theorem ordered_onsets_derive_both_physical_window_bounds {n k : ℕ}
    (g : InputGeometry n k) (event : Fin (k+1)) :
    51 ≤ g.onset event ∧ g.onset event+307 ≤ n := by
  have lower := g.ordered.monotone (Fin.zero_le event)
  have upper := g.ordered.monotone (Fin.le_last event)
  rw [g.first] at lower
  have last := g.last
  omega

theorem admitted_geometry_has_at_least_one_complete_raw_window {n k : ℕ}
    (g : InputGeometry n k) : 358 ≤ n := by
  have bounds := ordered_onsets_derive_both_physical_window_bounds g 0
  rw [g.first] at bounds
  omega

def windowIndex {n k : ℕ} (g : InputGeometry n k) (event : Fin (k+1)) (time : Fin 358) : Fin n :=
  ⟨(g.onset event - 51) + time.val,by
    have bounds := ordered_onsets_derive_both_physical_window_bounds g event
    have inside := time.isLt
    omega⟩

theorem literal_window_coordinate_and_support {n k : ℕ} (g : InputGeometry n k)
    (event : Fin (k+1)) (time : Fin 358) :
    (windowIndex g event time).val = (g.onset event - 51) + time.val ∧
      (g.onset event - 51) ≤ (windowIndex g event time).val ∧
      (windowIndex g event time).val < g.onset event + 307 := by
  have bounds := ordered_onsets_derive_both_physical_window_bounds g event
  have inside := time.isLt
  change ((g.onset event - 51) + time.val = (g.onset event - 51) + time.val) ∧
    (g.onset event - 51 ≤ (g.onset event - 51) + time.val) ∧
    ((g.onset event - 51) + time.val < g.onset event + 307)
  exact ⟨rfl,by omega,by omega⟩

theorem raw_window_is_exactly_half_open_358 {n k : ℕ} (g : InputGeometry n k)
    (event : Fin (k+1)) : (g.onset event+307)-(g.onset event-51)=358 := by
  have bounds := ordered_onsets_derive_both_physical_window_bounds g event
  omega

theorem the_last_raw_window_coordinate_is_plus306 {n k : ℕ} (g : InputGeometry n k)
    (event : Fin (k+1)) : (windowIndex g event (Fin.last 357)).val=g.onset event+306 := by
  have bounds := ordered_onsets_derive_both_physical_window_bounds g event
  simp only [windowIndex,Fin.val_last]
  omega

def rawEpochs {n k c : ℕ} (g : InputGeometry n k) (samples : Matrix (Fin n) (Fin c) ℝ) :
    Fin (k+1) → ERPFeatures.Epoch c 358 :=
  fun event channel time => samples (windowIndex g event time) channel

def filteredEpochs {n k c : ℕ} (g : InputGeometry n k)
    (filter : Matrix (Fin n) (Fin n) ℝ) (samples : Matrix (Fin n) (Fin c) ℝ) :
    Fin (k+1) → ERPFeatures.Epoch c 358 := rawEpochs g (filter*samples)

theorem filter_first_then_slice_transpose_uses_full_selection_coordinates {n k c : ℕ}
    (g : InputGeometry n k) (filter : Matrix (Fin n) (Fin n) ℝ)
    (samples : Matrix (Fin n) (Fin c) ℝ) (event : Fin (k+1)) (channel : Fin c) (time : Fin 358) :
    filteredEpochs g filter samples event channel time=
      ∑ raw,filter (windowIndex g event time) raw*samples raw channel := rfl

abbrev TraceFFT := (Fin 358 → ℝ) → (Fin 179 → ℝ)

def first178 (resampled : Fin 179 → ℝ) (time : Fin 178) : ℝ :=
  resampled (Fin.castLE (by decide : 178 ≤ 179) time)

def fftThenCrop {c : ℕ} (fft : TraceFFT) (epoch : ERPFeatures.Epoch c 358) : ERPFeatures.Epoch c 178 :=
  fun channel => first178 (fft (epoch channel))

theorem first178_coordinate_is_the_original_resampled_position (values : Fin 179 → ℝ) (time : Fin 178) :
    first178 values time=values ⟨time.val,lt_of_lt_of_le time.isLt (by decide)⟩ := rfl

theorem resampled_position_178_is_never_read_by_the_first178_crop (values changed : Fin 179 → ℝ)
    (same : ∀ time : Fin 178, values (Fin.castLE (by decide) time)=changed (Fin.castLE (by decide) time)) :
    first178 values=first178 changed := funext same

def physicalTime (time : Fin 178) : ℝ := -(51/256:ℝ)+time.val/128

theorem physical_output_time_is_negative_exactly_for_first26 (time : Fin 178) :
    physicalTime time < 0 ↔ time.val < 26 := by
  unfold physicalTime
  constructor
  · intro earlier
    have bound : (time.val : ℝ) < 51/2 := by linarith
    have natBound : time.val < 26 := by exact_mod_cast (show (time.val:ℝ) < 26 by linarith)
    exact natBound
  · intro before
    have bound : (time.val:ℝ) ≤ 25 := by exact_mod_cast (show time.val ≤ 25 by omega)
    linarith

theorem the_output_grid_has_no_sample_at_zero_time (time : Fin 178) : physicalTime time ≠ 0 := by
  intro zero
  have equation : (2*time.val : ℕ)=51 := by
    exact_mod_cast (show (2:ℝ)*(time.val:ℝ)=51 by unfold physicalTime at zero; linarith)
  omega

theorem first_and_last_output_times_are_the_declared_grid :
    physicalTime 0=-(51/256:ℝ) ∧ physicalTime (Fin.last 177)=303/256 := by
  norm_num [physicalTime]

def baselineMean (trace : Fin 178 → ℝ) : ℝ :=
  (∑ t : Fin 26,trace (Fin.castLE (by decide : 26 ≤ 178) t))/26

inductive Baseline where | none | meanOnly deriving DecidableEq
inductive Family where | waveform | toeplitz | xdawn deriving DecidableEq

def applyBaseline (mode : Baseline) (trace : Fin 178 → ℝ) (time : Fin 178) : ℝ :=
  match mode with
  | .none => trace time
  | .meanOnly => trace time-baselineMean trace

def epochSeam {c : ℕ} (fft : TraceFFT) (mode : Baseline) (epoch : ERPFeatures.Epoch c 358) :
    ERPFeatures.Epoch c 178 := fun channel => applyBaseline mode (fftThenCrop fft epoch channel)

theorem mean_only_baseline_uses_the_actual26_original_cropped_samples
    (trace : Fin 178 → ℝ) (time : Fin 178) :
    applyBaseline .meanOnly trace time=trace time-
      (∑ t : Fin 26,trace (Fin.castLE (by decide) t))/26 := rfl

theorem constant_cropped_voltage_is_preserved_without_baseline (value : ℝ) :
    applyBaseline .none (fun _ => value)=fun _ => value := rfl

theorem constant_cropped_voltage_is_removed_by_mean_only_baseline (value : ℝ) :
    applyBaseline .meanOnly (fun _ => value)=fun _ => 0 := by
  funext time
  simp [applyBaseline,baselineMean]

def familyBaselineAllowed (family : Family) (baseline : Baseline) : Prop :=
  match family with
  | .waveform => True
  | .toeplitz => baseline=.none
  | .xdawn => baseline=.meanOnly

theorem actual_family_baseline_admission :
    familyBaselineAllowed .waveform .none ∧ familyBaselineAllowed .waveform .meanOnly ∧
      familyBaselineAllowed .toeplitz .none ∧ ¬familyBaselineAllowed .toeplitz .meanOnly ∧
      familyBaselineAllowed .xdawn .meanOnly ∧ ¬familyBaselineAllowed .xdawn .none := by
  simp [familyBaselineAllowed]

theorem actual_toeplitz_grid_reuses_complete_equal_width_bins :
    178/4=44 ∧ ERPFeatures.retainedSamples 178 4=176 ∧ 178%4=2 := by
  norm_num [ERPFeatures.retainedSamples]

theorem actual_waveform_rounding_quotient_is_inside_the_nonhalf28_cell :
    (178*20/128:ℚ)=445/16 ∧ (27+1/2:ℚ)<445/16 ∧ (445/16:ℚ)<28+1/2 := by norm_num

theorem waveform_physical_width_from_individual_accepted_owner_operations
    {Scalar : Type} (ops : WaveformFeatureWidth.Arithmetic Scalar) (channels : ℕ)
    (rate timeValue numerator quotient : Scalar)
    (convert : ops.fromNat 178=.ok timeValue) (multiply : ops.multiply timeValue ops.twenty=.ok numerator)
    (divide : ops.divide numerator rate=.ok quotient) (round28 : ops.roundInt quotient=.ok 28) :
    WaveformFeatureWidth.expectedWidth ops channels 178 rate none=.ok ((channels:ℤ)*28) := by
  have width := WaveformFeatureWidth.width_from_individual_operations ops channels 178 rate none
    timeValue numerator quotient 28 convert multiply divide round28
  simpa only [WaveformFeatureWidth.selectedCount,show min (178:ℤ) 28=28 by norm_num] using width

end
end N2P3.SelectionERPSupport
