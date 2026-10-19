import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Lean.Elab.Tactic.Omega

/-!
# Actual uniform ERP bins, channel-prime coordinates and temporal projection

Source: structured_erp.py:76-85,125-145 and temporal_transfer.py:240-244.
The finite exact-real domain is AFTER both original arrays pass full-array
finiteness, N/C/T shape and fitted-geometry checks. In particular an ignored
tail changed to NaN can be refused by the source before cropping. The typed
arrays contain only finite real representatives and have fixed dimensions;
positive width, positive channels and width<=T are stated by InputAdmitted.
N=0 remains allowed, exactly as prediction's empty-batch branch.

Cropping, C-order reshape and mean are explicit finite indexing/reduction.
The source transposes (N,C,B) to (N,B,C) before flattening, so the coefficient
coordinate is t*C+c. This differs from PersonalLDA's c*B+t representation.
The optional temporal map is literal X@basis, with no orthogonality assumption
and no opaque trace/score function. All formulas and linearity follow from
these operations; no output-equality premise is used.

The affine layer covers pooled/toeplitz/temporal ERP after their state guards;
xDAWN/OAS/Riemann forward needs its separate numerical primitives. This file
does not prove training, QC, participant authority, float64 NumPy conversion,
IEEE accumulation, storage, or the physical preprocessing producer correct.
-/

namespace N2P3.ERPFeatures

open scoped BigOperators

noncomputable section

abbrev Vector (d : ℕ) := Fin d → ℝ
abbrev Epoch (channels times : ℕ) := Fin channels → Fin times → ℝ
abbrev Batch (epochs channels times : ℕ) := Fin epochs → Epoch channels times

variable {C T B R N : ℕ}

def InputAdmitted (channels times width : ℕ) : Prop :=
  0 < channels ∧ 0 < width ∧ width ≤ times

/-- Reduction in increasing source offset, equivalent over reals to mean. -/
def sumLoop : (width : ℕ) → Vector width → ℝ
  | 0, _ => 0
  | width+1, values => sumLoop width (fun u => values u.castSucc) + values (Fin.last width)

theorem sumLoop_eq_sum (width : ℕ) (values : Vector width) :
    sumLoop width values = ∑ u, values u := by
  induction width with
  | zero => simp [sumLoop]
  | succ width ih => rw [sumLoop,ih,Fin.sum_univ_castSucc]

def loopMean (width : ℕ) (values : Vector width) : ℝ := sumLoop width values / width

theorem loopMean_formula (width : ℕ) (values : Vector width) :
    loopMean width values = (∑ u, values u) / width := by rw [loopMean,sumLoop_eq_sum]

theorem loopMean_combine (width : ℕ) (a b : ℝ) (x y : Vector width) :
    loopMean width (fun u => a*x u+b*y u) = a*loopMean width x+b*loopMean width y := by
  simp only [loopMean_formula,Finset.sum_add_distrib,←Finset.mul_sum]
  ring

def retainedSamples (times width : ℕ) : ℕ := (times/width)*width

theorem retained_plus_discarded (times width : ℕ) :
    retainedSamples times width + times%width = times := by
  simpa [retainedSamples,Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div times width

theorem retained_not_longer (times width : ℕ) : retainedSamples times width ≤ times :=
  Nat.div_mul_le_self _ _

theorem positive_complete_bin_count (channels times width : ℕ)
    (h : InputAdmitted channels times width) : 0 < times/width := by
  apply Nat.div_pos h.2.2 h.2.1

def truncated (width : ℕ) (x : Epoch C T) : Epoch C (retainedSamples T width) :=
  fun c t => x c (Fin.castLE (retained_not_longer T width) t)

def reshapeBins (width : ℕ) (x : Epoch C T) : Fin C → Fin (T/width) → Fin width → ℝ :=
  fun c b u => truncated width x c (finProdFinEquiv (b,u))

def binCoordinate (width : ℕ) (bin : Fin (T/width)) (offset : Fin width) : Fin T :=
  Fin.castLE (retained_not_longer T width) (finProdFinEquiv (bin,offset))

theorem bin_coordinate_value (width : ℕ) (bin : Fin (T/width)) (offset : Fin width) :
    (binCoordinate width bin offset).val = bin.val*width+offset.val := by
  simp [binCoordinate,finProdFinEquiv,Nat.mul_comm,Nat.add_comm]

theorem reshaped_coordinate (width : ℕ) (x : Epoch C T) (c : Fin C)
    (bin : Fin (T/width)) (offset : Fin width) :
    reshapeBins width x c bin offset = x c (binCoordinate width bin offset) := rfl

theorem reshaped_support (width : ℕ) (bin : Fin (T/width)) (offset : Fin width) :
    (binCoordinate width bin offset).val < retainedSamples T width :=
  (finProdFinEquiv (bin,offset)).isLt

theorem reshaped_bin_interval (width : ℕ) (bin : Fin (T/width)) (offset : Fin width) :
    bin.val*width ≤ (binCoordinate width bin offset).val ∧
    (binCoordinate width bin offset).val < (bin.val+1)*width := by
  rw [bin_coordinate_value]
  have hu := offset.isLt
  constructor
  · omega
  · simpa only [Nat.add_mul,Nat.one_mul] using Nat.add_lt_add_left hu (bin.val*width)

def completeBins (width : ℕ) (x : Epoch C T) : Epoch C (T/width) :=
  fun c b => loopMean width (reshapeBins width x c b)

theorem complete_bin_formula (width : ℕ) (x : Epoch C T) (c : Fin C) (bin : Fin (T/width)) :
    completeBins width x c bin = (∑ u, x c (binCoordinate width bin u)) / width := by
  rw [completeBins,loopMean_formula]
  rfl

theorem complete_bin_local_support (width : ℕ) (x y : Epoch C T) (c : Fin C)
    (bin : Fin (T/width))
    (h : ∀ t, bin.val*width ≤ t.val → t.val < (bin.val+1)*width → x c t=y c t) :
    completeBins width x c bin = completeBins width y c bin := by
  rw [complete_bin_formula,complete_bin_formula]
  congr 1
  apply Finset.sum_congr rfl
  intro u _
  exact h _ (reshaped_bin_interval width bin u).1 (reshaped_bin_interval width bin u).2

/-- Both inputs are typed finite arrays of this same fitted shape. -/
theorem unused_tail_noninterference (width : ℕ) (x y : Epoch C T)
    (h : ∀ c t, t.val < retainedSamples T width → x c t=y c t) :
    completeBins width x = completeBins width y := by
  funext c b
  rw [complete_bin_formula,complete_bin_formula]
  congr 1
  apply Finset.sum_congr rfl
  intro u _
  exact h c _ (reshaped_support width b u)

def combineEpoch (a b : ℝ) (x y : Epoch C T) : Epoch C T := fun c t => a*x c t+b*y c t

theorem completeBins_combine (width : ℕ) (a b : ℝ) (x y : Epoch C T) :
    completeBins width (combineEpoch a b x y) =
      combineEpoch a b (completeBins width x) (completeBins width y) := by
  funext c bin
  exact loopMean_combine width a b (reshapeBins width x c bin) (reshapeBins width y c bin)

/-- Time/bin first, channel fastest, cast to the source coefficient length C*B. -/
def sourceColumn : Fin B × Fin C ≃ Fin (C*B) :=
  finProdFinEquiv.trans (finCongr (Nat.mul_comm B C))

theorem source_column_value (bin : Fin B) (c : Fin C) :
    (sourceColumn (bin,c)).val = bin.val*C+c.val := by
  simp [sourceColumn,finProdFinEquiv,Nat.mul_comm,Nat.add_comm]

def flatten (x : Epoch C B) : Vector (C*B) := fun j =>
  let pair := sourceColumn.symm j
  x pair.2 pair.1

theorem flatten_source_coordinate (x : Epoch C B) (bin : Fin B) (c : Fin C) :
    flatten x (sourceColumn (bin,c)) = x c bin := by simp [flatten]

def binFeatures (width : ℕ) (x : Epoch C T) : Vector (C*(T/width)) := flatten (completeBins width x)

theorem bin_feature_formula (width : ℕ) (x : Epoch C T) (bin : Fin (T/width)) (c : Fin C) :
    binFeatures width x (sourceColumn (bin,c)) =
      (∑ u, x c (binCoordinate width bin u)) / width := by
  rw [binFeatures,flatten_source_coordinate,complete_bin_formula]

theorem flatten_combine (a b : ℝ) (x y : Epoch C B) :
    flatten (combineEpoch a b x y) = fun j => a*flatten x j+b*flatten y j := rfl

theorem bin_features_combine (width : ℕ) (a b : ℝ) (x y : Epoch C T) :
    binFeatures width (combineEpoch a b x y) =
      fun j => a*binFeatures width x j+b*binFeatures width y j := by
  rw [binFeatures,completeBins_combine]
  rfl

theorem bin_features_ignore_finite_unused_tail (width : ℕ) (x y : Epoch C T)
    (h : ∀ c t, t.val < retainedSamples T width → x c t=y c t) :
    binFeatures width x=binFeatures width y := by
  simpa only [binFeatures] using congrArg flatten (unused_tail_noninterference width x y h)

def widthOneBins (x : Epoch C T) : Epoch C T :=
  fun c t => completeBins 1 x c ((finCongr (Nat.div_one T)).symm t)

theorem width_one_bins_identity (x : Epoch C T) : widthOneBins x=x := by
  funext c t
  unfold widthOneBins
  rw [complete_bin_formula]
  simp only [Fin.sum_univ_one, Nat.cast_one, div_one]
  congr 1
  apply Fin.ext
  simp [bin_coordinate_value]

theorem flatten_injective : Function.Injective (flatten : Epoch C B → Vector (C*B)) := by
  intro x y h
  funext c t
  have he := congrFun h (sourceColumn (t,c))
  simpa only [flatten_source_coordinate] using he

def projectEpoch (basis : Matrix (Fin T) (Fin R) ℝ) (x : Epoch C T) : Epoch C R :=
  fun c r => ∑ t, x c t * basis t r

def projectBatch (basis : Matrix (Fin T) (Fin R) ℝ) (x : Batch N C T) : Batch N C R :=
  fun n => projectEpoch basis (x n)

theorem project_coordinate (basis : Matrix (Fin T) (Fin R) ℝ) (x : Epoch C T) (c : Fin C) (r : Fin R) :
    projectEpoch basis x c r = ∑ t, x c t*basis t r := rfl

theorem project_combine (basis : Matrix (Fin T) (Fin R) ℝ) (a b : ℝ) (x y : Epoch C T) :
    projectEpoch basis (combineEpoch a b x y) =
      combineEpoch a b (projectEpoch basis x) (projectEpoch basis y) := by
  funext c r
  simp only [projectEpoch,combineEpoch,add_mul,Finset.sum_add_distrib]
  simp_rw [mul_assoc,←Finset.mul_sum]

theorem projection_channel_noninterference (basis : Matrix (Fin T) (Fin R) ℝ)
    (x y : Epoch C T) (c : Fin C) (h : ∀ t, x c t=y c t) :
    ∀ r, projectEpoch basis x c r=projectEpoch basis y c r := by
  intro r
  simp only [projectEpoch,h]

theorem projection_local_temporal_support (basis : Matrix (Fin T) (Fin R) ℝ)
    (x y : Epoch C T) (c : Fin C) (r : Fin R)
    (h : ∀ t, basis t r ≠ 0 → x c t=y c t) :
    projectEpoch basis x c r=projectEpoch basis y c r := by
  apply Finset.sum_congr rfl
  intro t _
  by_cases hz : basis t r=0
  · simp [hz]
  · rw [h t hz]

def temporalFeatures (basis : Matrix (Fin T) (Fin R) ℝ) (x : Epoch C T) : Vector (C*R) :=
  flatten (projectEpoch basis x)

theorem temporal_feature_formula (basis : Matrix (Fin T) (Fin R) ℝ)
    (x : Epoch C T) (r : Fin R) (c : Fin C) :
    temporalFeatures basis x (sourceColumn (r,c)) = ∑ t, x c t*basis t r := by
  rw [temporalFeatures,flatten_source_coordinate]
  rfl

structure Affine (d : ℕ) where
  coefficient : Vector d
  intercept : ℝ

def evaluate {d : ℕ} (state : Affine d) (features : Vector d) : ℝ :=
  (∑ j, features j*state.coefficient j)+state.intercept

def batchEvaluate {d : ℕ} (state : Affine d) (features : Fin N → Vector d) : Fin N → ℝ :=
  fun n => evaluate state (features n)

theorem flatten_affine_formula (state : Affine (C*B)) (x : Epoch C B) :
    evaluate state (flatten x) =
      (∑ t, ∑ c, x c t*state.coefficient (sourceColumn (t,c)))+state.intercept := by
  unfold evaluate
  congr 1
  rw [←Equiv.sum_comp sourceColumn]
  simp [Fintype.sum_prod_type,flatten_source_coordinate]

theorem binned_affine_formula (width : ℕ) (state : Affine (C*(T/width))) (x : Epoch C T) :
    evaluate state (binFeatures width x) =
      (∑ t, ∑ c, ((∑ u, x c (binCoordinate width t u)) / width)*
        state.coefficient (sourceColumn (t,c)))+state.intercept := by
  rw [binFeatures,flatten_affine_formula]
  simp only [complete_bin_formula]

def effectiveKernel (basis : Matrix (Fin T) (Fin R) ℝ) (state : Affine (C*R)) : Epoch C T :=
  fun c t => ∑ r, basis t r*state.coefficient (sourceColumn (r,c))

theorem temporal_affine_effective_kernel (basis : Matrix (Fin T) (Fin R) ℝ)
    (state : Affine (C*R)) (x : Epoch C T) :
    evaluate state (temporalFeatures basis x) =
      (∑ c, ∑ t, x c t*effectiveKernel basis state c t)+state.intercept := by
  rw [temporalFeatures,flatten_affine_formula]
  congr 1
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro c _
  simp only [projectEpoch,Finset.sum_mul,effectiveKernel,Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro t _
  apply Finset.sum_congr rfl
  intro r _
  ring

theorem evaluate_combine (state : Affine B) (a b : ℝ) (x y : Vector B) :
    evaluate state (fun j => a*x j+b*y j) =
      a*(evaluate state x-state.intercept)+b*(evaluate state y-state.intercept)+state.intercept := by
  simp only [evaluate,add_mul,Finset.sum_add_distrib]
  simp_rw [mul_assoc,←Finset.mul_sum]
  ring

theorem batch_evaluation_coordinate (state : Affine B) (x : Fin N → Vector B) (n : Fin N) :
    batchEvaluate state x n=evaluate state (x n) := rfl

theorem empty_batch_has_no_prediction_rows (state : Affine B) (x : Fin 0 → Vector B) :
    batchEvaluate state x = (fun n => Fin.elim0 n) := by
  funext n
  exact Fin.elim0 n

def fixtureEpoch : Epoch 2 5 := ![![1,3,5,7,999],![2,4,6,8,-999]]
def changedFiniteTail : Epoch 2 5 := ![![1,3,5,7,-73],![2,4,6,8,42]]
def fixtureState : Affine 4 := ⟨![1,2,3,4],-1⟩

theorem nonempty_fixture_admitted : InputAdmitted 2 5 2 := by norm_num [InputAdmitted]

theorem fixture_discards_one_sample : retainedSamples 5 2=4 ∧ 5%2=1 := by norm_num [retainedSamples]

theorem fixture_bin_values : completeBins 2 fixtureEpoch = ![![2,6],![3,7]] := by
  funext c b
  fin_cases c <;> fin_cases b <;>
    norm_num [completeBins,loopMean,sumLoop,reshapeBins,truncated,fixtureEpoch,finProdFinEquiv,
      Fin.castLE,retainedSamples]

theorem fixture_time_major_channel_prime_features :
    binFeatures 2 fixtureEpoch = ![2,3,6,7] := by
  funext j
  fin_cases j <;>
    norm_num [binFeatures,fixture_bin_values,flatten,sourceColumn,finProdFinEquiv,Fin.divNat,Fin.modNat]

theorem fixture_saved_score_53 : evaluate fixtureState (binFeatures 2 fixtureEpoch)=53 := by
  rw [fixture_time_major_channel_prime_features]
  norm_num [evaluate,fixtureState,Fin.sum_univ_succ]

/-- This separately defined wrong order is the waveform LDA convention. -/
def wrongChannelMajor (x : Epoch C B) : Vector (C*B) := fun j =>
  let pair := finProdFinEquiv.symm j
  x pair.1 pair.2

theorem fixture_wrong_order_features : wrongChannelMajor (completeBins 2 fixtureEpoch)=![2,6,3,7] := by
  rw [fixture_bin_values]
  funext j
  fin_cases j <;> norm_num [wrongChannelMajor,finProdFinEquiv,Fin.divNat,Fin.modNat]

theorem fixture_wrong_order_score_50 :
    evaluate fixtureState (wrongChannelMajor (completeBins 2 fixtureEpoch))=50 := by
  rw [fixture_wrong_order_features]
  norm_num [evaluate,fixtureState,Fin.sum_univ_succ]

theorem nonempty_order_changes_prediction :
    evaluate fixtureState (binFeatures 2 fixtureEpoch) ≠
      evaluate fixtureState (wrongChannelMajor (completeBins 2 fixtureEpoch)) := by
  rw [fixture_saved_score_53,fixture_wrong_order_score_50]
  norm_num

theorem fixture_finite_tail_change_keeps_bins : completeBins 2 changedFiniteTail=completeBins 2 fixtureEpoch := by
  apply unused_tail_noninterference 2 changedFiniteTail fixtureEpoch
  intro c t ht
  have hv : t.val<4 := by simpa [retainedSamples] using ht
  fin_cases c <;> fin_cases t <;> norm_num [changedFiniteTail,fixtureEpoch] at *

theorem fixture_finite_tail_change_keeps_score : evaluate fixtureState (binFeatures 2 changedFiniteTail)=53 := by
  rw [binFeatures,fixture_finite_tail_change_keeps_bins]
  exact fixture_saved_score_53

theorem fixture_doubled_signal_score_107 :
    evaluate fixtureState (binFeatures 2 (combineEpoch 1 1 fixtureEpoch fixtureEpoch))=107 := by
  rw [bin_features_combine,evaluate_combine,fixture_saved_score_53]
  norm_num [fixtureState]

def temporalFixtureEpoch : Epoch 2 2 := ![![1,2],![3,4]]
def temporalFixtureBasis : Matrix (Fin 2) (Fin 1) ℝ := ![![0],![1]]
def temporalFixtureState : Affine 2 := ⟨![2,3],1⟩

theorem nonempty_temporal_basis_is_exact_orthonormal :
    ∀ r s : Fin 1, (∑ t, temporalFixtureBasis t r*temporalFixtureBasis t s)=if r=s then 1 else 0 := by
  intro r s
  fin_cases r
  fin_cases s
  norm_num [temporalFixtureBasis,Fin.sum_univ_two]

theorem nonempty_temporal_projection_score_17 :
    evaluate temporalFixtureState (temporalFeatures temporalFixtureBasis temporalFixtureEpoch)=17 := by
  have he : temporalFeatures temporalFixtureBasis temporalFixtureEpoch=![2,4] := by
    funext j
    fin_cases j <;> norm_num [temporalFeatures,flatten,sourceColumn,finProdFinEquiv,
      projectEpoch,temporalFixtureBasis,temporalFixtureEpoch,Fin.sum_univ_two,Fin.divNat,Fin.modNat]
  rw [he]
  norm_num [evaluate,temporalFixtureState,Fin.sum_univ_two]

end

end N2P3.ERPFeatures
