import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Data.Real.StarOrdered
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.FinCases

/-!
# The source participant estimator for temporal ERP transfer

Source: baselines/temporal_transfer.py:146-216.  Each admitted participant
contributes a class-1 minus class-0 contrast on that participant's observed
channels, then D.transpose * D / trace(D.transpose * D).  The estimator averages
these matrices with equal PARTICIPANT weight.  Channel counts may differ.

This exact-real layer starts after the actual global dataset/row/event/identity
guards.  Bool labels represent the already checked two labels.  The source
requires at least two trials in each class; a positive energy is its explicit
admission guard.  The formulas below do not infer that guard from arbitrary
input data.  No eigenvector selection, numerical eigh correctness, uniqueness
at a tied cutoff, or physical source/target authorization follows from PSD.

The last section spells out the numerical orthogonality inequality used by
from_record (rtol=atol=1e-10 against I).  It is not exact orthogonality.
Real arithmetic models the declared formulas, not NumPy rounding, overflow,
dtype conversion, bytes, metadata authentication or source replay.
-/

namespace N2P3.ERPTemporalBasis

open Matrix
open scoped BigOperators

noncomputable section

def classRows {N : ℕ} (labels : Fin N → Bool) (label : Bool) : Finset (Fin N) :=
  Finset.univ.filter (fun i => labels i = label)

def classMean {N C T : ℕ} (X : Fin N → Matrix (Fin C) (Fin T) ℝ)
    (labels : Fin N → Bool) (label : Bool) : Matrix (Fin C) (Fin T) ℝ :=
  fun c t => (∑ i ∈ classRows labels label, X i c t) /
    ((classRows labels label).card : ℝ)

def classContrast {N C T : ℕ} (X : Fin N → Matrix (Fin C) (Fin T) ℝ)
    (labels : Fin N → Bool) : Matrix (Fin C) (Fin T) ℝ :=
  classMean X labels true - classMean X labels false

theorem class_contrast_coordinate {N C T : ℕ}
    (X : Fin N → Matrix (Fin C) (Fin T) ℝ) (labels : Fin N → Bool)
    (c : Fin C) (t : Fin T) :
    classContrast X labels c t =
      (∑ i ∈ classRows labels true, X i c t) /
        ((classRows labels true).card : ℝ) -
      (∑ i ∈ classRows labels false, X i c t) /
        ((classRows labels false).card : ℝ) := rfl

theorem class_mean_depends_on_selected_rows {N M C T : ℕ}
    (X Y : Fin M → Matrix (Fin C) (Fin T) ℝ) (rows : Fin N → Fin M)
    (labels : Fin N → Bool) (label : Bool)
    (h : ∀ i c t, X (rows i) c t = Y (rows i) c t) :
    classMean (fun i => X (rows i)) labels label =
      classMean (fun i => Y (rows i)) labels label := by
  ext c t
  unfold classMean
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  exact h i c t

theorem class_contrast_depends_on_selected_rows {N M C T : ℕ}
    (X Y : Fin M → Matrix (Fin C) (Fin T) ℝ) (rows : Fin N → Fin M)
    (labels : Fin N → Bool)
    (h : ∀ i c t, X (rows i) c t = Y (rows i) c t) :
    classContrast (fun i => X (rows i)) labels =
      classContrast (fun i => Y (rows i)) labels := by
  unfold classContrast
  rw [class_mean_depends_on_selected_rows X Y rows labels true h,
    class_mean_depends_on_selected_rows X Y rows labels false h]

def gram {C T : ℕ} (D : Matrix (Fin C) (Fin T) ℝ) :
    Matrix (Fin T) (Fin T) ℝ := Dᵀ * D

def energy {C T : ℕ} (D : Matrix (Fin C) (Fin T) ℝ) : ℝ := (gram D).trace

def normalizedGram {C T : ℕ} (D : Matrix (Fin C) (Fin T) ℝ) :
    Matrix (Fin T) (Fin T) ℝ := (energy D)⁻¹ • gram D

theorem gram_coordinate {C T : ℕ} (D : Matrix (Fin C) (Fin T) ℝ)
    (t u : Fin T) : gram D t u = ∑ c, D c t * D c u := rfl

theorem gram_symmetric {C T : ℕ} (D : Matrix (Fin C) (Fin T) ℝ) :
    (gram D)ᵀ = gram D := by
  ext t u
  simp only [Matrix.transpose_apply, gram_coordinate]
  apply Finset.sum_congr rfl
  intro c hc
  exact mul_comm _ _

theorem gram_posSemidef {C T : ℕ} (D : Matrix (Fin C) (Fin T) ℝ) :
    (gram D).PosSemidef := by
  simpa [gram, Matrix.conjTranspose_eq_transpose_of_trivial] using
    Matrix.posSemidef_conjTranspose_mul_self D

theorem energy_is_sum_of_squares {C T : ℕ} (D : Matrix (Fin C) (Fin T) ℝ) :
    energy D = ∑ t, ∑ c, D c t ^ 2 := by
  simp [energy, Matrix.trace, gram_coordinate, pow_two]

theorem energy_nonnegative {C T : ℕ} (D : Matrix (Fin C) (Fin T) ℝ) :
    0 ≤ energy D := by
  rw [energy_is_sum_of_squares]
  exact Finset.sum_nonneg fun t _ => Finset.sum_nonneg fun c _ => sq_nonneg _

theorem normalized_gram_coordinate {C T : ℕ} (D : Matrix (Fin C) (Fin T) ℝ)
    (t u : Fin T) : normalizedGram D t u = (∑ c, D c t * D c u) / energy D := by
  simp only [normalizedGram, Matrix.smul_apply, smul_eq_mul, gram_coordinate, div_eq_mul_inv]
  exact mul_comm _ _

theorem normalized_gram_posSemidef {C T : ℕ} (D : Matrix (Fin C) (Fin T) ℝ)
    (h : 0 < energy D) : (normalizedGram D).PosSemidef :=
  (gram_posSemidef D).smul (inv_nonneg.mpr h.le)

theorem normalized_gram_trace {C T : ℕ} (D : Matrix (Fin C) (Fin T) ℝ)
    (h : 0 < energy D) : (normalizedGram D).trace = 1 := by
  rw [normalizedGram, Matrix.trace_smul]
  change (energy D)⁻¹ * energy D = 1
  exact inv_mul_cancel₀ (ne_of_gt h)

theorem gram_scale {C T : ℕ} (a : ℝ) (D : Matrix (Fin C) (Fin T) ℝ) :
    gram (a • D) = a ^ 2 • gram D := by
  ext t u
  simp only [gram_coordinate, Matrix.smul_apply, smul_eq_mul, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro c hc
  ring

theorem energy_scale {C T : ℕ} (a : ℝ) (D : Matrix (Fin C) (Fin T) ℝ) :
    energy (a • D) = a ^ 2 * energy D := by
  simp [energy, gram_scale, Matrix.trace_smul]

theorem normalized_gram_scale_invariant {C T : ℕ} (a : ℝ)
    (D : Matrix (Fin C) (Fin T) ℝ) (ha : a ≠ 0) (hD : energy D ≠ 0) :
    normalizedGram (a • D) = normalizedGram D := by
  ext t u
  simp only [normalizedGram, energy_scale, gram_scale, Matrix.smul_apply, smul_eq_mul]
  field_simp

def participantAverage {P : Type*} [Fintype P] {T : ℕ} (C : P → ℕ)
    (D : (p : P) → Matrix (Fin (C p)) (Fin T) ℝ) :
    Matrix (Fin T) (Fin T) ℝ :=
  (Fintype.card P : ℝ)⁻¹ • ∑ p, normalizedGram (D p)

theorem participant_average_coordinate {P : Type*} [Fintype P] {T : ℕ}
    (C : P → ℕ) (D : (p : P) → Matrix (Fin (C p)) (Fin T) ℝ) (t u : Fin T) :
    participantAverage C D t u =
      (∑ p, (∑ c, D p c t * D p c u) / energy (D p)) / (Fintype.card P : ℝ) := by
  simp [participantAverage, Matrix.smul_apply, Matrix.sum_apply,
    normalized_gram_coordinate, div_eq_mul_inv, mul_comm]

theorem finite_sum_posSemidef {P : Type*} [DecidableEq P] {T : ℕ}
    (s : Finset P) (A : P → Matrix (Fin T) (Fin T) ℝ)
    (h : ∀ p ∈ s, (A p).PosSemidef) : (∑ p ∈ s, A p).PosSemidef := by
  induction s using Finset.induction_on with
  | empty => simpa using (Matrix.PosSemidef.zero : (0 : Matrix (Fin T) (Fin T) ℝ).PosSemidef)
  | @insert p s hp ih =>
    rw [Finset.sum_insert hp]
    exact (h p (Finset.mem_insert_self _ _)).add
      (ih (fun q hq => h q (Finset.mem_insert_of_mem hq)))

theorem participant_average_posSemidef {P : Type*} [Fintype P] {T : ℕ}
    (C : P → ℕ) (D : (p : P) → Matrix (Fin (C p)) (Fin T) ℝ)
    (h : ∀ p, 0 < energy (D p)) : (participantAverage C D).PosSemidef := by
  classical
  apply Matrix.PosSemidef.smul
  · exact finite_sum_posSemidef Finset.univ _ (fun p _ => normalized_gram_posSemidef _ (h p))
  · exact inv_nonneg.mpr (Nat.cast_nonneg _)

theorem participant_average_trace {P : Type*} [Fintype P] [Nonempty P] {T : ℕ}
    (C : P → ℕ) (D : (p : P) → Matrix (Fin (C p)) (Fin T) ℝ)
    (h : ∀ p, 0 < energy (D p)) : (participantAverage C D).trace = 1 := by
  rw [participantAverage, Matrix.trace_smul, Matrix.trace_sum]
  simp only [normalized_gram_trace _ (h _), Finset.sum_const, Finset.card_univ,
    nsmul_eq_mul, mul_one, smul_eq_mul]
  exact inv_mul_cancel₀ (Nat.cast_ne_zero.mpr (Fintype.card_ne_zero))

/-- The estimator's actual class-contrast-to-normalized-Gram composition.
The row/event/identity/mask guards are upstream; no fitted matrix is a premise. -/
def sourceEstimator {P : Type*} [Fintype P] {T : ℕ} (N C : P → ℕ)
    (X : (p : P) → Fin (N p) → Matrix (Fin (C p)) (Fin T) ℝ)
    (labels : (p : P) → Fin (N p) → Bool) : Matrix (Fin T) (Fin T) ℝ :=
  participantAverage C (fun p => classContrast (X p) (labels p))

theorem source_estimator_coordinate {P : Type*} [Fintype P] {T : ℕ} (N C : P → ℕ)
    (X : (p : P) → Fin (N p) → Matrix (Fin (C p)) (Fin T) ℝ)
    (labels : (p : P) → Fin (N p) → Bool) (t u : Fin T) :
    sourceEstimator N C X labels t u =
      (∑ p, (∑ c, classContrast (X p) (labels p) c t *
        classContrast (X p) (labels p) c u) /
        energy (classContrast (X p) (labels p))) / (Fintype.card P : ℝ) :=
  participant_average_coordinate _ _ _ _

theorem source_estimator_posSemidef {P : Type*} [Fintype P] {T : ℕ} (N C : P → ℕ)
    (X : (p : P) → Fin (N p) → Matrix (Fin (C p)) (Fin T) ℝ)
    (labels : (p : P) → Fin (N p) → Bool)
    (h : ∀ p, 0 < energy (classContrast (X p) (labels p))) :
    (sourceEstimator N C X labels).PosSemidef := participant_average_posSemidef _ _ h

theorem source_estimator_trace_one {P : Type*} [Fintype P] [Nonempty P] {T : ℕ}
    (N C : P → ℕ) (X : (p : P) → Fin (N p) → Matrix (Fin (C p)) (Fin T) ℝ)
    (labels : (p : P) → Fin (N p) → Bool)
    (h : ∀ p, 0 < energy (classContrast (X p) (labels p))) :
    (sourceEstimator N C X labels).trace = 1 := participant_average_trace _ _ h

def project {N C T R : ℕ} (basis : Matrix (Fin T) (Fin R) ℝ)
    (X : Fin N → Matrix (Fin C) (Fin T) ℝ) :
    Fin N → Matrix (Fin C) (Fin R) ℝ := fun i => X i * basis

theorem project_coordinate {N C T R : ℕ} (basis : Matrix (Fin T) (Fin R) ℝ)
    (X : Fin N → Matrix (Fin C) (Fin T) ℝ) (i : Fin N) (c : Fin C) (r : Fin R) :
    project basis X i c r = ∑ t, X i c t * basis t r := rfl

theorem project_add {N C T R : ℕ} (basis : Matrix (Fin T) (Fin R) ℝ)
    (X Y : Fin N → Matrix (Fin C) (Fin T) ℝ) :
    project basis (X + Y) = project basis X + project basis Y := by
  funext i
  exact Matrix.add_mul _ _ _

def approximateOrthonormal {T R : ℕ} (basis : Matrix (Fin T) (Fin R) ℝ) : Prop :=
  ∀ i j, |gram basis i j - (1 : Matrix (Fin R) (Fin R) ℝ) i j| ≤
    (1 / 10000000000 : ℝ) +
      (1 / 10000000000 : ℝ) * |(1 : Matrix (Fin R) (Fin R) ℝ) i j|

/-! Nonempty source-estimator instances and independent boundary witnesses. -/

def fixtureChannels (p : Fin 2) : ℕ := if p = 0 then 1 else 2

def fixtureContrast (p : Fin 2) : Matrix (Fin (fixtureChannels p)) (Fin 3) ℝ :=
  fun c t => if p = 0 then (if t = 0 then 2 else 0)
    else (if t.val = c.val + 1 then 3 else 0)

theorem fixture_source_energies :
    energy (fixtureContrast 0) = 4 ∧ energy (fixtureContrast 1) = 18 := by
  constructor
  · change energy (fun (_ : Fin 1) (t : Fin 3) => if t = 0 then 2 else 0) = 4
    norm_num [energy_is_sum_of_squares, Fin.sum_univ_succ]
  · change energy (fun (c : Fin 2) (t : Fin 3) => if t.val = c.val + 1 then 3 else 0) = 18
    norm_num [energy_is_sum_of_squares, Fin.sum_univ_succ]

theorem fixture_positive_energy : ∀ p, 0 < energy (fixtureContrast p) := by
  intro p
  fin_cases p
  · change 0 < energy (fixtureContrast (0 : Fin 2))
    rw [fixture_source_energies.1]; norm_num
  · change 0 < energy (fixtureContrast (1 : Fin 2))
    rw [fixture_source_energies.2]; norm_num

theorem fixture_first_normalized_gram :
    normalizedGram (fixtureContrast 0) = Matrix.diagonal (![1, 0, 0] : Fin 3 → ℝ) := by
  change normalizedGram (fun (_ : Fin 1) (t : Fin 3) => if t = 0 then 2 else 0) = _
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [normalized_gram_coordinate, energy_is_sum_of_squares, Fin.sum_univ_succ,
      Matrix.diagonal]

theorem fixture_second_normalized_gram :
    normalizedGram (fixtureContrast 1) = Matrix.diagonal (![0, 1/2, 1/2] : Fin 3 → ℝ) := by
  change normalizedGram (fun (c : Fin 2) (t : Fin 3) => if t.val = c.val + 1 then 3 else 0) = _
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [normalized_gram_coordinate, energy_is_sum_of_squares, Fin.sum_univ_succ,
      Matrix.diagonal]

theorem fixture_source_average :
    participantAverage fixtureChannels fixtureContrast =
      Matrix.diagonal (![1/2, 1/4, 1/4] : Fin 3 → ℝ) := by
  simp only [participantAverage, Fintype.card_fin, Fin.sum_univ_two,
    fixture_first_normalized_gram, fixture_second_normalized_gram]
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [Matrix.diagonal]

theorem fixture_source_average_posSemidef :
    (participantAverage fixtureChannels fixtureContrast).PosSemidef :=
  participant_average_posSemidef _ _ fixture_positive_energy

theorem fixture_source_average_trace_one :
    (participantAverage fixtureChannels fixtureContrast).trace = 1 :=
  participant_average_trace _ _ fixture_positive_energy

def fixtureLabels : Fin 4 → Bool := fun i => decide (2 ≤ i.val)

def fixtureTrials (p : Fin 2) : Fin 4 → Matrix (Fin (fixtureChannels p)) (Fin 3) ℝ :=
  fun i c t => if 2 ≤ i.val then fixtureContrast p c t else 0

theorem fixture_class_rows :
    classRows fixtureLabels false = {0, 1} ∧ classRows fixtureLabels true = {2, 3} := by
  decide

theorem fixture_has_two_trials_per_class :
    (classRows fixtureLabels false).card = 2 ∧ (classRows fixtureLabels true).card = 2 := by
  simp [fixture_class_rows.1, fixture_class_rows.2]

theorem fixture_trials_produce_actual_contrasts (p : Fin 2) :
    classContrast (fixtureTrials p) fixtureLabels = fixtureContrast p := by
  ext c t
  simp [classContrast, classMean, fixture_class_rows.1, fixture_class_rows.2, fixtureTrials]

theorem nonempty_raw_trials_source_estimator :
    sourceEstimator (fun _ : Fin 2 => 4) fixtureChannels fixtureTrials
      (fun _ => fixtureLabels) = Matrix.diagonal (![1/2, 1/4, 1/4] : Fin 3 → ℝ) := by
  unfold sourceEstimator
  simp_rw [fixture_trials_produce_actual_contrasts]
  exact fixture_source_average

def tiedContrasts (p : Fin 2) : Matrix (Fin 1) (Fin 2) ℝ :=
  fun _ t => if p = t then 1 else 0

theorem tied_source_energies : ∀ p, energy (tiedContrasts p) = 1 := by
  intro p
  fin_cases p <;> norm_num [energy_is_sum_of_squares, tiedContrasts, Fin.sum_univ_succ]

theorem tied_source_average :
    participantAverage (fun _ : Fin 2 => 1) tiedContrasts =
      (1/2 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [participant_average_coordinate, tiedContrasts, tied_source_energies,
      Fin.sum_univ_succ, Matrix.smul_apply, Matrix.one_apply]

def unitAxis (i : Fin 2) : Fin 2 → ℝ := fun j => if i = j then 1 else 0

theorem tied_axes_are_distinct : unitAxis 0 ≠ unitAxis 1 := by
  intro h
  have h0 := congrFun h 0
  norm_num [unitAxis] at h0

theorem tied_axes_have_equal_positive_eigenvalue (i : Fin 2) :
    (participantAverage (fun _ : Fin 2 => 1) tiedContrasts) *ᵥ unitAxis i =
      (1/2 : ℝ) • unitAxis i := by
  rw [tied_source_average, Matrix.smul_mulVec, Matrix.one_mulVec]

def nearlyUnitBasis : Matrix (Fin 1) (Fin 1) ℝ :=
  fun _ _ => 100000000001 / 100000000000

theorem nearly_unit_passes_tolerance : approximateOrthonormal nearlyUnitBasis := by
  intro i j
  fin_cases i
  fin_cases j
  norm_num [approximateOrthonormal, nearlyUnitBasis, gram_coordinate, Fin.sum_univ_succ]

theorem nearly_unit_not_exactly_orthogonal :
    gram nearlyUnitBasis ≠ (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  intro h
  have h0 := congrFun (congrFun h 0) 0
  norm_num [gram_coordinate, nearlyUnitBasis, Fin.sum_univ_succ] at h0

end

end N2P3.ERPTemporalBasis
