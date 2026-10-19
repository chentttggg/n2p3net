import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Data.Matrix.Block
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# N2P3-Net mathematical pilot

These are real-arithmetic specifications of selected source expressions, not a
verification of Python, BLAS, floating-point arithmetic, or the whole pipeline.
Finite index types represent actual aligned coordinates. No dimensions or input
data are numerically enumerated by the main theorems.
-/

namespace N2P3Pilot

open Matrix
open scoped BigOperators Matrix ComplexOrder

section Covariance

variable {n r m : Type*} [Fintype n] [Fintype r] [Fintype m]

/-- The exact coordinate concatenation used by np.column_stack([A, B]). -/
def horizontal (A : Matrix m n ℝ) (B : Matrix m r ℝ) : Matrix m (n ⊕ r) ℝ :=
  fun i => Sum.elim (A i) (B i)

omit [Fintype m] in
theorem horizontal_mul_block
    (A : Matrix m n ℝ) (B : Matrix m r ℝ)
    (G : Matrix n n ℝ) (C : Matrix n r ℝ) (D : Matrix r n ℝ) (O : Matrix r r ℝ) :
    horizontal A B * fromBlocks G C D O =
      horizontal (A * G + B * D) (A * C + B * O) := by
  ext i j
  cases j <;> simp [Matrix.mul_apply, horizontal, Fintype.sum_sum_type]

omit [Fintype m] in
theorem horizontal_mul_transpose
    (A C : Matrix m n ℝ) (B D : Matrix m r ℝ) :
    horizontal A B * (horizontal C D)ᵀ = A * Cᵀ + B * Dᵀ := by
  ext i j
  simp [Matrix.mul_apply, horizontal, Fintype.sum_sum_type]

omit [Fintype m] in
/-- The full raw/state covariance has both cross terms; independence is not assumed. -/
theorem joint_covariance_cross_terms
    (A : Matrix m n ℝ) (B : Matrix m r ℝ)
    (G : Matrix n n ℝ) (C : Matrix n r ℝ) (O : Matrix r r ℝ) :
    horizontal A B * fromBlocks G C Cᵀ O * (horizontal A B)ᵀ =
      A * G * Aᵀ + B * Cᵀ * Aᵀ + (A * C * Bᵀ + B * O * Bᵀ) := by
  rw [horizontal_mul_block, horizontal_mul_transpose, Matrix.add_mul, Matrix.add_mul]

/-- Any fixed linear operator preserves positive SEMI-definiteness. -/
theorem covariance_pushforward_psd
    (S : Matrix n n ℝ) (A : Matrix m n ℝ) (hS : S.PosSemidef) :
    (A * S * Aᵀ).PosSemidef := by
  simpa only [conjTranspose_eq_transpose_of_trivial] using
    hS.mul_mul_conjTranspose_same A

omit [Fintype n] in
/-- The final symmetric averaging in JointNoiseModel.project is exact over reals. -/
theorem symmetrization_preserves_symmetric
    (S : Matrix n n ℝ) (hS : Sᵀ = S) :
    (1 / 2 : ℝ) • (S + Sᵀ) = S := by
  rw [hS]
  ext i j
  simp
  ring

/-- A nonzero projection may destroy positive definiteness: baseline-like rank loss. -/
theorem projection_can_destroy_pd :
    ∃ (S A : Matrix (Fin 2) (Fin 2) ℝ),
      S.PosDef ∧ A ≠ 0 ∧ (A * S * Aᵀ).PosSemidef ∧ ¬(A * S * Aᵀ).PosDef := by
  let A : Matrix (Fin 2) (Fin 2) ℝ := diagonal ![1, 0]
  have hA : A * (1 : Matrix (Fin 2) (Fin 2) ℝ) * Aᵀ = A := by
    ext i j
    fin_cases i <;> fin_cases j <;> norm_num [A, Matrix.mul_apply, Fin.sum_univ_two]
  refine ⟨1, A, Matrix.PosDef.one, ?_, ?_, ?_⟩
  · intro h
    have := congrArg (fun M : Matrix (Fin 2) (Fin 2) ℝ => M 0 0) h
    norm_num [A] at this
  · exact covariance_pushforward_psd _ _ Matrix.PosSemidef.one
  · rw [hA]
    intro h
    have := h.2 (![0, 1] : Fin 2 → ℝ) (by simp)
    norm_num [A, Matrix.mulVec, dotProduct, Fin.sum_univ_two] at this

end Covariance

section Evidence

variable {ι κ : Type*} [Fintype ι]

/-- Real specification of membership_evidence(..., aggregation="sum"). -/
def membershipScore (M : Matrix ι κ ℝ) (available : ι → Bool)
    (evidence offset : ι → ℝ) (candidate : κ) : ℝ :=
  ∑ i, M i candidate * (if available i then evidence i - offset i else 0)

/-- Arbitrary changes to unavailable values cannot influence a sum score. -/
theorem unavailable_evidence_irrelevant
    (M : Matrix ι κ ℝ) (available : ι → Bool) (x y offset : ι → ℝ)
    (h : ∀ i, available i = true → x i = y i) (candidate : κ) :
    membershipScore M available x offset candidate =
      membershipScore M available y offset candidate := by
  apply Finset.sum_congr rfl
  intro i _
  by_cases hi : available i = true
  · simp [hi, h i hi]
  · simp [hi]

/-- Subtracting a common offset subtracts offset times the observed membership count. -/
theorem scalar_offset_formula
    (M : Matrix ι κ ℝ) (available : ι → Bool) (x : ι → ℝ)
    (offset : ℝ) (candidate : κ) :
    membershipScore M available x (fun _ => offset) candidate =
      membershipScore M available x (fun _ => 0) candidate -
        offset * (∑ i, M i candidate * (if available i then 1 else 0)) := by
  unfold membershipScore
  rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _
  by_cases hi : available i = true
  · simp [hi]
    ring
  · simp [hi]

/-- Equal observed counts are sufficient for offset-invariant pairwise differences. -/
theorem equal_counts_preserve_difference
    (M : Matrix ι κ ℝ) (available : ι → Bool) (x : ι → ℝ)
    (offset : ℝ) (a b : κ)
    (hcount : (∑ i, M i a * (if available i then 1 else 0)) =
      ∑ i, M i b * (if available i then 1 else 0)) :
    membershipScore M available x (fun _ => offset) a -
      membershipScore M available x (fun _ => offset) b =
    membershipScore M available x (fun _ => 0) a -
      membershipScore M available x (fun _ => 0) b := by
  rw [scalar_offset_formula M available x offset a,
    scalar_offset_formula M available x offset b, hcount]
  ring

/-- Unequal counts permit a strict rank reversal, even with binary memberships. -/
theorem unequal_counts_offset_counterexample :
    ∃ (M : Matrix (Fin 3) (Fin 2) ℝ) (x : Fin 3 → ℝ),
      (∀ i j, M i j = 0 ∨ M i j = 1) ∧
      membershipScore M (fun _ => true) x (fun _ => 0) 0 >
        membershipScore M (fun _ => true) x (fun _ => 0) 1 ∧
      membershipScore M (fun _ => true) x (fun _ => 2) 0 <
        membershipScore M (fun _ => true) x (fun _ => 2) 1 := by
  refine ⟨!![1, 0; 1, 0; 0, 1], ![1, 1, 1], ?_, ?_, ?_⟩
  · intro i j
    fin_cases i <;> fin_cases j <;> norm_num
  · norm_num [membershipScore, Fin.sum_univ_succ]
  · norm_num [membershipScore, Fin.sum_univ_succ]

end Evidence

section TimeSupport

/-- Source uses start >= boundary: touching half-open supports are disjoint. -/
theorem half_open_supports_disjoint
    (calStart calStop testStart testStop : ℝ) (h : calStop ≤ testStart) :
    Disjoint (Set.Ico calStart calStop) (Set.Ico testStart testStop) := by
  apply Set.disjoint_left.mpr
  intro x hx hy
  exact (not_lt_of_ge (le_trans h hy.1)) hx.2

/-- This is the membership predicate of binary_time_split, over exact times. -/
def inTestSuffix (maxCalibration row : ℕ) (onset tmin boundary : ℝ) : Prop :=
  maxCalibration ≤ row ∧ boundary ≤ onset + tmin

/-- Any accepted test row is outside every smaller calibration prefix. -/
theorem accepted_test_not_calibration
    (k maxCalibration row : ℕ) (onset tmin boundary : ℝ)
    (hk : k ≤ maxCalibration)
    (htest : inTestSuffix maxCalibration row onset tmin boundary) : ¬row < k := by
  exact Nat.not_lt.mpr (le_trans hk htest.1)

/-- The max-availability boundary separates each authorized epoch from accepted test support. -/
theorem accepted_test_support_separated
    (calStart calAvailable onset tmin testStop boundary : ℝ)
    (maxCalibration row : ℕ) (hcal : calAvailable ≤ boundary)
    (htest : inTestSuffix maxCalibration row onset tmin boundary) :
    Disjoint (Set.Ico calStart calAvailable) (Set.Ico (onset + tmin) testStop) := by
  exact half_open_supports_disjoint _ _ _ _ (le_trans hcal htest.2)

/-- Strict-overlap test for the half-open rest/epoch intervals in the input contract. -/
theorem half_open_overlap_iff
    (a b c d : ℝ) (hab : a < b) (hcd : c < d) :
    (∃ x, x ∈ Set.Ico a b ∧ x ∈ Set.Ico c d) ↔ a < d ∧ c < b := by
  constructor
  · rintro ⟨x, hx, hy⟩
    exact ⟨lt_of_le_of_lt hx.1 hy.2, lt_of_le_of_lt hy.1 hx.2⟩
  · rintro ⟨had, hcb⟩
    refine ⟨max a c, ⟨le_max_left _ _, max_lt hab hcb⟩,
      ⟨le_max_right _ _, max_lt had hcd⟩⟩

end TimeSupport

end N2P3Pilot
