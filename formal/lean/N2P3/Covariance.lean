import N2P3Pilot
import Mathlib.Probability.Moments.Covariance
import Mathlib.Analysis.CStarAlgebra.Matrix

/-!
# Actual covariance and operator error propagation

Random coordinates have finite second moments. No independence of raw samples,
epochs, or raw/state coordinates is assumed. Numerical bounds below are
conditional on certified real-matrix input/residual bounds, not a proof of
NumPy's eigenvalue or floating-point implementation.
-/

namespace N2P3.Covariance

open Matrix MeasureTheory ProbabilityTheory
open scoped BigOperators ProbabilityTheory Matrix ComplexOrder InnerProductSpace Matrix.Norms.L2Operator

section RandomVariables

variable {Ω n m r : Type*} [MeasurableSpace Ω]
variable [Fintype n] [Fintype m] [Fintype r]
variable (μ : Measure Ω) [IsProbabilityMeasure μ]

noncomputable def covarianceMatrix (X : n → Ω → ℝ) : Matrix n n ℝ :=
  fun i j => cov[X i, X j; μ]

noncomputable def crossCovarianceMatrix (X : n → Ω → ℝ) (Z : r → Ω → ℝ) :
    Matrix n r ℝ := fun i j => cov[X i, Z j; μ]

def linearImage (A : Matrix m n ℝ) (X : n → Ω → ℝ) : m → Ω → ℝ :=
  fun i ω => ∑ j, A i j * X j ω

theorem covariance_linear_combination
    (X : n → Ω → ℝ) (Y : r → Ω → ℝ)
    (hX : ∀ i, MemLp (X i) 2 μ) (hY : ∀ j, MemLp (Y j) 2 μ)
    (a : n → ℝ) (b : r → ℝ) :
    cov[fun ω => ∑ i, a i * X i ω, fun ω => ∑ j, b j * Y j ω; μ] =
      ∑ i, ∑ j, a i * b j * cov[X i, Y j; μ] := by
  rw [covariance_fun_sum_fun_sum
    (fun i => (hX i).const_mul (a i)) (fun j => (hY j).const_mul (b j))]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  rw [covariance_mul_left, covariance_mul_right]
  ring

omit [Fintype m] in
theorem covariance_linear_image
    (X : n → Ω → ℝ) (hX : ∀ i, MemLp (X i) 2 μ) (A : Matrix m n ℝ) :
    covarianceMatrix μ (linearImage A X) = A * covarianceMatrix μ X * Aᵀ := by
  ext i k
  change cov[fun ω => ∑ j, A i j * X j ω, fun ω => ∑ j, A k j * X j ω; μ] = _
  rw [covariance_linear_combination μ X X hX hX]
  simp only [Matrix.mul_apply, Matrix.transpose_apply, covarianceMatrix, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  apply Finset.sum_congr rfl
  intro l _
  ring

theorem covariance_matrix_psd
    (X : n → Ω → ℝ) (hX : ∀ i, MemLp (X i) 2 μ) :
    (covarianceMatrix μ X).PosSemidef := by
  constructor
  · ext i j
    simp [Matrix.conjTranspose_apply, covarianceMatrix, covariance_comm]
  · intro a
    have h := covariance_linear_combination μ X X hX hX a a
    have hn : 0 ≤ cov[fun ω => ∑ i, a i * X i ω,
        fun ω => ∑ i, a i * X i ω; μ] := by
      exact integral_nonneg fun _ => mul_self_nonneg _
    rw [h] at hn
    simpa only [covarianceMatrix, Matrix.mulVec, dotProduct, Pi.star_apply,
      star_trivial, Finset.mul_sum, mul_assoc, mul_left_comm, mul_comm] using hn

omit [Fintype n] [Fintype r] [IsProbabilityMeasure μ] in
theorem covariance_joint_block
    (X : n → Ω → ℝ) (Z : r → Ω → ℝ) :
    covarianceMatrix μ (Sum.elim X Z) =
      fromBlocks (covarianceMatrix μ X)
        (crossCovarianceMatrix μ X Z)
        (crossCovarianceMatrix μ X Z)ᵀ (covarianceMatrix μ Z) := by
  ext i j
  cases i <;> cases j <;>
    simp [covarianceMatrix, crossCovarianceMatrix, Matrix.transpose_apply, covariance_comm]

omit [Fintype m] in
theorem raw_state_projection_covariance
    (X : n → Ω → ℝ) (Z : r → Ω → ℝ)
    (hX : ∀ i, MemLp (X i) 2 μ) (hZ : ∀ j, MemLp (Z j) 2 μ)
    (A : Matrix m n ℝ) (B : Matrix m r ℝ) :
    covarianceMatrix μ (fun i ω => linearImage A X i ω + linearImage B Z i ω) =
      A * covarianceMatrix μ X * Aᵀ +
        B * (crossCovarianceMatrix μ X Z)ᵀ * Aᵀ +
        (A * crossCovarianceMatrix μ X Z * Bᵀ +
          B * covarianceMatrix μ Z * Bᵀ) := by
  have heq : (fun i ω => linearImage A X i ω + linearImage B Z i ω) =
      linearImage (N2P3Pilot.horizontal A B) (Sum.elim X Z) := by
    funext i ω
    simp [linearImage, N2P3Pilot.horizontal, Fintype.sum_sum_type]
  rw [heq, covariance_linear_image μ (Sum.elim X Z)
    (fun i => by cases i with | inl i => exact hX i | inr i => exact hZ i)]
  rw [covariance_joint_block, N2P3Pilot.joint_covariance_cross_terms]

omit [MeasurableSpace Ω] [Fintype m] in
theorem linear_image_compose
    (A : Matrix m r ℝ) (B : Matrix r n ℝ) (X : n → Ω → ℝ) :
    linearImage A (linearImage B X) = linearImage (A * B) X := by
  funext i ω
  change (A *ᵥ (B *ᵥ (fun j => X j ω))) i = ((A * B) *ᵥ (fun j => X j ω)) i
  rw [Matrix.mulVec_mulVec]

omit [Fintype m] in
theorem deterministic_state_projection_covariance
    (X : n → Ω → ℝ) (hX : ∀ i, MemLp (X i) 2 μ)
    (A : Matrix m n ℝ) (B : Matrix m r ℝ) (J : Matrix r n ℝ) :
    covarianceMatrix μ (fun i ω => linearImage A X i ω +
      linearImage B (linearImage J X) i ω) =
      (A + B * J) * covarianceMatrix μ X * (A + B * J)ᵀ := by
  rw [linear_image_compose]
  have heq : (fun i ω => linearImage A X i ω + linearImage (B * J) X i ω) =
      linearImage (A + B * J) X := by
    funext i ω
    simp [linearImage, add_mul, Finset.sum_add_distrib]
  rw [heq, covariance_linear_image μ X hX]

end RandomVariables

section ApproximatePSD

variable {n m : Type*} [Fintype n] [Fintype m]
variable [DecidableEq n] [DecidableEq m]

def energy (x : n → ℝ) : ℝ := x ⬝ᵥ x

def quad (S : Matrix n n ℝ) (x : n → ℝ) : ℝ := x ⬝ᵥ (S *ᵥ x)

/-- A real symmetric lower spectral bound, equivalently S ≥ -epsilon I. -/
def ApproxPSD (S : Matrix n n ℝ) (epsilon : ℝ) : Prop :=
  S.IsHermitian ∧ ∀ x, -epsilon * energy x ≤ quad S x

omit [DecidableEq n] in
theorem energy_nonneg (x : n → ℝ) : 0 ≤ energy x := by
  simpa [energy] using dotProduct_star_self_nonneg x

omit [DecidableEq n] in
theorem energy_pos {x : n → ℝ} (hx : x ≠ 0) : 0 < energy x := by
  have hn := energy_nonneg x
  have he : energy x ≠ 0 := by simpa [energy, dotProduct_self_eq_zero] using hx
  exact lt_of_le_of_ne hn (Ne.symm he)

omit [DecidableEq n] in
theorem energy_eq_norm_sq (x : n → ℝ) :
    energy x = ‖(WithLp.toLp 2 x : EuclideanSpace ℝ n)‖ ^ 2 := by
  simpa [EuclideanSpace.inner_eq_star_dotProduct, energy] using
    (real_inner_self_eq_norm_sq (WithLp.toLp 2 x : EuclideanSpace ℝ n))

omit [DecidableEq n] in
theorem quad_add (S T : Matrix n n ℝ) (x : n → ℝ) :
    quad (S + T) x = quad S x + quad T x := by
  simp [quad, Matrix.add_mulVec, dotProduct_add]

omit [DecidableEq n] in
theorem quad_smul (c : ℝ) (S : Matrix n n ℝ) (x : n → ℝ) :
    quad (c • S) x = c * quad S x := by
  simp [quad, Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]

theorem quad_identity (x : n → ℝ) :
    quad (1 : Matrix n n ℝ) x = energy x := by
  simp [quad, energy]

theorem approx_psd_iff_shift_psd (S : Matrix n n ℝ) (hSym : S.IsHermitian)
    (epsilon : ℝ) :
    ApproxPSD S epsilon ↔ (S + epsilon • (1 : Matrix n n ℝ)).PosSemidef := by
  have hid : (epsilon • (1 : Matrix n n ℝ)).IsHermitian := by
    change (epsilon • (1 : Matrix n n ℝ))ᴴ = _
    simp
  constructor
  · intro h
    refine ⟨hSym.add hid, fun x => ?_⟩
    change 0 ≤ quad (S + epsilon • (1 : Matrix n n ℝ)) x
    rw [quad_add, quad_smul, quad_identity]
    have hx := h.2 x
    linarith
  · intro h
    refine ⟨hSym, fun x => ?_⟩
    have hx := h.2 x
    change 0 ≤ quad (S + epsilon • (1 : Matrix n n ℝ)) x at hx
    rw [quad_add, quad_smul, quad_identity] at hx
    linarith

omit [DecidableEq n] [DecidableEq m] in
theorem quad_congruence (S : Matrix n n ℝ) (A : Matrix m n ℝ) (x : m → ℝ) :
    quad (A * S * Aᵀ) x = quad S (Aᵀ *ᵥ x) := by
  unfold quad
  rw [← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec, dotProduct_mulVec]
  have hv : x ᵥ* A = Aᵀ *ᵥ x := by simpa using Matrix.vecMul_transpose Aᵀ x
  rw [hv]

theorem l2_norm_transpose (A : Matrix m n ℝ) : ‖Aᵀ‖ = ‖A‖ := by
  classical
  simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using
    Matrix.l2_opNorm_conjTranspose A

omit [DecidableEq m] in
theorem energy_mulVec_le (A : Matrix m n ℝ) (x : n → ℝ) :
    energy (A *ᵥ x) ≤ ‖A‖ ^ 2 * energy x := by
  have h := Matrix.l2_opNorm_mulVec A (WithLp.toLp 2 x : EuclideanSpace ℝ n)
  change ‖(WithLp.toLp 2 (A *ᵥ x) : EuclideanSpace ℝ m)‖ ≤
    ‖A‖ * ‖(WithLp.toLp 2 x : EuclideanSpace ℝ n)‖ at h
  have hs := mul_self_le_mul_self (norm_nonneg _) h
  rw [energy_eq_norm_sq, energy_eq_norm_sq]
  nlinarith

theorem abs_quad_le_l2_norm (E : Matrix n n ℝ) (x : n → ℝ) :
    |quad E x| ≤ ‖E‖ * energy x := by
  let u : EuclideanSpace ℝ n := WithLp.toLp 2 x
  let v : EuclideanSpace ℝ n := WithLp.toLp 2 (E *ᵥ x)
  have hi : ⟪u, v⟫_ℝ = quad E x := by
    simp [u, v, EuclideanSpace.inner_eq_star_dotProduct, quad, dotProduct_comm]
  have hop := Matrix.l2_opNorm_mulVec E u
  change ‖v‖ ≤ ‖E‖ * ‖u‖ at hop
  calc
    |quad E x| = |⟪u, v⟫_ℝ| := congrArg abs hi.symm
    _ ≤ ‖u‖ * ‖v‖ := abs_real_inner_le_norm _ _
    _ ≤ ‖u‖ * (‖E‖ * ‖u‖) := mul_le_mul_of_nonneg_left hop (norm_nonneg _)
    _ = ‖E‖ * energy x := by rw [energy_eq_norm_sq]; dsimp [u]; ring

theorem norm_residual_approx_psd (E : Matrix n n ℝ) (hE : E.IsHermitian)
    (eta : ℝ) (hEta : ‖E‖ ≤ eta) : ApproxPSD E eta := by
  refine ⟨hE, fun x => ?_⟩
  have h := (abs_le.mp (abs_quad_le_l2_norm E x)).1
  have hm := mul_le_mul_of_nonneg_right hEta (energy_nonneg x)
  nlinarith

omit [DecidableEq n] in
theorem approx_psd_add (S T : Matrix n n ℝ) (epsilon eta : ℝ)
    (hS : ApproxPSD S epsilon) (hT : ApproxPSD T eta) :
    ApproxPSD (S + T) (epsilon + eta) := by
  refine ⟨hS.1.add hT.1, fun x => ?_⟩
  rw [quad_add]
  have hs := hS.2 x
  have ht := hT.2 x
  nlinarith

theorem approx_psd_pushforward (S : Matrix n n ℝ) (A : Matrix m n ℝ)
    (epsilon : ℝ) (hEpsilon : 0 ≤ epsilon) (hS : ApproxPSD S epsilon) :
    ApproxPSD (A * S * Aᵀ) (epsilon * ‖A‖ ^ 2) := by
  refine ⟨?_, fun x => ?_⟩
  · have ht : Sᵀ = S := by
      simpa only [Matrix.IsHermitian, Matrix.conjTranspose_eq_transpose_of_trivial] using hS.1
    change (A * S * Aᵀ)ᴴ = A * S * Aᵀ
    simp only [Matrix.conjTranspose_eq_transpose_of_trivial, Matrix.transpose_mul,
      Matrix.transpose_transpose, ht, Matrix.mul_assoc]
  · rw [quad_congruence]
    have hs := hS.2 (Aᵀ *ᵥ x)
    have he := energy_mulVec_le Aᵀ x
    rw [l2_norm_transpose] at he
    have hm := mul_le_mul_of_nonneg_left he hEpsilon
    nlinarith

theorem propagated_residual_lower_bound
    (S : Matrix n n ℝ) (A : Matrix m n ℝ) (E : Matrix m m ℝ)
    (epsilon eta : ℝ) (hEpsilon : 0 ≤ epsilon) (hS : ApproxPSD S epsilon)
    (hE : E.IsHermitian) (hEta : ‖E‖ ≤ eta) :
    ApproxPSD (A * S * Aᵀ + E) (epsilon * ‖A‖ ^ 2 + eta) := by
  exact approx_psd_add _ _ _ _ (approx_psd_pushforward S A epsilon hEpsilon hS)
    (norm_residual_approx_psd E hE eta hEta)

theorem ridge_makes_pos_def
    (S : Matrix n n ℝ) (epsilon rho : ℝ) (hS : ApproxPSD S epsilon)
    (hRho : epsilon < rho) : (S + rho • (1 : Matrix n n ℝ)).PosDef := by
  have hid : (rho • (1 : Matrix n n ℝ)).IsHermitian := by
    change (rho • (1 : Matrix n n ℝ))ᴴ = _
    simp
  refine ⟨hS.1.add hid, fun x hx => ?_⟩
  change 0 < quad (S + rho • (1 : Matrix n n ℝ)) x
  rw [quad_add, quad_smul, quad_identity]
  have hs := hS.2 x
  have he := energy_pos hx
  have hp := mul_pos (sub_pos.mpr hRho) he
  nlinarith

theorem propagated_residual_ridge_pos_def
    (S : Matrix n n ℝ) (A : Matrix m n ℝ) (E : Matrix m m ℝ)
    (epsilon eta rho : ℝ) (hEpsilon : 0 ≤ epsilon) (hS : ApproxPSD S epsilon)
    (hE : E.IsHermitian) (hEta : ‖E‖ ≤ eta)
    (hRho : epsilon * ‖A‖ ^ 2 + eta < rho) :
    (A * S * Aᵀ + E + rho • (1 : Matrix m m ℝ)).PosDef := by
  exact ridge_makes_pos_def _ _ _
    (propagated_residual_lower_bound S A E epsilon eta hEpsilon hS hE hEta) hRho

theorem approx_psd_eigenvalue_lower
    (S : Matrix n n ℝ) (epsilon : ℝ) (hS : ApproxPSD S epsilon) (i : n) :
    -epsilon ≤ hS.1.eigenvalues i := by
  let v : EuclideanSpace ℝ n := hS.1.eigenvectorBasis i
  have he : energy (fun j => v j) = 1 := by
    rw [energy_eq_norm_sq]
    change ‖hS.1.eigenvectorBasis i‖ ^ 2 = 1
    rw [hS.1.eigenvectorBasis.orthonormal.1 i]
    norm_num
  have hv : hS.1.eigenvalues i = quad S (fun j => v j) := by
    simpa [v, quad] using hS.1.eigenvalues_eq i
  have h := hS.2 (fun j => v j)
  rw [he, mul_one, ← hv] at h
  exact h

end ApproximatePSD

end N2P3.Covariance
