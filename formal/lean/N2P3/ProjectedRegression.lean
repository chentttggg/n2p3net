import N2P3.RetainedBasis
import N2P3.BayesDensity

/-!
# The source's ambient projected ridge fit in genuine observation coordinates

`continuous_decision.fit.solve(phi)` stacks independently whitened/nuisance-
projected D,Y, then computes inv(DᵀD+lambda I) and that inverse times DᵀY.
The source keeps the ambient N rows, including q deleted directions. This file
derives its exact equality to a proper ridge model on n-q real observations.
The complement B is constructed from the actual thin QR columns; nonredundancy
and completeness are proved, not inferred from shape or assumed fit equality.

The result concerns frozen phi/design/spatial covariance and fixed positive
ridge. Data-driven phi, covariance/hyperparameter uncertainty, cross-record
noise independence and numerical factorization refinement are not inferred.
The held/calibration independent-block model remains an explicit modeling
boundary; support projection does not prove that independence.
-/

namespace N2P3.ProjectedRegression

open Matrix MeasureTheory
open N2P3.ScaleGauge N2P3.NuisanceProjection N2P3.MatrixNormalCoordinates
open scoped Matrix BigOperators ComplexOrder

noncomputable section

variable {n q m p c : Type*} [Fintype n] [Fintype q] [Fintype m] [Fintype p] [Fintype c]
  [DecidableEq n] [DecidableEq q] [DecidableEq m] [DecidableEq p] [DecidableEq c]

def sourcePrecision (D : Matrix n p ℝ) (lam : ℝ) : Matrix p p ℝ := Dᵀ * D + lam • 1

def sourceRowCov (D : Matrix n p ℝ) (lam : ℝ) : Matrix p p ℝ := (sourcePrecision D lam)⁻¹

def sourceMean (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (lam : ℝ) : Matrix p c ℝ :=
  sourceRowCov D lam * Dᵀ * Y

omit [Fintype c] [Fintype p] [DecidableEq c] in
theorem source_precision_is_declared_precision (D : Matrix n p ℝ) (lam : ℝ) :
    sourcePrecision D lam = precision D (1 : Matrix n n ℝ) lam := by simp [sourcePrecision, precision]

omit [Fintype c] [DecidableEq c] in
theorem source_rowCov_is_declared_rowCov (D : Matrix n p ℝ) (lam : ℝ) :
    sourceRowCov D lam = rowCov D (1 : Matrix n n ℝ) lam := by
  rw [sourceRowCov, source_precision_is_declared_precision]
  rfl

omit [Fintype c] [DecidableEq c] in
theorem source_mean_is_declared_mean (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (lam : ℝ) :
    sourceMean D Y lam = mean D Y (1 : Matrix n n ℝ) lam := by
  simp [sourceMean, sourceRowCov, sourcePrecision, mean, rowCov, precision]

omit [Fintype p] [DecidableEq p] [Fintype c] [DecidableEq c] [DecidableEq q] [DecidableEq m] in
/-- Cross products are preserved by the actual retained projection. This is
frame algebra; final observation-model claims additionally establish BᵀB=I. -/
theorem frame_cross_product_preserved
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (hPB : B * Bᵀ = projection Q)
    {a b : Type*} (D : Matrix n a ℝ) (Y : Matrix n b ℝ) (hY : projection Q * Y = Y) :
    (Bᵀ * D)ᵀ * (Bᵀ * Y) = Dᵀ * Y := by
  calc
    _ = Dᵀ * ((B * Bᵀ) * Y) := by
      simp only [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc]
    _ = Dᵀ * Y := by rw [hPB, hY]

omit [Fintype c] [Fintype p] [DecidableEq c] [DecidableEq q] [DecidableEq m] in
theorem compressed_precision_equals_source
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (hPB : B * Bᵀ = projection Q)
    (D : Matrix n p ℝ) (hD : projection Q * D = D) (lam : ℝ) :
    sourcePrecision (Bᵀ * D) lam = sourcePrecision D lam := by
  rw [sourcePrecision, frame_cross_product_preserved Q B hPB D D hD]
  rfl

omit [Fintype c] [DecidableEq c] [DecidableEq q] [DecidableEq m] in
theorem compressed_rowCov_equals_source
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (hPB : B * Bᵀ = projection Q)
    (D : Matrix n p ℝ) (hD : projection Q * D = D) (lam : ℝ) :
    sourceRowCov (Bᵀ * D) lam = sourceRowCov D lam := by
  rw [sourceRowCov, compressed_precision_equals_source Q B hPB D hD lam]
  rfl

omit [Fintype c] [DecidableEq c] [DecidableEq q] [DecidableEq m] in
theorem compressed_mean_equals_source
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (hPB : B * Bᵀ = projection Q)
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (hD : projection Q * D = D)
    (hY : projection Q * Y = Y) (lam : ℝ) :
    sourceMean (Bᵀ * D) (Bᵀ * Y) lam = sourceMean D Y lam := by
  unfold sourceMean
  rw [compressed_rowCov_equals_source Q B hPB D hD lam,
    Matrix.mul_assoc, frame_cross_product_preserved Q B hPB D Y hY, ← Matrix.mul_assoc]

omit [Fintype c] [DecidableEq c] in
/-- A genuine n-q-coordinate observation model, not an assumed fit invariance:
the actual Gram/cross-Gram are proved equal before deriving mean/covariance. -/
theorem exists_actual_source_fit_coordinates
    (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1)
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (hD : projection Q * D = D)
    (hY : projection Q * Y = Y) (lam : ℝ) :
    ∃ B : Matrix n (Fin (Fintype.card n - Fintype.card q)) ℝ,
      Bᵀ * B = 1 ∧ B * Bᵀ = projection Q ∧
      (Bᵀ * D)ᵀ * (Bᵀ * D) = Dᵀ * D ∧
      (Bᵀ * D)ᵀ * (Bᵀ * Y) = Dᵀ * Y ∧
      rowCov (Bᵀ * D) (1 : Matrix (Fin (Fintype.card n - Fintype.card q))
        (Fin (Fintype.card n - Fintype.card q)) ℝ) lam = sourceRowCov D lam ∧
      mean (Bᵀ * D) (Bᵀ * Y) (1 : Matrix (Fin (Fintype.card n - Fintype.card q))
        (Fin (Fintype.card n - Fintype.card q)) ℝ) lam = sourceMean D Y lam := by
  obtain ⟨B, hB, _, hPB⟩ := RetainedBasis.exists_retained_basis Q hQ
  refine ⟨B, hB, hPB, frame_cross_product_preserved Q B hPB D D hD,
    frame_cross_product_preserved Q B hPB D Y hY, ?_, ?_⟩
  · rw [← source_rowCov_is_declared_rowCov]
    exact compressed_rowCov_equals_source Q B hPB D hD lam
  · rw [← source_mean_is_declared_mean]
    exact compressed_mean_equals_source Q B hPB D Y hD hY lam

/-- Positivity follows even for rank-deficient projected designs, through the
positive ridge. The source's symmetric averaging leaves this exact rowCov. -/
theorem source_rowCov_positive_definite (D : Matrix n p ℝ) (lam : ℝ) (hlam : 0 < lam) :
    (sourceRowCov D lam).PosDef := by
  rw [source_rowCov_is_declared_rowCov]
  exact rowCov_positive_definite D 1 lam Matrix.PosDef.one hlam

omit [Fintype c] [DecidableEq c] in
theorem exact_source_rowCov_symmetrization (D : Matrix n p ℝ) (lam : ℝ) (hlam : 0 < lam) :
    (1 / 2 : ℝ) • (sourceRowCov D lam + (sourceRowCov D lam)ᵀ) = sourceRowCov D lam := by
  have hs : (sourceRowCov D lam)ᵀ = sourceRowCov D lam := by
    simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using
      (source_rowCov_positive_definite D lam hlam).isHermitian.eq
  rw [hs]
  ext i j
  simp
  ring

/-- The source's actual R can normalize the proper reduced observation model:
its factor equation is transported through the proved Gram equality, while
the center is the source ambient projected ridge mean. -/
theorem exists_actual_source_posterior_density
    (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1)
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (hD : projection Q * D = D)
    (hY : projection Q * Y = Y) (lam : ℝ) (hlam : 0 < lam)
    (R : Matrix p p ℝ) (hR : Rᵀ * R = sourcePrecision D lam)
    (L Sigma : Matrix c c ℝ) (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef) :
    ∃ B : Matrix n (Fin (Fintype.card n - Fintype.card q)) ℝ,
      Bᵀ * B = 1 ∧ B * Bᵀ = projection Q ∧
      ∀ x : (p × c) → ℝ,
        BayesDensity.normalizedJointPDF (Bᵀ * D) (Bᵀ * Y)
          (1 : Matrix (Fin (Fintype.card n - Fintype.card q))
            (Fin (Fintype.card n - Fintype.card q)) ℝ) Sigma lam x =
          matrixPDF R L (sourceMean D Y lam) x := by
  obtain ⟨B, hB, _, hPB⟩ := RetainedBasis.exists_retained_basis Q hQ
  have hR' : Rᵀ * R = precision (Bᵀ * D)
      (1 : Matrix (Fin (Fintype.card n - Fintype.card q))
        (Fin (Fintype.card n - Fintype.card q)) ℝ) lam := by
    rw [← source_precision_is_declared_precision, compressed_precision_equals_source Q B hPB D hD lam]
    exact hR
  refine ⟨B, hB, hPB, ?_⟩
  intro x
  rw [BayesDensity.normalized_joint_equals_matrixPDF (Bᵀ * D) (Bᵀ * Y) 1 Sigma lam R L
    hR' hL Matrix.PosDef.one hlam hSigma x,
    ← source_mean_is_declared_mean, compressed_mean_equals_source Q B hPB D Y hD hY lam]

section HeldCoordinates

variable {h qh mh : Type*} [Fintype h] [Fintype qh] [Fintype mh]
  [DecidableEq h] [DecidableEq qh] [DecidableEq mh]

omit [Fintype c] [DecidableEq p] [DecidableEq c] [DecidableEq qh] in
theorem held_fitted_mean_retained
    (Qh : Matrix h qh ℝ) (Dh : Matrix h p ℝ) (hDh : projection Qh * Dh = Dh)
    (M : Matrix p c ℝ) : projection Qh * (Dh * M) = Dh * M := by
  rw [← Matrix.mul_assoc, hDh]

omit [DecidableEq p] [DecidableEq qh] in
theorem held_spatial_whitened_residual_retained
    (Qh : Matrix h qh ℝ) (Dh : Matrix h p ℝ) (Yh : Matrix h c ℝ)
    (hDh : projection Qh * Dh = Dh) (hYh : projection Qh * Yh = Yh)
    (M : Matrix p c ℝ) (Lsigma : Matrix c c ℝ) :
    projection Qh * ((Yh - Dh * M) * (Lsigma⁻¹)ᵀ) =
      (Yh - Dh * M) * (Lsigma⁻¹)ᵀ := by
  rw [← Matrix.mul_assoc, Matrix.mul_sub, hYh, held_fitted_mean_retained Qh Dh hDh M]

omit [Fintype c] [DecidableEq c] [DecidableEq p] [DecidableEq qh] in
theorem held_lowrank_design_retained
    (Qh : Matrix h qh ℝ) (Dh : Matrix h p ℝ) (hDh : projection Qh * Dh = Dh)
    (Lrow : Matrix p p ℝ) : projection Qh * (Dh * Lrow) = Dh * Lrow := by
  rw [← Matrix.mul_assoc, hDh]

omit [Fintype c] [DecidableEq c] [DecidableEq p] [Fintype h] [DecidableEq h] in
/-- Actual source U=Dh*chol(Prow), hence UUᵀ=Dh*Prow*Dhᵀ. -/
theorem held_lowrank_covariance_matches_source
    (Dh : Matrix h p ℝ) (Lrow Prow : Matrix p p ℝ) (hLrow : Lrow * Lrowᵀ = Prow) :
    (Dh * Lrow) * (Dh * Lrow)ᵀ = Dh * Prow * Dhᵀ := by
  simp only [Matrix.transpose_mul, Matrix.mul_assoc]
  rw [← Matrix.mul_assoc Lrow Lrowᵀ Dhᵀ, hLrow]

omit [Fintype c] [DecidableEq c] [DecidableEq p] [DecidableEq qh] in
/-- Compressed held covariance is derived from the singular projected noise
plus posterior-kernel uncertainty, using both nonredundant basis identities. -/
theorem held_compressed_predictive_covariance
    (Qh : Matrix h qh ℝ) (Bh : Matrix h mh ℝ) (hB : Bhᵀ * Bh = 1)
    (hPB : Bh * Bhᵀ = projection Qh) (Dh : Matrix h p ℝ) (Prow : Matrix p p ℝ) :
    Bhᵀ * (projection Qh + Dh * Prow * Dhᵀ) * Bh =
      1 + (Bhᵀ * Dh) * Prow * (Bhᵀ * Dh)ᵀ := by
  have hPBh : projection Qh * Bh = Bh := by
    rw [← hPB, Matrix.mul_assoc, hB, Matrix.mul_one]
  rw [Matrix.mul_add, Matrix.add_mul, Matrix.mul_assoc Bhᵀ (projection Qh) Bh, hPBh, hB]
  simp only [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc]

omit [Fintype c] [DecidableEq c] [DecidableEq p] [DecidableEq qh] [DecidableEq mh] in
theorem held_actual_small_gram_preserved
    (Qh : Matrix h qh ℝ) (Bh : Matrix h mh ℝ) (hPB : Bh * Bhᵀ = projection Qh)
    (Dh : Matrix h p ℝ) (hDh : projection Qh * Dh = Dh) (Lrow : Matrix p p ℝ) :
    ((Bhᵀ * Dh) * Lrow)ᵀ * ((Bhᵀ * Dh) * Lrow) =
      (Dh * Lrow)ᵀ * (Dh * Lrow) := by
  rw [Matrix.mul_assoc Bhᵀ Dh Lrow]
  exact frame_cross_product_preserved Qh Bh hPB _ _ (held_lowrank_design_retained Qh Dh hDh Lrow)

omit [DecidableEq c] in
/-- A held complement is constructed from Qh, then source fitted mean and U
produce exactly the proper reduced-coordinate predictive score. -/
theorem exists_actual_held_scoring_coordinates
    (Qh : Matrix h qh ℝ) (hQ : Qhᵀ * Qh = 1)
    (Dh : Matrix h p ℝ) (Yh : Matrix h c ℝ) (hDh : projection Qh * Dh = Dh)
    (hYh : projection Qh * Yh = Yh) (M : Matrix p c ℝ) (Lrow : Matrix p p ℝ) :
    ∃ Bh : Matrix h (Fin (Fintype.card h - Fintype.card qh)) ℝ,
      Bhᵀ * Bh = 1 ∧ Bh * Bhᵀ = projection Qh ∧
        GaussianLowRank.fullCandidateScore (Dh * Lrow) (Yh - Dh * M) =
          reducedCandidateScore Qh Bh (Dh * Lrow) (Yh - Dh * M) := by
  obtain ⟨Bh, hB, _, hPB⟩ := RetainedBasis.exists_retained_basis Qh hQ
  refine ⟨Bh, hB, hPB, ?_⟩
  apply identity_extension_score_equals_retained_score Qh Bh hB hPB _
    (held_lowrank_design_retained Qh Dh hDh Lrow) _
  rw [Matrix.mul_sub, hYh, held_fitted_mean_retained Qh Dh hDh M]

theorem exists_actual_spatial_whitened_held_score
    (Qh : Matrix h qh ℝ) (hQ : Qhᵀ * Qh = 1)
    (Dh : Matrix h p ℝ) (Yh : Matrix h c ℝ) (hDh : projection Qh * Dh = Dh)
    (hYh : projection Qh * Yh = Yh) (M : Matrix p c ℝ)
    (Lrow : Matrix p p ℝ) (Lsigma : Matrix c c ℝ) :
    ∃ Bh : Matrix h (Fin (Fintype.card h - Fintype.card qh)) ℝ,
      Bhᵀ * Bh = 1 ∧ Bh * Bhᵀ = projection Qh ∧
        GaussianLowRank.fullCandidateScore (Dh * Lrow) ((Yh - Dh * M) * (Lsigma⁻¹)ᵀ) =
          reducedCandidateScore Qh Bh (Dh * Lrow) ((Yh - Dh * M) * (Lsigma⁻¹)ᵀ) := by
  obtain ⟨Bh, hB, _, hPB⟩ := RetainedBasis.exists_retained_basis Qh hQ
  exact ⟨Bh, hB, hPB, identity_extension_score_equals_retained_score Qh Bh hB hPB _
    (held_lowrank_design_retained Qh Dh hDh Lrow) _
    (held_spatial_whitened_residual_retained Qh Dh Yh hDh hYh M Lsigma)⟩

end HeldCoordinates

section MultipleRecords

variable {n₁ n₂ q₁ q₂ : Type*} [Fintype n₁] [Fintype n₂] [Fintype q₁] [Fintype q₂]
  [DecidableEq n₁] [DecidableEq n₂] [DecidableEq q₁] [DecidableEq q₂]

def blockNuisance (Q₁ : Matrix n₁ q₁ ℝ) (Q₂ : Matrix n₂ q₂ ℝ) :
    Matrix (n₁ ⊕ n₂) (q₁ ⊕ q₂) ℝ := Matrix.fromBlocks Q₁ 0 0 Q₂

omit [Fintype q₁] [Fintype q₂] [DecidableEq n₁] [DecidableEq n₂] in
theorem block_nuisance_gram (Q₁ : Matrix n₁ q₁ ℝ) (Q₂ : Matrix n₂ q₂ ℝ)
    (hQ₁ : Q₁ᵀ * Q₁ = 1) (hQ₂ : Q₂ᵀ * Q₂ = 1) :
    (blockNuisance Q₁ Q₂)ᵀ * blockNuisance Q₁ Q₂ = 1 := by
  simp [blockNuisance, Matrix.fromBlocks_transpose, Matrix.fromBlocks_multiply, hQ₁, hQ₂,
    Matrix.fromBlocks_one]

omit [Fintype n₁] [Fintype n₂] [DecidableEq q₁] [DecidableEq q₂] in
theorem block_projection (Q₁ : Matrix n₁ q₁ ℝ) (Q₂ : Matrix n₂ q₂ ℝ) :
    projection (blockNuisance Q₁ Q₂) = Matrix.fromBlocks (projection Q₁) 0 0 (projection Q₂) := by
  simp only [projection, blockNuisance, Matrix.fromBlocks_transpose, Matrix.fromBlocks_multiply,
    Matrix.transpose_zero, Matrix.mul_zero, Matrix.zero_mul, add_zero, zero_add]
  ext i j
  cases i <;> cases j <;> simp [Matrix.fromBlocks, Matrix.sub_apply, Matrix.one_apply]

omit [DecidableEq q₁] [DecidableEq q₂] in
/-- Each recording is projected before stacking, as the source actually does.
The combined nuisance space preserves independent reset/QR record boundaries. -/
theorem stacked_projected_records_retained
    (Q₁ : Matrix n₁ q₁ ℝ) (Q₂ : Matrix n₂ q₂ ℝ)
    {a : Type*} (X₁ : Matrix n₁ a ℝ) (X₂ : Matrix n₂ a ℝ)
    (hX₁ : projection Q₁ * X₁ = X₁) (hX₂ : projection Q₂ * X₂ = X₂) :
    projection (blockNuisance Q₁ Q₂) * verticalStack X₁ X₂ = verticalStack X₁ X₂ := by
  rw [block_projection]
  ext i j
  cases i with
  | inl i => simpa [Matrix.mul_apply, verticalStack, Fintype.sum_sum_type] using
      congrArg (fun A : Matrix n₁ a ℝ => A i j) hX₁
  | inr i => simpa [Matrix.mul_apply, verticalStack, Fintype.sum_sum_type] using
      congrArg (fun A : Matrix n₂ a ℝ => A i j) hX₂

end MultipleRecords

end

end N2P3.ProjectedRegression
