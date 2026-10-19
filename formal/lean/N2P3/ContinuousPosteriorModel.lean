import N2P3.GaussianJointLaw
import N2P3.GaussianConvolution
import N2P3.RetainedGaussianLaw
import N2P3.MatrixNormalMoments
import N2P3.ProjectedRegression

/-!
# Conditional posterior moments and the source predictive score

Connect the actual disintegrating posterior kernel to its actual moments.
Then integrate a posterior coefficient density against the declared independent
white observation likelihood in genuine retained coordinates, and identify its
logarithm with the source low-rank score plus the common intrinsic constant.
No predictive integral, score identity, or existence of its covariance factors
is an input hypothesis. The independent-block statistical model remains an
explicit scope of the likelihood, not a conclusion about disjoint EEG scopes.
-/

namespace N2P3.ContinuousPosteriorModel

open Matrix MeasureTheory ProbabilityTheory
open N2P3.ScaleGauge N2P3.MatrixNormalCoordinates N2P3.BayesDensity
open N2P3.RetainedGaussianLaw N2P3.RetainedMatrixMeasure
open scoped Matrix ComplexOrder Kronecker BigOperators

noncomputable section

variable {n p c : Type*} [Fintype n] [Fintype p] [Fintype c]
variable [DecidableEq n] [DecidableEq p] [DecidableEq c]

theorem conditional_posterior_mean
    (D : Matrix n p ℝ) (K : Matrix n n ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (L Sigma : Matrix c c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef)
    (y : (n × c) → ℝ) (i : p) (j : c) :
    (∫ x : (p × c) → ℝ, x (i, j)
      ∂GaussianJointLaw.posteriorKernel D K lam R L y) = mean D (matricize y) K lam i j := by
  rw [GaussianJointLaw.posteriorKernel_apply]
  exact MatrixNormalMoments.source_posterior_mean D (matricize y) K lam R L Sigma
    hR hL hK hlam hSigma i j

theorem conditional_posterior_covariance
    (D : Matrix n p ℝ) (K : Matrix n n ℝ) (lam : ℝ)
    (R : Matrix p p ℝ) (L Sigma : Matrix c c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) (y : (n × c) → ℝ) :
    Covariance.covarianceMatrix (GaussianJointLaw.posteriorKernel D K lam R L y)
      (fun i x => x i) = rowCov D K lam ⊗ₖ Sigma := by
  rw [GaussianJointLaw.posteriorKernel_apply]
  exact MatrixNormalMoments.source_posterior_covariance D (matricize y) K lam R L Sigma
    hR hL hK hlam hSigma

section HeldCoordinates

variable {q m : Type*} [Fintype q] [Fintype m] [DecidableEq m]

/-- The posterior coefficient uncertainty is integrated, including the
candidate-dependent covariance determinant. B is the genuine retained basis
already derived from the actual thin QR, and the small source factor T keeps
its explicit exact solver interface. -/
theorem source_predictive_integral_log_score [LinearOrder p]
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ)
    (hB : Bᵀ * B = 1) (hPB : B * Bᵀ = NuisanceProjection.projection Q)
    (D : Matrix n p ℝ) (hD : NuisanceProjection.projection Q * D = D)
    (center : Matrix p c ℝ) (P C : Matrix p p ℝ)
    (hP : P.PosDef) (hC : C * Cᵀ = P)
    (L Sigma : Matrix c c ℝ) (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef)
    (T : Matrix p p ℝ) (hT : T * Tᵀ = GaussianLowRank.smallCovariance (D * C))
    (htri : T.BlockTriangular OrderDual.toDual) (hdiag : ∀ i, 0 < T i i)
    (Y : Matrix n c ℝ) (hY : NuisanceProjection.projection Q * Y = Y) :
    Real.log (∫ x : (p × c) → ℝ,
      likelihoodDensity (Bᵀ * D) (matricize x) (1 : Matrix m m ℝ) L
        (vectorize (Bᵀ * Y)) * matrixPDF C⁻¹ L center x) =
      GaussianLowRank.sourceCandidateScore (D * C) T (spatialWhiten L (Y - D * center)) +
        gaussianLogConstant (Fintype.card m) L := by
  let Db := Bᵀ * D
  have hA : P⁻¹.PosDef := hP.inv
  have hpred : GaussianConvolution.predictiveCovariance Db P⁻¹ 1 =
      GaussianLowRank.fullCovariance (Bᵀ * (D * C)) := by
    rw [GaussianConvolution.predictiveCovariance,
      Matrix.nonsing_inv_nonsing_inv P (positive_definite_det_unit hP)]
    rw [← hC]
    simp only [GaussianLowRank.fullCovariance, Db, Matrix.transpose_mul, Matrix.mul_assoc]
  obtain ⟨G, hG⟩ := covariance_factor_exists _
    (GaussianLowRank.fullCovariance_posDef (Bᵀ * (D * C)))
  obtain ⟨F, hF⟩ := covariance_factor_exists _
    (GaussianConvolution.updatedPrecision_posDef Db P⁻¹ 1 hA Matrix.PosDef.one)
  let M : GaussianConvolution.Model (n := m) (p := p) (c := c) :=
    { D := Db, A := P⁻¹, K := 1, Sigma := Sigma, mean := center,
      R := C⁻¹, B := Fᵀ, H := 1, G := G, L := L,
      hA := hA, hK := Matrix.PosDef.one, hSigma := hSigma,
      hR := likelihood_row_factor_gram C P hC,
      hB := by simpa only [Matrix.transpose_transpose] using hF,
      hH := by simp, hG := hG.trans hpred.symm, hL := hL }
  have hintegral := M.predictive_marginal_explicit (Bᵀ * Y)
  change (∫ x : (p × c) → ℝ,
    likelihoodDensity Db (matricize x) 1 L (vectorize (Bᵀ * Y)) *
      matrixPDF C⁻¹ L center x) = matrixPDF G⁻¹ L (Db * center) (vectorize (Bᵀ * Y)) at hintegral
  rw [hintegral]
  have hrec : channelLift B *ᵥ vectorize (Bᵀ * Y) = vectorize Y := by
    rw [channelLift_vectorize, ← Matrix.mul_assoc, hPB, hY]
  have hpdf := candidatePDF_on_embedding B hB G L (D * center) (vectorize (Bᵀ * Y))
  rw [hrec] at hpdf
  have hcenter : Bᵀ * (D * center) = Db * center := (Matrix.mul_assoc _ _ _).symm
  rw [hcenter] at hpdf
  rw [← hpdf]
  exact candidatePDF_log_equals_source_score Q B hPB (D * C)
    (by rw [← Matrix.mul_assoc, hD]) Y (D * center) hY
    (by rw [← Matrix.mul_assoc, hD]) G L Sigma hG hL hSigma T hT htri hdiag

/-- The integral is taken against the actual calibration posterior kernel of
the joint law, not against a distribution identified only by its moments. -/
theorem fitted_posterior_predictive_log_score [LinearOrder p]
    {nc : Type*} [Fintype nc] [DecidableEq nc]
    (Dc : Matrix nc p ℝ) (Yc : Matrix nc c ℝ) (Kc : Matrix nc nc ℝ) (lam : ℝ)
    (C : Matrix p p ℝ) (hC : C * Cᵀ = rowCov Dc Kc lam)
    (hKc : Kc.PosDef) (hlam : 0 < lam)
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ)
    (hB : Bᵀ * B = 1) (hPB : B * Bᵀ = NuisanceProjection.projection Q)
    (D : Matrix n p ℝ) (hD : NuisanceProjection.projection Q * D = D)
    (L Sigma : Matrix c c ℝ) (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef)
    (T : Matrix p p ℝ) (hT : T * Tᵀ = GaussianLowRank.smallCovariance (D * C))
    (htri : T.BlockTriangular OrderDual.toDual) (hdiag : ∀ i, 0 < T i i)
    (Y : Matrix n c ℝ) (hY : NuisanceProjection.projection Q * Y = Y) :
    IsMarkovKernel (GaussianJointLaw.posteriorKernel Dc Kc lam C⁻¹ L) ∧
    Real.log (∫ x : (p × c) → ℝ,
      likelihoodDensity (Bᵀ * D) (matricize x) (1 : Matrix m m ℝ) L
        (vectorize (Bᵀ * Y))
      ∂GaussianJointLaw.posteriorKernel Dc Kc lam C⁻¹ L (vectorize Yc)) =
      GaussianLowRank.sourceCandidateScore (D * C) T
        (spatialWhiten L (Y - D * mean Dc Yc Kc lam)) +
        gaussianLogConstant (Fintype.card m) L := by
  have hprec := likelihood_row_factor_gram C (rowCov Dc Kc lam) hC
  unfold rowCov at hprec
  rw [Matrix.nonsing_inv_nonsing_inv _
    (positive_definite_det_unit (precision_positive_definite Dc Kc lam hKc hlam))] at hprec
  refine ⟨GaussianJointLaw.posteriorKernel_markov Dc Kc L Sigma lam C⁻¹ hprec
    hL hKc hlam hSigma, ?_⟩
  have h := source_predictive_integral_log_score Q B hB hPB D hD
    (mean Dc Yc Kc lam) (rowCov Dc Kc lam) C
    (rowCov_positive_definite Dc Kc lam hKc hlam)
    hC L Sigma hL hSigma T hT htri hdiag Y hY
  rw [GaussianJointLaw.posteriorKernel_apply]
  change Real.log (∫ x : (p × c) → ℝ,
    likelihoodDensity (Bᵀ * D) (matricize x) 1 L (vectorize (Bᵀ * Y))
    ∂volume.withDensity (fun x => ENNReal.ofReal (matrixPDF C⁻¹ L (mean Dc Yc Kc lam) x))) = _
  have hmeas : Measurable (fun x : (p × c) → ℝ =>
      ENNReal.ofReal (matrixPDF C⁻¹ L (mean Dc Yc Kc lam) x)) :=
    (GaussianJointLaw.matrixPDF_continuous_composed C⁻¹ L
      (continuous_const : Continuous (fun _ : (p × c) → ℝ => mean Dc Yc Kc lam))
      continuous_id).measurable.ennreal_ofReal
  rw [integral_withDensity_eq_integral_toReal_smul (μ := volume) hmeas
    (Filter.Eventually.of_forall (fun _ => ENNReal.ofReal_lt_top))]
  simp_rw [ENNReal.toReal_ofReal (matrixPDF_nonnegative C⁻¹ L _ _), smul_eq_mul]
  simpa only [mul_comm] using h

/-- The actual ambient projected calibration fit is first compressed into its
nonredundant observation space. Its disintegrating posterior, using the
source's row-covariance Cholesky factor C, then gives the actual source score. -/
theorem projected_source_posterior_predictive_log_score [LinearOrder p]
    {nc qc mc : Type*} [Fintype nc] [Fintype qc] [Fintype mc]
    [DecidableEq nc] [DecidableEq qc] [DecidableEq mc]
    (Qc : Matrix nc qc ℝ) (Bc : Matrix nc mc ℝ) (hBc : Bcᵀ * Bc = 1)
    (hPBc : Bc * Bcᵀ = NuisanceProjection.projection Qc)
    (Dc : Matrix nc p ℝ) (Yc : Matrix nc c ℝ)
    (hDc : NuisanceProjection.projection Qc * Dc = Dc)
    (hYc : NuisanceProjection.projection Qc * Yc = Yc)
    (lam : ℝ) (hlam : 0 < lam)
    (C : Matrix p p ℝ) (hC : C * Cᵀ = ProjectedRegression.sourceRowCov Dc lam)
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ)
    (hB : Bᵀ * B = 1) (hPB : B * Bᵀ = NuisanceProjection.projection Q)
    (D : Matrix n p ℝ) (hD : NuisanceProjection.projection Q * D = D)
    (L Sigma : Matrix c c ℝ) (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef)
    (T : Matrix p p ℝ) (hT : T * Tᵀ = GaussianLowRank.smallCovariance (D * C))
    (htri : T.BlockTriangular OrderDual.toDual) (hdiag : ∀ i, 0 < T i i)
    (Y : Matrix n c ℝ) (hY : NuisanceProjection.projection Q * Y = Y) :
    IsMarkovKernel (GaussianJointLaw.posteriorKernel (Bcᵀ * Dc)
      (Bcᵀ * NuisanceProjection.projection Qc * Bc) lam C⁻¹ L) ∧
    Real.log (∫ x : (p × c) → ℝ,
      likelihoodDensity (Bᵀ * D) (matricize x) (1 : Matrix m m ℝ) L
        (vectorize (Bᵀ * Y))
      ∂GaussianJointLaw.posteriorKernel (Bcᵀ * Dc)
        (Bcᵀ * NuisanceProjection.projection Qc * Bc) lam C⁻¹ L
        (vectorize (Bcᵀ * Yc))) =
      GaussianLowRank.sourceCandidateScore (D * C) T
        (spatialWhiten L (Y - D * ProjectedRegression.sourceMean Dc Yc lam)) +
        gaussianLogConstant (Fintype.card m) L := by
  have hC' : C * Cᵀ = rowCov (Bcᵀ * Dc) (1 : Matrix mc mc ℝ) lam := by
    rw [← ProjectedRegression.source_rowCov_is_declared_rowCov,
      ProjectedRegression.compressed_rowCov_equals_source Qc Bc hPBc Dc hDc lam]
    exact hC
  have h := fitted_posterior_predictive_log_score (Bcᵀ * Dc) (Bcᵀ * Yc) 1 lam C hC'
    Matrix.PosDef.one hlam Q B hB hPB D hD L Sigma hL hSigma T hT htri hdiag Y hY
  rw [← ProjectedRegression.source_mean_is_declared_mean,
    ProjectedRegression.compressed_mean_equals_source Qc Bc hPBc Dc Yc hDc hYc lam] at h
  have hKobs : Bcᵀ * NuisanceProjection.projection Qc * Bc = 1 := by
    simp only [← hPBc, ← Matrix.mul_assoc, hBc, Matrix.one_mul]
  rw [hKobs]
  exact h

end HeldCoordinates

end

end N2P3.ContinuousPosteriorModel
