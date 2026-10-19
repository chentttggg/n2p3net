import N2P3.ContinuousPosteriorModel
import N2P3.CandidateProbability
import N2P3.CandidateTie

/-!
# Actual projected Gaussian likelihoods feed the finite candidate decision

The finite normalization theorem is instantiated with the actual integral
against the disintegrating calibration posterior. Positivity and the common
log-density constant are derived, not assumed in the candidate model. Each
candidate shares the same held observation coordinates, spatial covariance,
calibration fit and candidate vocabulary. Only its public design changes.

This is the declared fixed-parameter, independent held-noise model. The exact
factor equations remain numerical interfaces; no real EEG model validity or
floating-point correspondence is inferred from this candidate normalization.
-/

namespace N2P3.GaussianCandidates

open Matrix MeasureTheory ProbabilityTheory
open N2P3.ScaleGauge N2P3.MatrixNormalCoordinates N2P3.BayesDensity
open N2P3.RetainedGaussianLaw N2P3.RetainedMatrixMeasure
open scoped Matrix ComplexOrder Kronecker BigOperators

noncomputable section

variable {n q m p c : Type*} [Fintype n] [Fintype q] [Fintype m]
  [Fintype p] [Fintype c] [DecidableEq n] [DecidableEq m]
  [DecidableEq p] [DecidableEq c]

/-- Integrate the actual coefficient density; no log/integral conclusion is
an input. A predictive covariance factor is constructed from positivity. -/
theorem coefficient_integral_eq_candidatePDF
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ)
    (hB : Bᵀ * B = 1) (hPB : B * Bᵀ = NuisanceProjection.projection Q)
    (D : Matrix n p ℝ) (center : Matrix p c ℝ) (P C : Matrix p p ℝ)
    (hP : P.PosDef) (hC : C * Cᵀ = P)
    (L Sigma : Matrix c c ℝ) (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef)
    (G : Matrix m m ℝ) (hG : G * Gᵀ = GaussianLowRank.fullCovariance (Bᵀ * (D * C)))
    (Y : Matrix n c ℝ) (hY : NuisanceProjection.projection Q * Y = Y) :
    (∫ x : (p × c) → ℝ,
      likelihoodDensity (Bᵀ * D) (matricize x) (1 : Matrix m m ℝ) L
        (vectorize (Bᵀ * Y)) * matrixPDF C⁻¹ L center x) =
      candidatePDF B G L (D * center) (vectorize Y) := by
  let Db := Bᵀ * D
  have hA : P⁻¹.PosDef := hP.inv
  have hpred : GaussianConvolution.predictiveCovariance Db P⁻¹ 1 =
      GaussianLowRank.fullCovariance (Bᵀ * (D * C)) := by
    rw [GaussianConvolution.predictiveCovariance,
      Matrix.nonsing_inv_nonsing_inv P (positive_definite_det_unit hP), ← hC]
    simp only [GaussianLowRank.fullCovariance, Db, Matrix.transpose_mul, Matrix.mul_assoc]
  obtain ⟨F, hF⟩ := covariance_factor_exists _
    (GaussianConvolution.updatedPrecision_posDef Db P⁻¹ 1 hA Matrix.PosDef.one)
  let M : GaussianConvolution.Model (n := m) (p := p) (c := c) :=
    { D := Db, A := P⁻¹, K := 1, Sigma := Sigma, mean := center,
      R := C⁻¹, B := Fᵀ, H := 1, G := G, L := L,
      hA := hA, hK := Matrix.PosDef.one, hSigma := hSigma,
      hR := likelihood_row_factor_gram C P hC,
      hB := by simpa only [Matrix.transpose_transpose] using hF,
      hH := by simp, hG := hG.trans hpred.symm, hL := hL }
  have hi := M.predictive_marginal_explicit (Bᵀ * Y)
  change (∫ x : (p × c) → ℝ,
    likelihoodDensity Db (matricize x) 1 L (vectorize (Bᵀ * Y)) *
      matrixPDF C⁻¹ L center x) = matrixPDF G⁻¹ L (Db * center) (vectorize (Bᵀ * Y)) at hi
  have hrec : channelLift B *ᵥ vectorize (Bᵀ * Y) = vectorize Y := by
    rw [channelLift_vectorize, ← Matrix.mul_assoc, hPB, hY]
  have hp := candidatePDF_on_embedding B hB G L (D * center) (vectorize (Bᵀ * Y))
  rw [hrec, ← Matrix.mul_assoc] at hp
  exact hi.trans hp.symm

variable {nc qc mc κ : Type*} [Fintype nc] [Fintype qc] [Fintype mc] [Fintype κ]
  [DecidableEq nc] [DecidableEq mc] [LinearOrder p]

structure Model where
  Qc : Matrix nc qc ℝ
  Bc : Matrix nc mc ℝ
  hBc : Bcᵀ * Bc = 1
  hPBc : Bc * Bcᵀ = NuisanceProjection.projection Qc
  Dc : Matrix nc p ℝ
  Yc : Matrix nc c ℝ
  hDc : NuisanceProjection.projection Qc * Dc = Dc
  hYc : NuisanceProjection.projection Qc * Yc = Yc
  lam : ℝ
  hlam : 0 < lam
  C : Matrix p p ℝ
  hC : C * Cᵀ = ProjectedRegression.sourceRowCov Dc lam
  Q : Matrix n q ℝ
  B : Matrix n m ℝ
  hB : Bᵀ * B = 1
  hPB : B * Bᵀ = NuisanceProjection.projection Q
  D : κ → Matrix n p ℝ
  hD : ∀ i, NuisanceProjection.projection Q * D i = D i
  L : Matrix c c ℝ
  Sigma : Matrix c c ℝ
  hL : L * Lᵀ = Sigma
  hSigma : Sigma.PosDef
  T : κ → Matrix p p ℝ
  hT : ∀ i, T i * (T i)ᵀ = GaussianLowRank.smallCovariance (D i * C)
  htri : ∀ i, (T i).BlockTriangular OrderDual.toDual
  hdiag : ∀ i j, 0 < T i j j
  Y : Matrix n c ℝ
  hY : NuisanceProjection.projection Q * Y = Y

namespace Model

variable (M : Model (n := n) (q := q) (m := m) (p := p) (c := c)
  (nc := nc) (qc := qc) (mc := mc) (κ := κ))

def sourceCenter : Matrix p c ℝ := ProjectedRegression.sourceMean M.Dc M.Yc M.lam

def calibrationCov : Matrix mc mc ℝ := M.Bcᵀ * NuisanceProjection.projection M.Qc * M.Bc

def posteriorMeasure : Measure ((p × c) → ℝ) :=
  GaussianJointLaw.posteriorKernel (M.Bcᵀ * M.Dc) M.calibrationCov M.lam M.C⁻¹ M.L
    (vectorize (M.Bcᵀ * M.Yc))

def evidence (i : κ) : ℝ :=
  GaussianLowRank.sourceCandidateScore (M.D i * M.C) (M.T i)
    (spatialWhiten M.L (M.Y - M.D i * M.sourceCenter))

def density (i : κ) : ℝ :=
  ∫ x : (p × c) → ℝ,
    likelihoodDensity (M.Bᵀ * M.D i) (matricize x) (1 : Matrix m m ℝ) M.L
      (vectorize (M.Bᵀ * M.Y)) ∂M.posteriorMeasure

def observationFactor (i : κ) : Matrix m m ℝ :=
  Classical.choose (covariance_factor_exists _
    (GaussianLowRank.fullCovariance_posDef (M.Bᵀ * (M.D i * M.C))))

omit [DecidableEq c] [Fintype κ] in
theorem observationFactor_cov (i : κ) :
    M.observationFactor i * (M.observationFactor i)ᵀ =
      GaussianLowRank.fullCovariance (M.Bᵀ * (M.D i * M.C)) :=
  Classical.choose_spec (covariance_factor_exists _
    (GaussianLowRank.fullCovariance_posDef (M.Bᵀ * (M.D i * M.C))))

omit [DecidableEq c] [Fintype κ] in
theorem calibrationCov_one : M.calibrationCov = 1 := by
  simp only [calibrationCov, ← M.hPBc, ← Matrix.mul_assoc, M.hBc, Matrix.one_mul]

omit [Fintype κ] in
theorem posteriorMeasure_as_matrixMeasure :
    M.posteriorMeasure = matrixKernelMeasure M.C⁻¹ M.L M.sourceCenter := by
  rw [posteriorMeasure, M.calibrationCov_one, GaussianJointLaw.posteriorKernel_apply,
    ← ProjectedRegression.source_mean_is_declared_mean]
  change matrixKernelMeasure M.C⁻¹ M.L
    (ProjectedRegression.sourceMean (M.Bcᵀ * M.Dc) (M.Bcᵀ * M.Yc) M.lam) = _
  rw [ProjectedRegression.compressed_mean_equals_source M.Qc M.Bc M.hPBc
    M.Dc M.Yc M.hDc M.hYc M.lam]
  rfl

omit [Fintype κ] in
/-- Positivity is obtained from this actual integral identity, before any
log-score normalization is used. -/
theorem density_eq_candidatePDF (i : κ) :
    M.density i = candidatePDF M.B (M.observationFactor i) M.L
      (M.D i * M.sourceCenter) (vectorize M.Y) := by
  rw [density, M.posteriorMeasure_as_matrixMeasure]
  change (∫ x : (p × c) → ℝ,
    likelihoodDensity (M.Bᵀ * M.D i) (matricize x) 1 M.L (vectorize (M.Bᵀ * M.Y))
    ∂volume.withDensity (fun x => ENNReal.ofReal (matrixPDF M.C⁻¹ M.L M.sourceCenter x))) = _
  have hmeas : Measurable (fun x : (p × c) → ℝ =>
      ENNReal.ofReal (matrixPDF M.C⁻¹ M.L M.sourceCenter x)) :=
    (GaussianJointLaw.matrixPDF_continuous_composed M.C⁻¹ M.L
      (continuous_const : Continuous (fun _ : (p × c) → ℝ => M.sourceCenter))
      continuous_id).measurable.ennreal_ofReal
  rw [integral_withDensity_eq_integral_toReal_smul (μ := volume) hmeas
    (Filter.Eventually.of_forall (fun _ => ENNReal.ofReal_lt_top))]
  simp_rw [ENNReal.toReal_ofReal (matrixPDF_nonnegative M.C⁻¹ M.L _ _), smul_eq_mul]
  have h := coefficient_integral_eq_candidatePDF M.Q M.B M.hB M.hPB (M.D i) M.sourceCenter
    (ProjectedRegression.sourceRowCov M.Dc M.lam) M.C
    (ProjectedRegression.source_rowCov_positive_definite M.Dc M.lam M.hlam) M.hC
    M.L M.Sigma M.hL M.hSigma (M.observationFactor i) (M.observationFactor_cov i) M.Y M.hY
  simpa only [mul_comm] using h

omit [Fintype κ] in
theorem density_positive (i : κ) : 0 < M.density i := by
  rw [M.density_eq_candidatePDF i]
  exact retainedPDF_positive M.B (M.D i * M.C) (M.observationFactor i) M.L M.Sigma
    (M.observationFactor_cov i) M.hL M.hSigma _

omit [Fintype κ] in
theorem log_density_eq_source (i : κ) :
    Real.log (M.density i) = M.evidence i + gaussianLogConstant (Fintype.card m) M.L := by
  classical
  exact (ContinuousPosteriorModel.projected_source_posterior_predictive_log_score
    M.Qc M.Bc M.hBc M.hPBc M.Dc M.Yc M.hDc M.hYc M.lam M.hlam M.C M.hC
    M.Q M.B M.hB M.hPB (M.D i) (M.hD i) M.L M.Sigma M.hL M.hSigma
    (M.T i) (M.hT i) (M.htri i) (M.hdiag i) M.Y M.hY).2

theorem source_probability_is_normalized_predictive [Nonempty κ] (i : κ) :
    CandidateProbability.probability M.evidence i =
      CandidateProbability.normalizedDensity M.density i :=
  CandidateProbability.probability_eq_normalized_density M.evidence M.density
    (gaussianLogConstant (Fintype.card m) M.L) M.density_positive M.log_density_eq_source i

theorem explicit_candidate_prior (prior : κ → ℝ) (i : κ) :
    CandidateProbability.posterior prior M.evidence i =
      prior i * M.density i / ∑ j, prior j * M.density j :=
  CandidateProbability.posterior_eq_weighted_density prior M.evidence M.density
    (gaussianLogConstant (Fintype.card m) M.L) M.density_positive M.log_density_eq_source i

/-- The preceding weight identity becomes a probability statement only for
an explicit normalized nonnegative candidate prior. -/
theorem proper_candidate_posterior (prior : κ → ℝ) (hprior : CandidateProbability.IsPrior prior) :
    (∀ i, 0 ≤ CandidateProbability.posterior prior M.evidence i) ∧
      (∑ i, CandidateProbability.posterior prior M.evidence i) = 1 ∧
      ∀ i, CandidateProbability.posterior prior M.evidence i =
        prior i * M.density i / ∑ j, prior j * M.density j :=
  ⟨CandidateProbability.posterior_nonnegative prior M.evidence hprior,
    CandidateProbability.posterior_sum_one prior M.evidence hprior,
    M.explicit_candidate_prior prior⟩

theorem source_nll_is_predictive_nll [Nonempty κ] (truth : κ) :
    CandidateProbability.nll M.evidence truth =
      -Real.log (CandidateProbability.normalizedDensity M.density truth) := by
  rw [CandidateProbability.nll_eq_neg_log_probability,
    M.source_probability_is_normalized_predictive truth]

theorem source_winner_is_density_winner [Nonempty κ] [LinearOrder κ] (tol : ℝ) :
    CandidateTie.winner M.evidence tol =
      CandidateTie.winner (fun i => Real.log (M.density i)) tol := by
  simp_rw [M.log_density_eq_source]
  exact (CandidateTie.winner_add_constant M.evidence tol _).symm

end Model

abbrev One := Fin 1

/-- A concrete legal instance rules out mutually inconsistent model premises.
Its two public hypotheses happen to have identical zero designs. -/
def zeroDesignModel : Model (n := One) (q := Fin 0) (m := One) (p := One) (c := One)
    (nc := One) (qc := Fin 0) (mc := One) (κ := Fin 2) where
  Qc := 0
  Bc := 1
  hBc := by simp
  hPBc := by simp [NuisanceProjection.projection]
  Dc := 0
  Yc := 0
  hDc := by simp
  hYc := by simp
  lam := 1
  hlam := by norm_num
  C := 1
  hC := by simp [ProjectedRegression.sourceRowCov, ProjectedRegression.sourcePrecision]
  Q := 0
  B := 1
  hB := by simp
  hPB := by simp [NuisanceProjection.projection]
  D := fun _ => 0
  hD := by intro i; simp
  L := 1
  Sigma := 1
  hL := by simp
  hSigma := Matrix.PosDef.one
  T := fun _ => 1
  hT := by intro i; simp [GaussianLowRank.smallCovariance]
  htri := fun _ => Matrix.blockTriangular_one
  hdiag := by intro i j; simp
  Y := 0
  hY := by simp

theorem zeroDesignModel_actual_likelihood_positive (i : Fin 2) :
    0 < zeroDesignModel.density i := zeroDesignModel.density_positive i

end

end N2P3.GaussianCandidates
