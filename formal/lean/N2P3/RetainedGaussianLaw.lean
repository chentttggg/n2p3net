import N2P3.RetainedMatrixMeasure
import N2P3.BayesDensity
import Mathlib.Analysis.Matrix.Order

/-!
# A normalized Gaussian law on the actual retained observation space

The row covariance is the actual compressed `I + (BᵀU)(BᵀU)ᵀ` and the channel
covariance is `L Lᵀ`. An exact covariance factor is an explicit interface; no
normalization or score identity is assumed. The measure is a pushforward of
the independently normalized nonredundant Gaussian, with a proved density
relative to the intrinsic retained volume. All statements are exact-real.
-/

namespace N2P3.RetainedGaussianLaw

open Matrix MeasureTheory
open N2P3.RetainedMeasure N2P3.RetainedMatrixMeasure N2P3.MatrixNormalCoordinates
open scoped Matrix BigOperators Kronecker ENNReal ComplexOrder

noncomputable section

variable {n q m r c : Type*}
variable [Fintype n] [Fintype q] [Fintype m] [Fintype r] [Fintype c]
variable [DecidableEq n] [DecidableEq q] [DecidableEq m] [DecidableEq r] [DecidableEq c]

def retainedPDF (B : Matrix n m ℝ) (H : Matrix m m ℝ) (L : Matrix c c ℝ)
    (x : (n × c) → ℝ) : ℝ :=
  matrixPDF H⁻¹ L 0 (channelLift Bᵀ *ᵥ x)

def retainedLaw (B : Matrix n m ℝ) (H : Matrix m m ℝ) (L : Matrix c c ℝ) :
    Measure ((n × c) → ℝ) :=
  Measure.map (Matrix.toLin' (channelLift B)) (matrixKernelMeasure H⁻¹ L 0)

theorem matrixPDF_continuous (H : Matrix m m ℝ) (L : Matrix c c ℝ) :
    Continuous (matrixPDF H⁻¹ L (0 : Matrix m c ℝ)) := by
  change Continuous (fun x => |(whiteningMatrix H⁻¹ L).det| *
    N2P3.GaussianMeasure.productPDF (1 / 2) (whiteningMatrix H⁻¹ L *ᵥ (x - vectorize 0)))
  exact continuous_const.mul ((N2P3.GaussianMeasure.productPDF_continuous _).comp
    ((Matrix.toLin' (whiteningMatrix H⁻¹ L)).continuous_on_pi.comp
      (continuous_id.sub continuous_const)))

theorem retainedPDF_continuous (B : Matrix n m ℝ) (H : Matrix m m ℝ) (L : Matrix c c ℝ) :
    Continuous (retainedPDF B H L) :=
  (matrixPDF_continuous H L).comp (Matrix.toLin' (channelLift Bᵀ)).continuous_on_pi

omit [DecidableEq n] in
theorem retainedPDF_nonnegative (B : Matrix n m ℝ) (H : Matrix m m ℝ) (L : Matrix c c ℝ)
    (x : (n × c) → ℝ) : 0 ≤ retainedPDF B H L x :=
  matrixPDF_nonnegative H⁻¹ L 0 _

omit [DecidableEq n] in
theorem retainedPDF_on_embedding (B : Matrix n m ℝ) (hB : Bᵀ * B = 1)
    (H : Matrix m m ℝ) (L : Matrix c c ℝ) (z : (m × c) → ℝ) :
    retainedPDF B H L (channelLift B *ᵥ z) = matrixPDF H⁻¹ L 0 z := by
  unfold retainedPDF
  rw [Matrix.mulVec_mulVec, ← channelLift_mul, hB, channelLift_one, Matrix.one_mulVec]

/-- Genuine pushforward density with respect to embedded retained volume. -/
theorem retainedLaw_withDensity (B : Matrix n m ℝ) (hB : Bᵀ * B = 1)
    (H : Matrix m m ℝ) (L : Matrix c c ℝ) :
    retainedLaw B H L = (coordinateVolume (channelLift B)).withDensity
      (fun x => ENNReal.ofReal (retainedPDF B H L x)) := by
  apply Measure.ext
  intro s hs
  rw [retainedLaw, Measure.map_apply (matrix_embedding_measurable _) hs,
    matrixKernelMeasure, N2P3.GaussianMeasure.affineGaussianMeasure,
    withDensity_apply _ ((matrix_embedding_measurable _ ) hs), withDensity_apply _ hs,
    coordinateVolume, setLIntegral_map hs
      ((retainedPDF_continuous B H L).measurable.ennreal_ofReal)
      (matrix_embedding_measurable _)]
  apply lintegral_congr
  intro z
  congr 1
  exact (retainedPDF_on_embedding B hB H L z).symm

omit [DecidableEq n] [DecidableEq r] in
/-- The normalization is inherited from the proved Gaussian integral, not an input. -/
theorem retainedLaw_probability (B : Matrix n m ℝ) (U : Matrix n r ℝ)
    (H : Matrix m m ℝ) (L Sigma : Matrix c c ℝ)
    (hH : H * Hᵀ = GaussianLowRank.fullCovariance (Bᵀ * U))
    (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef) :
    IsProbabilityMeasure (retainedLaw B H L) := by
  have hHu := N2P3.ScaleGauge.whiteningFactor_unit H _ hH
    (GaussianLowRank.fullCovariance_posDef (Bᵀ * U))
  haveI := matrixKernelMeasure_probability H⁻¹ L (0 : Matrix m c ℝ)
    (Matrix.isUnit_nonsing_inv_det H hHu)
    (N2P3.ScaleGauge.whiteningFactor_unit L Sigma hL hSigma)
  exact Measure.isProbabilityMeasure_map (matrix_embedding_measurable _).aemeasurable

omit [DecidableEq r] in
theorem retainedDensity_mass_one (B : Matrix n m ℝ) (hB : Bᵀ * B = 1)
    (U : Matrix n r ℝ) (H : Matrix m m ℝ) (L Sigma : Matrix c c ℝ)
    (hH : H * Hᵀ = GaussianLowRank.fullCovariance (Bᵀ * U))
    (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef) :
    (∫⁻ x, ENNReal.ofReal (retainedPDF B H L x) ∂coordinateVolume (channelLift B)) = 1 := by
  haveI := retainedLaw_probability B U H L Sigma hH hL hSigma
  have hmass := measure_univ (μ := retainedLaw B H L)
  rw [retainedLaw_withDensity B hB, withDensity_apply _ MeasurableSet.univ,
    Measure.restrict_univ] at hmass
  exact hmass

omit [DecidableEq q] in
theorem retainedLaw_supported (Q : Matrix n q ℝ) (B : Matrix n m ℝ)
    (hB : Bᵀ * B = 1) (hPB : B * Bᵀ = NuisanceProjection.projection Q)
    (H : Matrix m m ℝ) (L : Matrix c c ℝ) :
    retainedLaw B H L {x | NuisanceProjection.projection Q * matricize x ≠ matricize x} = 0 := by
  rw [retainedLaw_withDensity B hB]
  exact (withDensity_absolutelyContinuous _ _)
    (matrixCoordinateVolume_supported Q B hB hPB)

def spatialWhiten (L : Matrix c c ℝ) (E : Matrix n c ℝ) : Matrix n c ℝ := E * (L⁻¹)ᵀ

def gaussianLogConstant (rows : ℕ) (L : Matrix c c ℝ) : ℝ :=
  (rows : ℝ) * Real.log |(L⁻¹).det| -
    (rows * Fintype.card c : ℕ) * Real.log (Real.sqrt (2 * Real.pi))

theorem inverse_factor_gram (H K : Matrix m m ℝ) (hH : H * Hᵀ = K) :
    (H⁻¹)ᵀ * H⁻¹ = K⁻¹ := by
  rw [← hH, Matrix.mul_inv_rev, ← Matrix.transpose_nonsing_inv]

omit [DecidableEq m] in
theorem weightedEnergy_as_whitened_channels
    (K : Matrix m m ℝ) (L Sigma : Matrix c c ℝ) (E : Matrix m c ℝ)
    (hL : L * Lᵀ = Sigma) :
    N2P3.GaussianConditioning.weightedEnergy Sigma (Eᵀ * K * E) =
      ∑ channel, (fun i => spatialWhiten L E i channel) ⬝ᵥ
        (K *ᵥ (fun i => spatialWhiten L E i channel)) := by
  rw [N2P3.GaussianConditioning.weightedEnergy_as_congruence Sigma L _ hL]
  have htrace : L⁻¹ * (Eᵀ * K * E) * (L⁻¹)ᵀ =
      (spatialWhiten L E)ᵀ * K * spatialWhiten L E := by
    simp only [spatialWhiten, Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc]
  rw [htrace]
  simp only [Matrix.trace, Matrix.diag_apply, Matrix.mul_apply, Matrix.transpose_apply,
    dotProduct, Matrix.mulVec, Finset.sum_mul]
  congr 1
  ext channel
  rw [Finset.sum_comm]
  simp only [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

omit [DecidableEq r] in
theorem log_jacobianNormalizer_covariance
    (U : Matrix m r ℝ) (H : Matrix m m ℝ) (L Sigma : Matrix c c ℝ)
    (hH : H * Hᵀ = GaussianLowRank.fullCovariance U)
    (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef) :
    Real.log (jacobianNormalizer H⁻¹ L) =
      -(1 / 2 : ℝ) * (Fintype.card c : ℝ) *
        Real.log (GaussianLowRank.fullCovariance U).det + gaussianLogConstant (Fintype.card m) L := by
  have hHu := N2P3.ScaleGauge.whiteningFactor_unit H _ hH (GaussianLowRank.fullCovariance_posDef U)
  have hLu := N2P3.ScaleGauge.whiteningFactor_unit L Sigma hL hSigma
  have hHne : H.det ≠ 0 := isUnit_iff_ne_zero.mp hHu
  have hLine : (L⁻¹).det ≠ 0 := isUnit_iff_ne_zero.mp (Matrix.isUnit_nonsing_inv_det L hLu)
  have hHine : (H⁻¹).det ≠ 0 := isUnit_iff_ne_zero.mp (Matrix.isUnit_nonsing_inv_det H hHu)
  have hsqrt : Real.sqrt (2 * Real.pi) ≠ 0 := ne_of_gt
    (Real.sqrt_pos.mpr (mul_pos (by norm_num) Real.pi_pos))
  have hKlog : Real.log (GaussianLowRank.fullCovariance U).det = 2 * Real.log H.det := by
    rw [← hH, Matrix.det_mul, Matrix.det_transpose, Real.log_mul hHne hHne]
    ring
  unfold jacobianNormalizer gaussianLogConstant
  rw [whiteningMatrix_abs_det, Real.log_mul
      (mul_ne_zero (pow_ne_zero _ (abs_ne_zero.mpr hHine)) (pow_ne_zero _ (abs_ne_zero.mpr hLine)))
      (pow_ne_zero _ (inv_ne_zero hsqrt)),
    Real.log_mul (pow_ne_zero _ (abs_ne_zero.mpr hHine)) (pow_ne_zero _ (abs_ne_zero.mpr hLine)),
    Real.log_pow, Real.log_pow, Real.log_pow, Real.log_abs, Real.log_abs,
    Matrix.det_nonsing_inv, Ring.inverse_eq_inv, Real.log_inv, Real.log_inv, hKlog,
    Fintype.card_prod, Nat.cast_mul]
  ring

omit [DecidableEq r] in
/-- The normalized Gaussian log density contains precisely the candidate score
and a constant depending only on retained dimension and spatial covariance. -/
theorem matrixPDF_log_equals_score
    (U : Matrix m r ℝ) (H : Matrix m m ℝ) (L Sigma : Matrix c c ℝ) (E : Matrix m c ℝ)
    (hH : H * Hᵀ = GaussianLowRank.fullCovariance U)
    (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef) :
    Real.log (matrixPDF H⁻¹ L 0 (vectorize E)) =
      GaussianLowRank.fullCandidateScore U (spatialWhiten L E) +
        gaussianLogConstant (Fintype.card m) L := by
  have hHu := N2P3.ScaleGauge.whiteningFactor_unit H _ hH (GaussianLowRank.fullCovariance_posDef U)
  have hJ := N2P3.BayesDensity.jacobianNormalizer_positive H⁻¹ L
    (Matrix.isUnit_nonsing_inv_det H hHu) (N2P3.ScaleGauge.whiteningFactor_unit L Sigma hL hSigma)
  rw [matrixPDF_as_trace_kernel H⁻¹ (GaussianLowRank.fullCovariance U)⁻¹ L Sigma 0 E
      (inverse_factor_gram H _ hH) hL, sub_zero,
    Real.log_mul (ne_of_gt hJ) (ne_of_gt (Real.exp_pos _)), Real.log_exp,
    log_jacobianNormalizer_covariance U H L Sigma hH hL hSigma,
    weightedEnergy_as_whitened_channels _ L Sigma E hL]
  unfold GaussianLowRank.fullCandidateScore
  ring

omit [DecidableEq q] in
theorem retainedPDF_log_equals_full_score
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (hPB : B * Bᵀ = NuisanceProjection.projection Q)
    (U : Matrix n r ℝ) (hU : NuisanceProjection.projection Q * U = U)
    (E : Matrix n c ℝ) (hE : NuisanceProjection.projection Q * E = E)
    (H : Matrix m m ℝ) (L Sigma : Matrix c c ℝ)
    (hH : H * Hᵀ = GaussianLowRank.fullCovariance (Bᵀ * U))
    (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef) :
    Real.log (retainedPDF B H L (vectorize E)) =
      GaussianLowRank.fullCandidateScore U (spatialWhiten L E) +
        gaussianLogConstant (Fintype.card m) L := by
  unfold retainedPDF
  rw [channelLift_vectorize, matrixPDF_log_equals_score (Bᵀ * U) H L Sigma (Bᵀ * E) hH hL hSigma]
  have hwhite : NuisanceProjection.projection Q * spatialWhiten L E = spatialWhiten L E := by
    rw [spatialWhiten, ← Matrix.mul_assoc, hE]
  have hcoord : spatialWhiten L (Bᵀ * E) = Bᵀ * spatialWhiten L E := Matrix.mul_assoc _ _ _
  rw [hcoord, ← NuisanceProjection.identity_extension_score_equals_frame_extension_score
    Q B hPB U hU (spatialWhiten L E) hwhite]

omit [DecidableEq q] in
/-- The exact source's small factor, nonnegative quadratic and clamp are
connected to the normalized density, with all observed channels counted. -/
theorem retainedPDF_log_equals_source_score [LinearOrder r]
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (hPB : B * Bᵀ = NuisanceProjection.projection Q)
    (U : Matrix n r ℝ) (hU : NuisanceProjection.projection Q * U = U)
    (E : Matrix n c ℝ) (hE : NuisanceProjection.projection Q * E = E)
    (H : Matrix m m ℝ) (L Sigma : Matrix c c ℝ)
    (hH : H * Hᵀ = GaussianLowRank.fullCovariance (Bᵀ * U))
    (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef)
    (T : Matrix r r ℝ) (hT : T * Tᵀ = GaussianLowRank.smallCovariance U)
    (htri : T.BlockTriangular OrderDual.toDual) (hdiag : ∀ i, 0 < T i i) :
    Real.log (retainedPDF B H L (vectorize E)) =
      GaussianLowRank.sourceCandidateScore U T (spatialWhiten L E) +
        gaussianLogConstant (Fintype.card m) L := by
  rw [GaussianLowRank.source_candidate_score_equals_full U T _ hT htri hdiag]
  exact retainedPDF_log_equals_full_score Q B hPB U hU E hE H L Sigma hH hL hSigma

omit [DecidableEq n] [DecidableEq r] in
theorem retainedPDF_positive (B : Matrix n m ℝ) (U : Matrix n r ℝ)
    (H : Matrix m m ℝ) (L Sigma : Matrix c c ℝ)
    (hH : H * Hᵀ = GaussianLowRank.fullCovariance (Bᵀ * U))
    (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef) (x : (n × c) → ℝ) :
    0 < retainedPDF B H L x := by
  have hHu := N2P3.ScaleGauge.whiteningFactor_unit H _ hH
    (GaussianLowRank.fullCovariance_posDef (Bᵀ * U))
  unfold retainedPDF matrixPDF
  rw [N2P3.GaussianMeasure.affinePDF_as_quadratic_kernel]
  apply mul_pos
  · exact mul_pos (abs_pos.mpr (whiteningMatrix_det_ne_zero H⁻¹ L
      (Matrix.isUnit_nonsing_inv_det H hHu)
      (N2P3.ScaleGauge.whiteningFactor_unit L Sigma hL hSigma)))
      (pow_pos (inv_pos.mpr (Real.sqrt_pos.mpr (div_pos Real.pi_pos (by norm_num)))) _)
  · exact Real.exp_pos _

omit [DecidableEq q] in
theorem retainedPDF_basis_independent_on_support
    (Q : Matrix n q ℝ) (B C : Matrix n m ℝ)
    (hPB : B * Bᵀ = NuisanceProjection.projection Q)
    (hPC : C * Cᵀ = NuisanceProjection.projection Q)
    (U : Matrix n r ℝ) (hU : NuisanceProjection.projection Q * U = U)
    (H J : Matrix m m ℝ) (L Sigma : Matrix c c ℝ)
    (hH : H * Hᵀ = GaussianLowRank.fullCovariance (Bᵀ * U))
    (hJ : J * Jᵀ = GaussianLowRank.fullCovariance (Cᵀ * U))
    (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef)
    (E : Matrix n c ℝ) (hE : NuisanceProjection.projection Q * E = E) :
    retainedPDF B H L (vectorize E) = retainedPDF C J L (vectorize E) := by
  have hlog : Real.log (retainedPDF B H L (vectorize E)) =
      Real.log (retainedPDF C J L (vectorize E)) := by
    rw [retainedPDF_log_equals_full_score Q B hPB U hU E hE H L Sigma hH hL hSigma,
      retainedPDF_log_equals_full_score Q C hPC U hU E hE J L Sigma hJ hL hSigma]
  have h := congrArg Real.exp hlog
  rwa [Real.exp_log (retainedPDF_positive B U H L Sigma hH hL hSigma _),
    Real.exp_log (retainedPDF_positive C U J L Sigma hJ hL hSigma _)] at h

omit [DecidableEq q] in
/-- The Gaussian probability law, not merely the volume, is independent of
the complete orthonormal retained basis and of its legal exact factor. -/
theorem retainedLaw_basis_independent
    (Q : Matrix n q ℝ) (B C : Matrix n m ℝ)
    (hB : Bᵀ * B = 1) (hC : Cᵀ * C = 1)
    (hPB : B * Bᵀ = NuisanceProjection.projection Q)
    (hPC : C * Cᵀ = NuisanceProjection.projection Q)
    (U : Matrix n r ℝ) (hU : NuisanceProjection.projection Q * U = U)
    (H J : Matrix m m ℝ) (L Sigma : Matrix c c ℝ)
    (hH : H * Hᵀ = GaussianLowRank.fullCovariance (Bᵀ * U))
    (hJ : J * Jᵀ = GaussianLowRank.fullCovariance (Cᵀ * U))
    (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef) :
    retainedLaw B H L = retainedLaw C J L := by
  rw [retainedLaw_withDensity B hB, retainedLaw_withDensity C hC,
    ← matrixCoordinateVolume_basis_independent Q B C hB hC hPB hPC]
  apply withDensity_congr_ae
  have hsupport : ∀ᵐ x ∂coordinateVolume (channelLift (c := c) B),
      NuisanceProjection.projection Q * matricize x = matricize x :=
    ae_iff.mpr (matrixCoordinateVolume_supported Q B hB hPB)
  filter_upwards [hsupport] with x hx
  congr 1
  exact retainedPDF_basis_independent_on_support Q B C hPB hPC U hU H J L Sigma
    hH hJ hL hSigma (matricize x) hx

/-- A candidate with nonzero retained mean is a genuine observation measure. -/
def candidatePDF (B : Matrix n m ℝ) (H : Matrix m m ℝ) (L : Matrix c c ℝ)
    (M : Matrix n c ℝ) (x : (n × c) → ℝ) : ℝ :=
  retainedPDF B H L (x - vectorize M)

def candidateLaw (B : Matrix n m ℝ) (H : Matrix m m ℝ) (L : Matrix c c ℝ)
    (M : Matrix n c ℝ) : Measure ((n × c) → ℝ) :=
  Measure.map (Matrix.toLin' (channelLift B)) (matrixKernelMeasure H⁻¹ L (Bᵀ * M))

omit [DecidableEq n] in
theorem candidatePDF_on_embedding (B : Matrix n m ℝ) (hB : Bᵀ * B = 1)
    (H : Matrix m m ℝ) (L : Matrix c c ℝ) (M : Matrix n c ℝ) (z : (m × c) → ℝ) :
    candidatePDF B H L M (channelLift B *ᵥ z) = matrixPDF H⁻¹ L (Bᵀ * M) z := by
  unfold candidatePDF retainedPDF
  rw [Matrix.mulVec_sub, Matrix.mulVec_mulVec, ← channelLift_mul, hB,
    channelLift_one, Matrix.one_mulVec, channelLift_vectorize]
  have hz : vectorize (0 : Matrix m c ℝ) = (0 : (m × c) → ℝ) := rfl
  simp only [matrixPDF, N2P3.GaussianMeasure.affinePDF, hz, sub_zero]

theorem candidateLaw_withDensity (B : Matrix n m ℝ) (hB : Bᵀ * B = 1)
    (H : Matrix m m ℝ) (L : Matrix c c ℝ) (M : Matrix n c ℝ) :
    candidateLaw B H L M = (coordinateVolume (channelLift B)).withDensity
      (fun x => ENNReal.ofReal (candidatePDF B H L M x)) := by
  have hc : Continuous (candidatePDF B H L M) :=
    (retainedPDF_continuous B H L).comp (continuous_id.sub continuous_const)
  apply Measure.ext
  intro s hs
  rw [candidateLaw, Measure.map_apply (matrix_embedding_measurable _) hs,
    matrixKernelMeasure, N2P3.GaussianMeasure.affineGaussianMeasure,
    withDensity_apply _ ((matrix_embedding_measurable _ ) hs), withDensity_apply _ hs,
    coordinateVolume, setLIntegral_map hs hc.measurable.ennreal_ofReal (matrix_embedding_measurable _)]
  apply lintegral_congr
  intro z
  congr 1
  exact (candidatePDF_on_embedding B hB H L M z).symm

omit [DecidableEq n] [DecidableEq r] in
theorem candidateLaw_probability (B : Matrix n m ℝ) (U : Matrix n r ℝ)
    (H : Matrix m m ℝ) (L Sigma : Matrix c c ℝ) (M : Matrix n c ℝ)
    (hH : H * Hᵀ = GaussianLowRank.fullCovariance (Bᵀ * U))
    (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef) :
    IsProbabilityMeasure (candidateLaw B H L M) := by
  have hHu := N2P3.ScaleGauge.whiteningFactor_unit H _ hH
    (GaussianLowRank.fullCovariance_posDef (Bᵀ * U))
  haveI := matrixKernelMeasure_probability H⁻¹ L (Bᵀ * M)
    (Matrix.isUnit_nonsing_inv_det H hHu) (N2P3.ScaleGauge.whiteningFactor_unit L Sigma hL hSigma)
  exact Measure.isProbabilityMeasure_map (matrix_embedding_measurable _).aemeasurable

omit [DecidableEq r] in
theorem candidateDensity_mass_one (B : Matrix n m ℝ) (hB : Bᵀ * B = 1)
    (U : Matrix n r ℝ) (H : Matrix m m ℝ) (L Sigma : Matrix c c ℝ) (M : Matrix n c ℝ)
    (hH : H * Hᵀ = GaussianLowRank.fullCovariance (Bᵀ * U))
    (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef) :
    (∫⁻ x, ENNReal.ofReal (candidatePDF B H L M x) ∂coordinateVolume (channelLift B)) = 1 := by
  haveI := candidateLaw_probability B U H L Sigma M hH hL hSigma
  have hmass := measure_univ (μ := candidateLaw B H L M)
  rw [candidateLaw_withDensity B hB, withDensity_apply _ MeasurableSet.univ,
    Measure.restrict_univ] at hmass
  exact hmass

omit [DecidableEq q] in
theorem candidatePDF_log_equals_full_score
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (hPB : B * Bᵀ = NuisanceProjection.projection Q)
    (U : Matrix n r ℝ) (hU : NuisanceProjection.projection Q * U = U)
    (Y M : Matrix n c ℝ) (hY : NuisanceProjection.projection Q * Y = Y)
    (hM : NuisanceProjection.projection Q * M = M)
    (H : Matrix m m ℝ) (L Sigma : Matrix c c ℝ)
    (hH : H * Hᵀ = GaussianLowRank.fullCovariance (Bᵀ * U))
    (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef) :
    Real.log (candidatePDF B H L M (vectorize Y)) =
      GaussianLowRank.fullCandidateScore U (spatialWhiten L (Y - M)) +
        gaussianLogConstant (Fintype.card m) L := by
  unfold candidatePDF
  rw [← vectorize_sub]
  exact retainedPDF_log_equals_full_score Q B hPB U hU (Y - M)
    (by rw [Matrix.mul_sub, hY, hM]) H L Sigma hH hL hSigma

omit [DecidableEq q] in
theorem candidateLaw_basis_independent
    (Q : Matrix n q ℝ) (B C : Matrix n m ℝ)
    (hB : Bᵀ * B = 1) (hC : Cᵀ * C = 1)
    (hPB : B * Bᵀ = NuisanceProjection.projection Q)
    (hPC : C * Cᵀ = NuisanceProjection.projection Q)
    (U : Matrix n r ℝ) (hU : NuisanceProjection.projection Q * U = U)
    (M : Matrix n c ℝ) (hM : NuisanceProjection.projection Q * M = M)
    (H J : Matrix m m ℝ) (L Sigma : Matrix c c ℝ)
    (hH : H * Hᵀ = GaussianLowRank.fullCovariance (Bᵀ * U))
    (hJ : J * Jᵀ = GaussianLowRank.fullCovariance (Cᵀ * U))
    (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef) :
    candidateLaw B H L M = candidateLaw C J L M := by
  rw [candidateLaw_withDensity B hB, candidateLaw_withDensity C hC,
    ← matrixCoordinateVolume_basis_independent Q B C hB hC hPB hPC]
  apply withDensity_congr_ae
  have hsupport : ∀ᵐ x ∂coordinateVolume (channelLift (c := c) B),
      NuisanceProjection.projection Q * matricize x = matricize x :=
    ae_iff.mpr (matrixCoordinateVolume_supported Q B hB hPB)
  filter_upwards [hsupport] with x hx
  congr 1
  unfold candidatePDF
  change retainedPDF B H L (vectorize (matricize x - M)) =
    retainedPDF C J L (vectorize (matricize x - M))
  exact retainedPDF_basis_independent_on_support Q B C hPB hPC U hU H J L Sigma
    hH hJ hL hSigma (matricize x - M) (by rw [Matrix.mul_sub, hx, hM])

omit [DecidableEq q] in
theorem candidateLaw_supported (Q : Matrix n q ℝ) (B : Matrix n m ℝ)
    (hB : Bᵀ * B = 1) (hPB : B * Bᵀ = NuisanceProjection.projection Q)
    (H : Matrix m m ℝ) (L : Matrix c c ℝ) (M : Matrix n c ℝ) :
    candidateLaw B H L M {x | NuisanceProjection.projection Q * matricize x ≠ matricize x} = 0 := by
  rw [candidateLaw_withDensity B hB]
  exact (withDensity_absolutelyContinuous _ _) (matrixCoordinateVolume_supported Q B hB hPB)

omit [DecidableEq q] in
theorem candidatePDF_log_equals_source_score [LinearOrder r]
    (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (hPB : B * Bᵀ = NuisanceProjection.projection Q)
    (U : Matrix n r ℝ) (hU : NuisanceProjection.projection Q * U = U)
    (Y M : Matrix n c ℝ) (hY : NuisanceProjection.projection Q * Y = Y)
    (hM : NuisanceProjection.projection Q * M = M)
    (H : Matrix m m ℝ) (L Sigma : Matrix c c ℝ)
    (hH : H * Hᵀ = GaussianLowRank.fullCovariance (Bᵀ * U))
    (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef)
    (T : Matrix r r ℝ) (hT : T * Tᵀ = GaussianLowRank.smallCovariance U)
    (htri : T.BlockTriangular OrderDual.toDual) (hdiag : ∀ i, 0 < T i i) :
    Real.log (candidatePDF B H L M (vectorize Y)) =
      GaussianLowRank.sourceCandidateScore U T (spatialWhiten L (Y - M)) +
        gaussianLogConstant (Fintype.card m) L := by
  rw [GaussianLowRank.source_candidate_score_equals_full U T _ hT htri hdiag]
  exact candidatePDF_log_equals_full_score Q B hPB U hU Y M hY hM H L Sigma hH hL hSigma

/-- SPD proves that an exact covariance factor exists. This mathematical
square root does not assert a numerical Cholesky or QR implementation. -/
theorem covariance_factor_exists (K : Matrix m m ℝ) (hK : K.PosDef) :
    ∃ H : Matrix m m ℝ, H * Hᵀ = K := by
  obtain ⟨F, _, hF⟩ := Matrix.posDef_iff_eq_conjTranspose_mul_self.mp hK
  refine ⟨Fᵀ, ?_⟩
  simpa only [Matrix.transpose_transpose, Matrix.conjTranspose_eq_transpose_of_trivial] using hF.symm

/-- Full source-scoring probability statement. Q supplies a genuinely
nonredundant B; SPD supplies both row and spatial factors. Gaussian probability,
intrinsic density, mass one, support and source log-score identity are all
derived. The only solver interface left is the actual small source Cholesky
factor and its triangular/positive-diagonal equations. -/
theorem exists_actual_candidate_gaussian [LinearOrder r]
    (Q : Matrix n q ℝ) (hQ : Qᵀ * Q = 1)
    (U : Matrix n r ℝ) (hU : NuisanceProjection.projection Q * U = U)
    (M : Matrix n c ℝ) (hM : NuisanceProjection.projection Q * M = M)
    (Sigma : Matrix c c ℝ) (hSigma : Sigma.PosDef)
    (T : Matrix r r ℝ) (hT : T * Tᵀ = GaussianLowRank.smallCovariance U)
    (htri : T.BlockTriangular OrderDual.toDual) (hdiag : ∀ i, 0 < T i i) :
    ∃ (B : Matrix n (Fin (Fintype.card n - Fintype.card q)) ℝ)
      (H : Matrix (Fin (Fintype.card n - Fintype.card q))
        (Fin (Fintype.card n - Fintype.card q)) ℝ)
      (L : Matrix c c ℝ) (ν : Measure ((n × c) → ℝ)),
      Bᵀ * B = 1 ∧ B * Bᵀ = NuisanceProjection.projection Q ∧
        H * Hᵀ = GaussianLowRank.fullCovariance (Bᵀ * U) ∧ L * Lᵀ = Sigma ∧
        ν = candidateLaw B H L M ∧ IsProbabilityMeasure ν ∧
        ν = (coordinateVolume (channelLift B)).withDensity
          (fun x => ENNReal.ofReal (candidatePDF B H L M x)) ∧
        (∫⁻ x, ENNReal.ofReal (candidatePDF B H L M x)
          ∂coordinateVolume (channelLift B)) = 1 ∧
        ν {x | NuisanceProjection.projection Q * matricize x ≠ matricize x} = 0 ∧
        ∀ Y : Matrix n c ℝ, NuisanceProjection.projection Q * Y = Y →
          Real.log (candidatePDF B H L M (vectorize Y)) =
            GaussianLowRank.sourceCandidateScore U T (spatialWhiten L (Y - M)) +
              gaussianLogConstant (Fintype.card n - Fintype.card q) L := by
  obtain ⟨B, hB, _, hPB⟩ := RetainedBasis.exists_retained_basis Q hQ
  obtain ⟨H, hH⟩ := covariance_factor_exists _ (GaussianLowRank.fullCovariance_posDef (Bᵀ * U))
  obtain ⟨L, hL⟩ := covariance_factor_exists Sigma hSigma
  refine ⟨B, H, L, candidateLaw B H L M, hB, hPB, hH, hL, rfl,
    candidateLaw_probability B U H L Sigma M hH hL hSigma,
    candidateLaw_withDensity B hB H L M,
    candidateDensity_mass_one B hB U H L Sigma M hH hL hSigma,
    candidateLaw_supported Q B hB hPB H L M, ?_⟩
  intro Y hY
  simpa only [Fintype.card_fin] using candidatePDF_log_equals_source_score
    Q B hPB U hU Y M hY hM H L Sigma hH hL hSigma T hT htri hdiag

end

end N2P3.RetainedGaussianLaw
