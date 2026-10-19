import N2P3.OperatorGLSStartup
import N2P3.ContinuousPosteriorModel
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma

/-!
# Independently reset/projected OperatorGLS records and actual ridge producer

The active asset source and E2 caller are fixed in
operator_gls_caller_source_contract_01.json. Each record uses the actual
startup loop and original-n nuisance grid before stacking. The source's QR
and inverse-factor equations derive its mean/rowCov. No whole fitted model,
posterior, or score-equivalence predicate is an input field.
All probability conclusions use the declared independent fixed-parameter
observation model; finite raw scope separation is not stochastic independence.
-/
namespace N2P3.OperatorGLSCaller
open Matrix MeasureTheory ProbabilityTheory
open N2P3.OperatorGLSStartup N2P3.TemporalWhitening N2P3.ScaleGauge
open N2P3.NuisanceProjection N2P3.ProjectedRegression N2P3.RetainedGaussianLaw
open scoped BigOperators Matrix ComplexOrder
noncomputable section

def project {q C : Type*} [Fintype q] (noise : Noise) (n : ℕ)
    (Q : Matrix (Fin (rows noise n)) q ℝ) (x : Signal C) : Matrix (Fin (rows noise n)) C ℝ :=
  white noise n x - Q * (Qᵀ * white noise n x)

theorem actual_project_is_its_orthogonal_projection {q C : Type*} [Fintype q]
    (noise : Noise) (n : ℕ) (Q : Matrix (Fin (rows noise n)) q ℝ) (x : Signal C) :
    project noise n Q x = projection Q * white noise n x := by
  simp [project, projection, Matrix.sub_mul, Matrix.mul_assoc]

theorem actual_projected_values_are_retained {q C : Type*} [Fintype q] [DecidableEq q]
    (noise : Noise) (n : ℕ) (Q : Matrix (Fin (rows noise n)) q ℝ) (orthogonal : Qᵀ * Q = 1)
    (x : Signal C) : projection Q * project noise n Q x = project noise n Q x := by
  rw [actual_project_is_its_orthogonal_projection, ← Matrix.mul_assoc, projection_idempotent Q orthogonal]

theorem actual_original_nuisance_is_removed {q : Type*} [Fintype q] [DecidableEq q]
    (noise : Noise) (n : ℕ) (Q : Matrix (Fin (rows noise n)) q ℝ) (R : Matrix q (Fin 2) ℝ)
    (orthogonal : Qᵀ * Q = 1) (qr : white noise n (originalNuisance n) = Q * R) :
    project noise n Q (originalNuisance n) = 0 := by
  rw [actual_project_is_its_orthogonal_projection, qr, ← Matrix.mul_assoc,
    projection_annihilates_qr_columns Q orthogonal, Matrix.zero_mul]

def sampleProject {n : ℕ} {q C : Type*} [Fintype q] (noise : Noise)
    (Q : Matrix (Fin (rows noise n)) q ℝ) (hasFilter : Bool)
    (F : Matrix (Fin n) (Fin n) ℝ) (X : Matrix (Fin n) C ℝ) :=
  project noise n Q (finiteSignal (processedSamples hasFilter F X))

def designProject {n : ℕ} {q C : Type*} [Fintype q] (noise : Noise)
    (Q : Matrix (Fin (rows noise n)) q ℝ) (hasFilter filterDesign : Bool)
    (F : Matrix (Fin n) (Fin n) ℝ) (D : Matrix (Fin n) C ℝ) :=
  project noise n Q (finiteSignal (processedDesign hasFilter filterDesign F D))

def stack {ι C : Type*} {size : ι → ℕ} (values : (record : ι) → Matrix (Fin (size record)) C ℝ) :
    Matrix ((record : ι) × Fin (size record)) C ℝ := fun row c => values row.1 row.2 c

theorem actual_stack_row_count {ι : Type*} [Fintype ι] (noise : Noise) (sizes : ι → ℕ) :
    Fintype.card ((record : ι) × Fin (rows noise (sizes record))) = ∑ record, rows noise (sizes record) := by
  simp [Fintype.card_sigma]

theorem stack_cross_gram_is_the_sum_of_independent_record_cross_grams
    {ι p c : Type*} [Fintype ι] (size : ι → ℕ)
    (D : (record : ι) → Matrix (Fin (size record)) p ℝ)
    (Y : (record : ι) → Matrix (Fin (size record)) c ℝ) :
    (stack D)ᵀ * stack Y = ∑ record, (D record)ᵀ * Y record := by
  ext i j
  simp [stack, Matrix.mul_apply, Fintype.sum_sigma, Matrix.sum_apply]

theorem locally_compressed_stack_preserves_actual_cross_gram
    {ι p c : Type*} [Fintype ι] (size qsize kept : ι → ℕ)
    (Q : (record : ι) → Matrix (Fin (size record)) (Fin (qsize record)) ℝ)
    (B : (record : ι) → Matrix (Fin (size record)) (Fin (kept record)) ℝ)
    (complete : ∀ record, B record * (B record)ᵀ = projection (Q record))
    (D : (record : ι) → Matrix (Fin (size record)) p ℝ)
    (Y : (record : ι) → Matrix (Fin (size record)) c ℝ)
    (retained : ∀ record, projection (Q record) * Y record = Y record) :
    (stack (fun record => (B record)ᵀ * D record))ᵀ * stack (fun record => (B record)ᵀ * Y record) =
      (stack D)ᵀ * stack Y := by
  rw [stack_cross_gram_is_the_sum_of_independent_record_cross_grams,
    stack_cross_gram_is_the_sum_of_independent_record_cross_grams]
  apply Finset.sum_congr rfl
  intro record _
  exact frame_cross_product_preserved (Q record) (B record) (complete record) (D record) (Y record) (retained record)

theorem genuine_local_bases_exist_for_the_actual_reset_records
    {ι : Type*} (size qsize : ι → ℕ)
    (Q : (record : ι) → Matrix (Fin (size record)) (Fin (qsize record)) ℝ)
    (orthogonal : ∀ record, (Q record)ᵀ * Q record = 1) :
    ∃ B : (record : ι) → Matrix (Fin (size record)) (Fin (size record - qsize record)) ℝ,
      (∀ record, (B record)ᵀ * B record = 1) ∧
      (∀ record, B record * (B record)ᵀ = projection (Q record)) := by
  have localBases : ∀ record, ∃ B : Matrix (Fin (size record)) (Fin (size record - qsize record)) ℝ,
      Bᵀ * B = 1 ∧ B * Bᵀ = projection (Q record) := by
    intro record
    have full := N2P3.RetainedBasis.exists_retained_basis (Q record) (orthogonal record)
    rw [Fintype.card_fin, Fintype.card_fin] at full
    obtain ⟨B, first, _, last⟩ := full
    exact ⟨B, first, last⟩
  choose B first last using localBases
  exact ⟨B, first, last⟩

theorem compressed_record_stack_has_actual_source_mean_and_covariance
    {ι p c : Type*} [Fintype ι] [DecidableEq ι] [Fintype p] [DecidableEq p]
    (size qsize kept : ι → ℕ)
    (Q : (record : ι) → Matrix (Fin (size record)) (Fin (qsize record)) ℝ)
    (B : (record : ι) → Matrix (Fin (size record)) (Fin (kept record)) ℝ)
    (complete : ∀ record, B record * (B record)ᵀ = projection (Q record))
    (D : (record : ι) → Matrix (Fin (size record)) p ℝ)
    (Y : (record : ι) → Matrix (Fin (size record)) c ℝ)
    (designRetained : ∀ record, projection (Q record) * D record = D record)
    (sampleRetained : ∀ record, projection (Q record) * Y record = Y record) (lambda : ℝ) :
    rowCov (stack (fun record => (B record)ᵀ * D record)) 1 lambda = sourceRowCov (stack D) lambda ∧
    mean (stack (fun record => (B record)ᵀ * D record))
      (stack (fun record => (B record)ᵀ * Y record)) 1 lambda = sourceMean (stack D) (stack Y) lambda := by
  have gram := locally_compressed_stack_preserves_actual_cross_gram size qsize kept Q B complete D D designRetained
  have cross := locally_compressed_stack_preserves_actual_cross_gram size qsize kept Q B complete D Y sampleRetained
  have row : sourceRowCov (stack (fun record => (B record)ᵀ * D record)) lambda = sourceRowCov (stack D) lambda := by
    simp only [sourceRowCov, sourcePrecision, gram]
  constructor
  · simpa only [← source_rowCov_is_declared_rowCov] using row
  · simp only [← source_mean_is_declared_mean, sourceMean, row]
    simpa only [Matrix.mul_assoc] using
      congrArg (fun G => sourceRowCov (stack D) lambda * G) cross

abbrev nuisanceCount (noise : Noise) (n : ℕ) := min (rows noise n) 2

/-- These are the independent finite source arrays and the individual nuisance
QR observations. Filter/no-filter flags are shared by the actual model. -/
structure RecordInput (noise : Noise) (n : ℕ) (p c : Type*) where
  rawDesign : Matrix (Fin n) p ℝ
  rawSamples : Matrix (Fin n) c ℝ
  filter : Matrix (Fin n) (Fin n) ℝ
  Q : Matrix (Fin (rows noise n)) (Fin (nuisanceCount noise n)) ℝ
  nuisanceFactor : Matrix (Fin (nuisanceCount noise n)) (Fin 2) ℝ

def RecordInput.design {noise : Noise} {n : ℕ} {p c : Type*} (input : RecordInput noise n p c)
    (hasFilter filterDesign : Bool) := designProject noise input.Q hasFilter filterDesign input.filter input.rawDesign

def RecordInput.samples {noise : Noise} {n : ℕ} {p c : Type*} (input : RecordInput noise n p c)
    (hasFilter : Bool) := sampleProject noise input.Q hasFilter input.filter input.rawSamples

def RecordInput.Orthogonal {noise : Noise} {n : ℕ} {p c : Type*} (input : RecordInput noise n p c) : Prop :=
  input.Qᵀ * input.Q = 1

def RecordInput.NuisanceQR {noise : Noise} {n : ℕ} {p c : Type*} (input : RecordInput noise n p c) : Prop :=
  white noise n (originalNuisance n) = input.Q * input.nuisanceFactor

theorem actual_record_design_and_samples_are_retained {noise : Noise} {n : ℕ} {p c : Type*}
    (input : RecordInput noise n p c) (orthogonal : input.Orthogonal) (hasFilter filterDesign : Bool) :
    projection input.Q * input.design hasFilter filterDesign = input.design hasFilter filterDesign ∧
    projection input.Q * input.samples hasFilter = input.samples hasFilter := by
  exact ⟨actual_projected_values_are_retained noise n input.Q orthogonal _,
    actual_projected_values_are_retained noise n input.Q orthogonal _⟩

def producedRowCov {p : Type*} [Fintype p] [DecidableEq p]
    (factor : Matrix p p ℝ) : Matrix p p ℝ :=
  (1/2 : ℝ) • (factor⁻¹ * (factor⁻¹)ᵀ + (factor⁻¹ * (factor⁻¹)ᵀ)ᵀ)

def producedMean {n p c : Type*} [Fintype n] [Fintype p] [DecidableEq p]
    (factor : Matrix p p ℝ) (D : Matrix n p ℝ) (Y : Matrix n c ℝ) : Matrix p c ℝ :=
  (factor⁻¹ * (factor⁻¹)ᵀ) * (Dᵀ * Y)

theorem actual_augmented_qr_recovers_source_row_covariance
    {n p : Type*} [Fintype n] [Fintype p] [DecidableEq n] [DecidableEq p]
    (D : Matrix n p ℝ) (lambda : ℝ) (positive : 0 < lambda)
    (Q : Matrix (n ⊕ p) p ℝ) (factor : Matrix p p ℝ)
    (qr : augmentedDesign D lambda = Q * factor) (orthogonal : Qᵀ * Q = 1) :
    producedRowCov factor = sourceRowCov D lambda := by
  have raw := augmentedQR_rowCov (1 : Matrix n n ℝ) 1 D lambda Q factor (by simp) positive.le
    (by simpa using qr) orthogonal
  have row : factor⁻¹ * (factor⁻¹)ᵀ = sourceRowCov D lambda := by
    simpa only [source_rowCov_is_declared_rowCov] using raw
  rw [producedRowCov, row]
  exact exact_source_rowCov_symmetrization D lambda positive

theorem actual_augmented_qr_recovers_source_mean
    {n p c : Type*} [Fintype n] [Fintype p] [DecidableEq n] [DecidableEq p]
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (lambda : ℝ) (nonnegative : 0 ≤ lambda)
    (Q : Matrix (n ⊕ p) p ℝ) (factor : Matrix p p ℝ)
    (qr : augmentedDesign D lambda = Q * factor) (orthogonal : Qᵀ * Q = 1) :
    producedMean factor D Y = sourceMean D Y lambda := by
  have raw := augmentedQR_mean (1 : Matrix n n ℝ) 1 D Y lambda Q factor (by simp) nonnegative
    (by simpa using qr) orthogonal
  simpa only [producedMean, source_mean_is_declared_mean, inv_one, Matrix.one_mul, Matrix.mul_assoc] using raw

theorem inverse_factor_solver_equations_recover_actual_factor_inverse
    {n p : Type*} [Fintype n] [Fintype p] [DecidableEq n] [DecidableEq p]
    (D : Matrix n p ℝ) (lambda : ℝ) (positive : 0 < lambda)
    (Q : Matrix (n ⊕ p) p ℝ) (factor inverse : Matrix p p ℝ)
    (qr : augmentedDesign D lambda = Q * factor) (orthogonal : Qᵀ * Q = 1)
    (solve : factor * inverse = 1) : inverse = factor⁻¹ := by
  have unit := augmentedQR_unit (1 : Matrix n n ℝ) 1 D lambda Q factor (by simp)
    (by simpa using qr) orthogonal Matrix.PosDef.one positive
  have equality := congrArg (fun A : Matrix p p ℝ => factor⁻¹ * A) solve
  simpa only [← Matrix.mul_assoc, Matrix.nonsing_inv_mul factor unit, Matrix.one_mul, Matrix.mul_one] using equality

theorem actual_source_inverse_array_fields_are_the_declared_fit
    {n p c : Type*} [Fintype n] [Fintype p] [DecidableEq n] [DecidableEq p]
    (D : Matrix n p ℝ) (Y : Matrix n c ℝ) (lambda : ℝ) (positive : 0 < lambda)
    (Q : Matrix (n ⊕ p) p ℝ) (factor inverse : Matrix p p ℝ)
    (qr : augmentedDesign D lambda = Q * factor) (orthogonal : Qᵀ * Q = 1)
    (solve : factor * inverse = 1) :
    (1/2 : ℝ) • (inverse * inverseᵀ + (inverse * inverseᵀ)ᵀ) = sourceRowCov D lambda ∧
      (inverse * inverseᵀ) * (Dᵀ * Y) = sourceMean D Y lambda := by
  rw [inverse_factor_solver_equations_recover_actual_factor_inverse D lambda positive Q factor inverse qr orthogonal solve]
  exact ⟨actual_augmented_qr_recovers_source_row_covariance D lambda positive Q factor qr orthogonal,
    actual_augmented_qr_recovers_source_mean D Y lambda positive.le Q factor qr orthogonal⟩

def designs {ι p c : Type*} {noise : Noise} {sizes : ι → ℕ}
    (inputs : (record : ι) → RecordInput noise (sizes record) p c) (hasFilter filterDesign : Bool) :=
  fun record => (inputs record).design hasFilter filterDesign

def samples {ι p c : Type*} {noise : Noise} {sizes : ι → ℕ}
    (inputs : (record : ι) → RecordInput noise (sizes record) p c) (hasFilter : Bool) :=
  fun record => (inputs record).samples hasFilter

/-- This is a probability composition for every actual startup/prior-scale
branch. Data and design are individually processed/whitened/projected before
their stack, and the exact source QR/inverse producer supplies the actual
center and rowCov. Calibration and held noise are the declared independent
coordinate model; real raw-prefix independence is not inferred. -/
theorem actual_all_branch_caller_predictive_integral
    {ι p c : Type*} [Fintype ι] [DecidableEq ι] [Fintype p] [DecidableEq p]
    [LinearOrder p] [Fintype c] [DecidableEq c]
    (noise : Noise) (sizes : ι → ℕ) (heldSize : ℕ)
    (inputs : (record : ι) → RecordInput noise (sizes record) p c)
    (held : RecordInput noise heldSize p c) (hasFilter filterDesign : Bool)
    (inputOrthogonal : ∀ record, (inputs record).Orthogonal)
    (inputNuisanceQR : ∀ record, (inputs record).NuisanceQR)
    (heldOrthogonal : held.Orthogonal) (heldNuisanceQR : held.NuisanceQR)
    (scale : PriorScale) (ridge : ℝ) (ridgePositive : 0 < ridge)
    (stationary : noise.mode = .stationary → noise.coefficients ≠ [] → 0 < noise.startupCov00)
    (fitQ : Matrix (((record : ι) × Fin (rows noise (sizes record))) ⊕ p) p ℝ)
    (fitFactor fitInverse C : Matrix p p ℝ)
    (fitQR : augmentedDesign (stack (designs inputs hasFilter filterDesign)) (effectiveLambda scale noise ridge) =
      fitQ * fitFactor) (fitOrthogonal : fitQᵀ * fitQ = 1) (inverseSolve : fitFactor * fitInverse = 1)
    (kernelFactor : C * Cᵀ = (1/2 : ℝ) • (fitInverse * fitInverseᵀ + (fitInverse * fitInverseᵀ)ᵀ))
    (L Sigma : Matrix c c ℝ) (spatialFactor : L * Lᵀ = Sigma) (spatialPositive : Sigma.PosDef)
    (T : Matrix p p ℝ)
    (smallFactor : T * Tᵀ = GaussianLowRank.smallCovariance (held.design hasFilter filterDesign * C))
    (smallLower : T.BlockTriangular OrderDual.toDual) (smallDiagonal : ∀ i, 0 < T i i) :
    ∃ Bc : (record : ι) → Matrix (Fin (rows noise (sizes record)))
        (Fin (rows noise (sizes record) - nuisanceCount noise (sizes record))) ℝ,
      (∀ record, (Bc record)ᵀ * Bc record = 1 ∧
        Bc record * (Bc record)ᵀ = projection (inputs record).Q) ∧
    ∃ Bh : Matrix (Fin (rows noise heldSize)) (Fin (rows noise heldSize - nuisanceCount noise heldSize)) ℝ,
      Bhᵀ * Bh = 1 ∧ Bh * Bhᵀ = projection held.Q ∧
      (∀ record, project noise (sizes record) (inputs record).Q (originalNuisance (sizes record)) = 0) ∧
      project noise heldSize held.Q (originalNuisance heldSize) = 0 ∧
      IsMarkovKernel (GaussianJointLaw.posteriorKernel
        (stack (fun record => (Bc record)ᵀ * designs inputs hasFilter filterDesign record))
        1 (effectiveLambda scale noise ridge) C⁻¹ L) ∧
      Real.log (∫ x : (p × c) → ℝ,
        BayesDensity.likelihoodDensity (Bhᵀ * held.design hasFilter filterDesign) (MatrixNormalCoordinates.matricize x)
          1 L (MatrixNormalCoordinates.vectorize (Bhᵀ * held.samples hasFilter))
        ∂GaussianJointLaw.posteriorKernel
          (stack (fun record => (Bc record)ᵀ * designs inputs hasFilter filterDesign record))
          1 (effectiveLambda scale noise ridge) C⁻¹ L
          (MatrixNormalCoordinates.vectorize (stack (fun record => (Bc record)ᵀ * samples inputs hasFilter record)))) =
        GaussianLowRank.sourceCandidateScore (held.design hasFilter filterDesign * C) T
          (spatialWhiten L (held.samples hasFilter - held.design hasFilter filterDesign *
            ((fitInverse * fitInverseᵀ) *
              ((stack (designs inputs hasFilter filterDesign))ᵀ * stack (samples inputs hasFilter))))) +
          gaussianLogConstant (rows noise heldSize - nuisanceCount noise heldSize) L := by
  have positive := all_startup_and_scale_branches_have_positive_actual_lambda scale noise ridge ridgePositive stationary
  obtain ⟨Bc, basisOrthogonal, basisComplete⟩ := genuine_local_bases_exist_for_the_actual_reset_records
    (fun record => rows noise (sizes record)) (fun record => nuisanceCount noise (sizes record))
    (fun record => (inputs record).Q) inputOrthogonal
  have retainedD : ∀ record, projection (inputs record).Q * designs inputs hasFilter filterDesign record =
      designs inputs hasFilter filterDesign record := fun record =>
    (actual_record_design_and_samples_are_retained (inputs record) (inputOrthogonal record) hasFilter filterDesign).1
  have retainedY : ∀ record, projection (inputs record).Q * samples inputs hasFilter record =
      samples inputs hasFilter record := fun record =>
    (actual_record_design_and_samples_are_retained (inputs record) (inputOrthogonal record) hasFilter filterDesign).2
  have compressed := compressed_record_stack_has_actual_source_mean_and_covariance
    (fun record => rows noise (sizes record)) (fun record => nuisanceCount noise (sizes record))
    (fun record => rows noise (sizes record) - nuisanceCount noise (sizes record))
    (fun record => (inputs record).Q) Bc basisComplete (designs inputs hasFilter filterDesign)
    (samples inputs hasFilter) retainedD retainedY (effectiveLambda scale noise ridge)
  have actualFields := actual_source_inverse_array_fields_are_the_declared_fit
    (stack (designs inputs hasFilter filterDesign)) (stack (samples inputs hasFilter))
    (effectiveLambda scale noise ridge) positive fitQ fitFactor fitInverse fitQR fitOrthogonal inverseSolve
  have compressedFactor : C * Cᵀ = rowCov
      (stack (fun record => (Bc record)ᵀ * designs inputs hasFilter filterDesign record))
      1 (effectiveLambda scale noise ridge) := kernelFactor.trans (actualFields.1.trans compressed.1.symm)
  have compressedMean : mean
      (stack (fun record => (Bc record)ᵀ * designs inputs hasFilter filterDesign record))
      (stack (fun record => (Bc record)ᵀ * samples inputs hasFilter record)) 1 (effectiveLambda scale noise ridge) =
      (fitInverse * fitInverseᵀ) * ((stack (designs inputs hasFilter filterDesign))ᵀ * stack (samples inputs hasFilter)) :=
    compressed.2.trans actualFields.2.symm
  have heldBasis := N2P3.RetainedBasis.exists_retained_basis held.Q heldOrthogonal
  rw [Fintype.card_fin, Fintype.card_fin] at heldBasis
  obtain ⟨Bh, heldBOrthogonal, _, heldBComplete⟩ := heldBasis
  have heldRetained := actual_record_design_and_samples_are_retained held heldOrthogonal hasFilter filterDesign
  have probability := ContinuousPosteriorModel.fitted_posterior_predictive_log_score
    (stack (fun record => (Bc record)ᵀ * designs inputs hasFilter filterDesign record))
    (stack (fun record => (Bc record)ᵀ * samples inputs hasFilter record))
    1 (effectiveLambda scale noise ridge) C compressedFactor Matrix.PosDef.one positive
    held.Q Bh heldBOrthogonal heldBComplete (held.design hasFilter filterDesign) heldRetained.1
    L Sigma spatialFactor spatialPositive T smallFactor smallLower smallDiagonal (held.samples hasFilter) heldRetained.2
  rw [compressedMean] at probability
  refine ⟨Bc, fun record => ⟨basisOrthogonal record, basisComplete record⟩,
    Bh, heldBOrthogonal, heldBComplete, ?_, ?_, probability.1, ?_⟩
  · intro record
    exact actual_original_nuisance_is_removed noise (sizes record) (inputs record).Q
      (inputs record).nuisanceFactor (inputOrthogonal record) (inputNuisanceQR record)
  · exact actual_original_nuisance_is_removed noise heldSize held.Q held.nuisanceFactor heldOrthogonal heldNuisanceQR
  · simpa only [Fintype.card_fin] using probability.2

end
end N2P3.OperatorGLSCaller
