import N2P3.SelectionERPSupport
import N2P3.SelectionERPEvidence
import N2P3.PersonalLDA

/-! SelectionERP family forwards and their candidate-level composition.
The waveform and Toeplitz affine graphs are explicit. xDAWN constructs its
filtered/prototype augmented traces and literal OAS covariance before TWO
named native operations: Cholesky admission and frozen-reference Riemann
tangent coordinates. Neither operation answers decoder/fit/scorer success.
Native nonfinite scalar/cast/FFT inputs lie outside the finite-real operation
domain; no IEEE or full owner-fitting theorem is asserted. -/
namespace N2P3.SelectionERPCaller
open Matrix
open scoped BigOperators Matrix ComplexOrder
open N2P3.SelectionERPSupport
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

inductive PredictionDtype where | float32 | float64 deriving DecidableEq
inductive NumericFailure where | covariance | tangent deriving DecidableEq
abbrev NumericCheck := Except NumericFailure

structure WaveformFields (channels : ℕ) where
  mean : Fin (channels*28) → ℝ
  scale : Fin (channels*28) → ℝ
  coefficient : Fin (channels*28) → ℝ
  intercept : ℝ
  predictionDtype : PredictionDtype

structure XDawnFields (channels : ℕ) where
  filters : Matrix (Fin 8) (Fin channels) ℝ
  prototypes : Matrix (Fin 8) (Fin 178) ℝ
  reference : Matrix (Fin 16) (Fin 16) ℝ
  coefficient : Fin 136 → ℝ
  intercept : ℝ

structure NumericalOperations where
  fft20Hz : PersonalLDA.TraceMap 178 28
  cast32 : ℝ → ℝ
  choleskyAdmission : Matrix (Fin 16) (Fin 16) ℝ → NumericCheck Unit
  riemannTangent : Matrix (Fin 16) (Fin 16) ℝ → Matrix (Fin 16) (Fin 16) ℝ →
    NumericCheck (Fin 136 → ℝ)

def coefficientCast (ops : NumericalOperations) (dtype : PredictionDtype) (value : ℝ) : ℝ :=
  match dtype with | .float32 => ops.cast32 value | .float64 => value

def waveformFeature {c : ℕ} (ops : NumericalOperations) (epoch : ERPFeatures.Epoch c 178)
    (coordinate : Fin (c*28)) : ℝ :=
  let pair := finProdFinEquiv.symm coordinate
  ops.cast32 (ops.fft20Hz (epoch pair.1) pair.2)

def standardizedWaveformFeature {c : ℕ} (ops : NumericalOperations) (state : WaveformFields c)
    (epoch : ERPFeatures.Epoch c 178) (coordinate : Fin (c*28)) : ℝ :=
  let feature := waveformFeature ops epoch coordinate
  let centered := ops.cast32 (feature-ops.cast32 (state.mean coordinate))
  ops.cast32 (centered/ops.cast32 (state.scale coordinate))

def waveformLogit {c : ℕ} (ops : NumericalOperations) (state : WaveformFields c)
    (epoch : ERPFeatures.Epoch c 178) : ℝ :=
  (∑ j,standardizedWaveformFeature ops state epoch j*
    coefficientCast ops state.predictionDtype (state.coefficient j))+
      coefficientCast ops state.predictionDtype state.intercept

theorem waveform_feature_uses_original_physical_channel_major_order {c : ℕ}
    (ops : NumericalOperations) (epoch : ERPFeatures.Epoch c 178) (channel : Fin c) (time : Fin 28) :
    waveformFeature ops epoch (finProdFinEquiv (channel,time))=ops.cast32 (ops.fft20Hz (epoch channel) time) := by
  simp [waveformFeature]

theorem waveform_inplace_scaler_casts_are_not_omitted {c : ℕ} (ops : NumericalOperations)
    (state : WaveformFields c) (epoch : ERPFeatures.Epoch c 178) (coordinate : Fin (c*28)) :
    standardizedWaveformFeature ops state epoch coordinate=
      ops.cast32 (ops.cast32 (waveformFeature ops epoch coordinate-ops.cast32 (state.mean coordinate))/
        ops.cast32 (state.scale coordinate)) := rfl

theorem waveform_affine_field_formula_uses_the_stored_prediction_dtype {c : ℕ}
    (ops : NumericalOperations) (state : WaveformFields c) (epoch : ERPFeatures.Epoch c 178) :
    waveformLogit ops state epoch=
      (∑ j,standardizedWaveformFeature ops state epoch j*
        coefficientCast ops state.predictionDtype (state.coefficient j))+
        coefficientCast ops state.predictionDtype state.intercept := rfl

def toeplitzLogit {c : ℕ} (state : ERPFeatures.Affine (c*44)) (epoch : ERPFeatures.Epoch c 178) : ℝ :=
  ERPFeatures.evaluate state (ERPFeatures.binFeatures 4 epoch)

theorem toeplitz_forward_reuses_the_actual_channel_prime_four_sample_bins {c : ℕ}
    (state : ERPFeatures.Affine (c*44)) (epoch : ERPFeatures.Epoch c 178) :
    toeplitzLogit state epoch=
      (∑ time : Fin 44,∑ channel : Fin c,
        ((∑ offset : Fin 4,epoch channel (ERPFeatures.binCoordinate 4 time offset))/4)*
          state.coefficient (ERPFeatures.sourceColumn (time,channel)))+state.intercept :=
  by simpa only [toeplitzLogit,Nat.cast_ofNat] using
    (ERPFeatures.binned_affine_formula (C := c) (T := 178) 4 state epoch)

def xdawnFiltered {c : ℕ} (state : XDawnFields c) (epoch : ERPFeatures.Epoch c 178) :
    Matrix (Fin 8) (Fin 178) ℝ := state.filters*(show Matrix (Fin c) (Fin 178) ℝ from epoch)

def virtualRow : Fin 8 ⊕ Fin 8 ≃ Fin 16 := finSumFinEquiv

def xdawnAugmented {c : ℕ} (state : XDawnFields c) (epoch : ERPFeatures.Epoch c 178) :
    Matrix (Fin 16) (Fin 178) ℝ :=
  fun row time => match virtualRow.symm row with
    | .inl prototype => state.prototypes prototype time
    | .inr observed => xdawnFiltered state epoch observed time

theorem xdawn_augmentation_keeps_frozen_prototypes_before_filtered_held_epoch {c : ℕ}
    (state : XDawnFields c) (epoch : ERPFeatures.Epoch c 178) (row : Fin 8) (time : Fin 178) :
    xdawnAugmented state epoch (virtualRow (.inl row)) time=state.prototypes row time ∧
      xdawnAugmented state epoch (virtualRow (.inr row)) time=
        ∑ channel,state.filters row channel*epoch channel time := by
  simp [xdawnAugmented,xdawnFiltered,Matrix.mul_apply]

def timeCentered (augmented : Matrix (Fin 16) (Fin 178) ℝ) : Matrix (Fin 16) (Fin 178) ℝ :=
  fun row time => augmented row time-(∑ t,augmented row t)/178

def empiricalCovariance (augmented : Matrix (Fin 16) (Fin 178) ℝ) : Matrix (Fin 16) (Fin 16) ℝ :=
  (1/178:ℝ) • (timeCentered augmented*(timeCentered augmented)ᵀ)

def oasAlpha (covariance : Matrix (Fin 16) (Fin 16) ℝ) : ℝ :=
  (∑ i,∑ j,(covariance i j)^2)/(16*16)

def oasMu (covariance : Matrix (Fin 16) (Fin 16) ℝ) : ℝ := (∑ i,covariance i i)/16

def oasDenominator (covariance : Matrix (Fin 16) (Fin 16) ℝ) : ℝ :=
  179*(oasAlpha covariance-(oasMu covariance)^2/16)

def oasRho (covariance : Matrix (Fin 16) (Fin 16) ℝ) : ℝ :=
  if oasDenominator covariance=0 then 1 else
    min ((oasAlpha covariance+(oasMu covariance)^2)/oasDenominator covariance) 1

def oasCovariance (augmented : Matrix (Fin 16) (Fin 178) ℝ) : Matrix (Fin 16) (Fin 16) ℝ :=
  let covariance := empiricalCovariance augmented
  (1-oasRho covariance) • covariance+(oasRho covariance*oasMu covariance) • 1

theorem actual_oas_zero_denominator_uses_full_spherical_shrinkage
    (covariance : Matrix (Fin 16) (Fin 16) ℝ) (zero : oasDenominator covariance=0) :
    oasRho covariance=1 := by simp [oasRho,zero]

theorem actual_empirical_covariance_is_a_positive_semidefinite_gram
    (augmented : Matrix (Fin 16) (Fin 178) ℝ) : (empiricalCovariance augmented).PosSemidef := by
  have gram : (timeCentered augmented*(timeCentered augmented)ᵀ).PosSemidef := by
    simpa using (Matrix.PosSemidef.one : (1 : Matrix (Fin 178) (Fin 178) ℝ).PosSemidef).mul_mul_conjTranspose_same (timeCentered augmented)
  exact gram.smul (by norm_num : (0:ℝ) ≤ 1/178)

def xdawnLogit {c : ℕ} (ops : NumericalOperations) (state : XDawnFields c)
    (epoch : ERPFeatures.Epoch c 178) : NumericCheck ℝ := do
  let covariance := oasCovariance (xdawnAugmented state epoch)
  ops.choleskyAdmission covariance
  let tangent ← ops.riemannTangent state.reference covariance
  pure ((∑ j,tangent j*state.coefficient j)+state.intercept)

theorem xdawn_actual_cholesky_failure_precedes_tangent_and_affine_layer {c : ℕ}
    (ops : NumericalOperations) (state : XDawnFields c) (epoch : ERPFeatures.Epoch c 178)
    (failure : ops.choleskyAdmission (oasCovariance (xdawnAugmented state epoch))=.error .covariance) :
    xdawnLogit ops state epoch=.error .covariance := by simp [xdawnLogit,failure]

theorem xdawn_logit_from_individual_covariance_and_frozen_tangent_operations {c : ℕ}
    (ops : NumericalOperations) (state : XDawnFields c) (epoch : ERPFeatures.Epoch c 178)
    (tangent : Fin 136 → ℝ)
    (factor : ops.choleskyAdmission (oasCovariance (xdawnAugmented state epoch))=.ok ())
    (coordinates : ops.riemannTangent state.reference (oasCovariance (xdawnAugmented state epoch))=.ok tangent) :
    xdawnLogit ops state epoch=.ok ((∑ j,tangent j*state.coefficient j)+state.intercept) := by
  simp [xdawnLogit,factor,coordinates]

def xdawn4FieldsAllowed (channels observedRank effectiveFilters : ℕ) : Prop :=
  4 ≤ observedRank ∧ observedRank ≤ channels ∧ effectiveFilters=4

theorem xdawn4_fields_force_four_actual_observed_spatial_dimensions
    (channels observedRank effectiveFilters : ℕ) (fields : xdawn4FieldsAllowed channels observedRank effectiveFilters) :
    4 ≤ channels ∧ effectiveFilters=4 := ⟨le_trans fields.1 fields.2.1,fields.2.2⟩

inductive FamilyFields (channels : ℕ) where
  | waveform (state : WaveformFields channels)
  | toeplitz (state : ERPFeatures.Affine (channels*44))
  | xdawn (state : XDawnFields channels)

def family {c : ℕ} : FamilyFields c → Family
  | .waveform _ => .waveform
  | .toeplitz _ => .toeplitz
  | .xdawn _ => .xdawn

def familyLogit {c : ℕ} (ops : NumericalOperations) : FamilyFields c → ERPFeatures.Epoch c 178 → NumericCheck ℝ
  | .waveform state,epoch => .ok (waveformLogit ops state epoch)
  | .toeplitz state,epoch => .ok (toeplitzLogit state epoch)
  | .xdawn state,epoch => xdawnLogit ops state epoch

/-- A literal sequence of individual operations. Source batch stages invoke
this separately for all covariance admissions and then all tangent outputs. -/
def collectRows {A : Type} : (rows : ℕ) → (Fin rows → NumericCheck A) → NumericCheck (Fin rows → A)
  | 0,_ => .ok (fun row => Fin.elim0 row)
  | rows+1,step => do
    let first ← step 0
    let remaining ← collectRows rows (fun row => step row.succ)
    pure (Fin.cases first remaining)

theorem collected_rows_are_derived_from_individual_operation_results {A : Type} (rows : ℕ)
    (step : Fin rows → NumericCheck A) (value : Fin rows → A)
    (individual : ∀ row,step row=.ok (value row)) : collectRows rows step=.ok value := by
  induction rows with
  | zero =>
    have same : (fun row : Fin 0 => Fin.elim0 row)=value := funext (fun row => Fin.elim0 row)
    simpa only [collectRows] using congrArg
      (fun f : Fin 0 → A => (Except.ok f : NumericCheck (Fin 0 → A))) same
  | succ rows ih =>
    have tail := ih (fun row => step row.succ) (fun row => value row.succ) (fun row => individual row.succ)
    have same : Fin.cases (value 0) (fun row => value row.succ)=value := by
      funext row
      exact Fin.cases rfl (fun _ => rfl) row
    simp [collectRows,individual 0,tail,same]

/-- Source structured_erp.predict_logit computes the covariance batch, admits
ALL covariances with its batched Cholesky, and only then calls tangent.transform.
Interleaving Cholesky and tangent per trial would change its failure priority. -/
def xdawnBatchLogits {rows c : ℕ} (ops : NumericalOperations) (state : XDawnFields c)
    (epochs : Fin rows → ERPFeatures.Epoch c 178) : NumericCheck (Fin rows → ℝ) := do
  let covariance := fun row => oasCovariance (xdawnAugmented state (epochs row))
  let _ ← collectRows rows (fun row => ops.choleskyAdmission (covariance row))
  let tangent ← collectRows rows (fun row => ops.riemannTangent state.reference (covariance row))
  pure (fun row => (∑ j,tangent row j*state.coefficient j)+state.intercept)

theorem xdawn_batch_result_from_individual_covariance_and_tangent_operations {rows c : ℕ}
    (ops : NumericalOperations) (state : XDawnFields c) (epochs : Fin rows → ERPFeatures.Epoch c 178)
    (coordinates : Fin rows → Fin 136 → ℝ)
    (factor : ∀ row,ops.choleskyAdmission (oasCovariance (xdawnAugmented state (epochs row)))=.ok ())
    (tangent : ∀ row,ops.riemannTangent state.reference
      (oasCovariance (xdawnAugmented state (epochs row)))=.ok (coordinates row)) :
    xdawnBatchLogits ops state epochs=.ok (fun row => (∑ j,coordinates row j*state.coefficient j)+state.intercept) := by
  have factors := collected_rows_are_derived_from_individual_operation_results rows
    (fun row => ops.choleskyAdmission (oasCovariance (xdawnAugmented state (epochs row)))) (fun _ => ()) factor
  have tangents := collected_rows_are_derived_from_individual_operation_results rows
    (fun row => ops.riemannTangent state.reference (oasCovariance (xdawnAugmented state (epochs row)))) coordinates tangent
  simp [xdawnBatchLogits,factors,tangents]

theorem a_later_covariance_failure_precedes_every_earlier_row_tangent_call {c : ℕ}
    (ops : NumericalOperations) (state : XDawnFields c) (epochs : Fin 2 → ERPFeatures.Epoch c 178)
    (first : ops.choleskyAdmission (oasCovariance (xdawnAugmented state (epochs 0)))=.ok ())
    (later : ops.choleskyAdmission (oasCovariance (xdawnAugmented state (epochs 1)))=.error .covariance) :
    xdawnBatchLogits ops state epochs=.error .covariance := by
  simp [xdawnBatchLogits,collectRows,first,later]

def familyBatchLogits {rows c : ℕ} (ops : NumericalOperations) :
    FamilyFields c → (Fin rows → ERPFeatures.Epoch c 178) → NumericCheck (Fin rows → ℝ)
  | .waveform state,epochs => .ok (fun row => waveformLogit ops state (epochs row))
  | .toeplitz state,epochs => .ok (fun row => toeplitzLogit state (epochs row))
  | .xdawn state,epochs => xdawnBatchLogits ops state epochs

inductive ScoreFailure where
  | declaration | membership | frozenMetadata | recordIdentity | calibrationReplay | numeric (failure : NumericFailure)
  deriving DecidableEq

/-- The source tests identity with Python False, not generic falsiness. This
small atom classification is independent of all record/decoder success. -/
inductive CensorFlag where | pythonFalse | pythonTrue | numericZero | numpyFalse | other
  deriving DecidableEq

def completeCalibrationFlag (flag : CensorFlag) : Bool :=
  match flag with | .pythonFalse => true | _ => false

theorem selection_anchor_calibration_accepts_only_literal_uncensored_false :
    completeCalibrationFlag .pythonFalse=true ∧ completeCalibrationFlag .pythonTrue=false ∧
      completeCalibrationFlag .numericZero=false ∧ completeCalibrationFlag .numpyFalse=false := by
  exact ⟨rfl,rfl,rfl,rfl⟩

def labelBudgetAdmission (expected actual nonTargets targets : ℕ) : Prop :=
  4 ≤ expected ∧ actual=expected ∧ nonTargets+targets=actual ∧ 2 ≤ nonTargets ∧ 2 ≤ targets

theorem independent_actual_counts_satisfy_the_e2_label_budget : labelBudgetAdmission 600 600 500 100 := by
  norm_num [labelBudgetAdmission]

def score {rows c : ℕ} (fft : TraceFFT) (ops : NumericalOperations) (state : FamilyFields c)
    (baseline : Baseline) (epochs : Fin rows → ERPFeatures.Epoch c 358)
    (membership : Fin rows → Fin 36 → ℤ) (priorLogOdds : ℝ) :
    Except ScoreFailure (SelectionERPEvidence.Readout (H := Fin 36)) :=
  if ¬familyBaselineAllowed (family state) baseline then .error .declaration
  else if CandidateCoverage.membershipReason rows 36 membership ≠ none then .error .membership
  else do
    let values ← (familyBatchLogits ops state (fun row => epochSeam fft baseline (epochs row))).mapError ScoreFailure.numeric
    pure (SelectionERPEvidence.readout (SelectionERPEvidence.evidence membership values priorLogOdds))

structure FrozenContext where
  sampleRate : ℝ
  channels : List String
  vocabulary : List String
  trainingRecordIds : List String
  trainingSignalDigests : List String

structure PredictionContext where
  sampleRate : ℝ
  channels : List String
  vocabulary : List String
  recordId : String
  signalDigest : String

def frozenMetadataMatches (frozen : FrozenContext) (provided : PredictionContext) : Prop :=
  provided.sampleRate=frozen.sampleRate ∧ provided.channels=frozen.channels ∧
    provided.vocabulary=frozen.vocabulary

/-- `nonblank` is one string-strip/nonempty operation. `signalDigest` is one
digest observation on supplied samples/onsets, with no injectivity assertion.
Integrity/array decoding precede this finite canonical context layer. -/
def scoreWithContext {rows c : ℕ} (nonblank : String → Bool)
    (frozen : FrozenContext) (provided : PredictionContext) (fft : TraceFFT) (ops : NumericalOperations)
    (state : FamilyFields c) (baseline : Baseline) (epochs : Fin rows → ERPFeatures.Epoch c 358)
    (membership : Fin rows → Fin 36 → ℤ) (prior : ℝ) :=
  if ¬frozenMetadataMatches frozen provided then Except.error ScoreFailure.frozenMetadata
  else if nonblank provided.recordId=false then Except.error ScoreFailure.recordIdentity
  else if provided.recordId ∈ frozen.trainingRecordIds ∨ provided.signalDigest ∈ frozen.trainingSignalDigests
    then Except.error ScoreFailure.calibrationReplay
  else score fft ops state baseline epochs membership prior

theorem changing_frozen_word_or_order_refuses_before_the_numeric_score {rows c : ℕ}
    (nonblank : String → Bool) (frozen : FrozenContext) (provided : PredictionContext)
    (fft : TraceFFT) (ops : NumericalOperations) (state : FamilyFields c) (baseline : Baseline)
    (epochs : Fin rows → ERPFeatures.Epoch c 358) (membership : Fin rows → Fin 36 → ℤ) (prior : ℝ)
    (different : provided.vocabulary ≠ frozen.vocabulary) :
    scoreWithContext nonblank frozen provided fft ops state baseline epochs membership prior=
      .error .frozenMetadata := by simp [scoreWithContext,frozenMetadataMatches,different]

theorem an_exact_training_signal_digest_is_refused_even_with_a_new_record_name {rows c : ℕ}
    (nonblank : String → Bool) (frozen : FrozenContext) (provided : PredictionContext)
    (fft : TraceFFT) (ops : NumericalOperations) (state : FamilyFields c) (baseline : Baseline)
    (epochs : Fin rows → ERPFeatures.Epoch c 358) (membership : Fin rows → Fin 36 → ℤ) (prior : ℝ)
    (matching : frozenMetadataMatches frozen provided) (name : nonblank provided.recordId=true)
    (reused : provided.signalDigest ∈ frozen.trainingSignalDigests) :
    scoreWithContext nonblank frozen provided fft ops state baseline epochs membership prior=
      .error .calibrationReplay := by simp [scoreWithContext,matching,name,reused]

theorem independent_metadata_and_record_checks_feed_the_actual_finite_score {rows c : ℕ}
    (nonblank : String → Bool) (frozen : FrozenContext) (provided : PredictionContext)
    (fft : TraceFFT) (ops : NumericalOperations) (state : FamilyFields c) (baseline : Baseline)
    (epochs : Fin rows → ERPFeatures.Epoch c 358) (membership : Fin rows → Fin 36 → ℤ) (prior : ℝ)
    (rate : provided.sampleRate=frozen.sampleRate) (channels : provided.channels=frozen.channels)
    (word : provided.vocabulary=frozen.vocabulary) (name : nonblank provided.recordId=true)
    (newId : provided.recordId ∉ frozen.trainingRecordIds) (newSignal : provided.signalDigest ∉ frozen.trainingSignalDigests) :
    scoreWithContext nonblank frozen provided fft ops state baseline epochs membership prior=
      score fft ops state baseline epochs membership prior := by
  simp [scoreWithContext,frozenMetadataMatches,rate,channels,word,name,newId,newSignal]

theorem forbidden_family_baseline_refuses_before_any_numeric_operation {rows c : ℕ}
    (fft : TraceFFT) (ops : NumericalOperations) (state : FamilyFields c) (baseline : Baseline)
    (epochs : Fin rows → ERPFeatures.Epoch c 358) (membership : Fin rows → Fin 36 → ℤ) (prior : ℝ)
    (forbidden : ¬familyBaselineAllowed (family state) baseline) :
    score fft ops state baseline epochs membership prior=.error .declaration := by simp [score,forbidden]

theorem actual_membership_refusal_precedes_the_epoch_and_family_numeric_calls {rows c : ℕ}
    (fft : TraceFFT) (ops : NumericalOperations) (state : FamilyFields c) (baseline : Baseline)
    (epochs : Fin rows → ERPFeatures.Epoch c 358) (membership : Fin rows → Fin 36 → ℤ) (prior : ℝ)
    (declaration : familyBaselineAllowed (family state) baseline)
    (refused : CandidateCoverage.membershipReason rows 36 membership ≠ none) :
    score fft ops state baseline epochs membership prior=.error .membership := by simp [score,declaration,refused]

theorem waveform_actual_candidate_score_is_derived_from_its_four_fields_and_cast_operations {rows c : ℕ}
    (fft : TraceFFT) (ops : NumericalOperations) (state : WaveformFields c) (baseline : Baseline)
    (epochs : Fin rows → ERPFeatures.Epoch c 358) (membership : Fin rows → Fin 36 → ℤ) (prior : ℝ)
    (admission : CandidateCoverage.membershipReason rows 36 membership=none) :
    score fft ops (.waveform state) baseline epochs membership prior=
      .ok (SelectionERPEvidence.readout (SelectionERPEvidence.evidence membership
        (fun row => waveformLogit ops state (epochSeam fft baseline (epochs row))) prior)) := by
  simp [score,family,familyBaselineAllowed,familyBatchLogits,admission]

theorem toeplitz_actual_candidate_score_is_derived_from_frozen_bin_affine_fields {rows c : ℕ}
    (fft : TraceFFT) (ops : NumericalOperations) (state : ERPFeatures.Affine (c*44))
    (epochs : Fin rows → ERPFeatures.Epoch c 358) (membership : Fin rows → Fin 36 → ℤ) (prior : ℝ)
    (admission : CandidateCoverage.membershipReason rows 36 membership=none) :
    score fft ops (.toeplitz state) .none epochs membership prior=
      .ok (SelectionERPEvidence.readout (SelectionERPEvidence.evidence membership
        (fun row => toeplitzLogit state (epochSeam fft .none (epochs row))) prior)) := by
  simp [score,family,familyBaselineAllowed,familyBatchLogits,admission]

theorem xdawn_actual_candidate_score_is_derived_from_individual_covariance_tangent_atoms {rows c : ℕ}
    (fft : TraceFFT) (ops : NumericalOperations) (state : XDawnFields c)
    (epochs : Fin rows → ERPFeatures.Epoch c 358) (membership : Fin rows → Fin 36 → ℤ) (prior : ℝ)
    (coordinates : Fin rows → Fin 136 → ℝ)
    (admission : CandidateCoverage.membershipReason rows 36 membership=none)
    (factor : ∀ row,ops.choleskyAdmission
      (oasCovariance (xdawnAugmented state (epochSeam fft .meanOnly (epochs row))))=.ok ())
    (tangent : ∀ row,ops.riemannTangent state.reference
      (oasCovariance (xdawnAugmented state (epochSeam fft .meanOnly (epochs row))))=.ok (coordinates row)) :
    score fft ops (.xdawn state) .meanOnly epochs membership prior=
      .ok (SelectionERPEvidence.readout (SelectionERPEvidence.evidence membership
        (fun row => (∑ j,coordinates row j*state.coefficient j)+state.intercept) prior)) := by
  have batch := xdawn_batch_result_from_individual_covariance_and_tangent_operations ops state
    (fun row => epochSeam fft .meanOnly (epochs row)) coordinates factor tangent
  simp [score,family,familyBaselineAllowed,familyBatchLogits,admission,batch]

theorem complete_physical_toeplitz_result_uses_actual36_ordered_llr_scores {rows c : ℕ}
    (fft : TraceFFT) (ops : NumericalOperations) (state : ERPFeatures.Affine (c*44))
    (epochs : Fin rows → ERPFeatures.Epoch c 358) (schedule : Fin rows → CandidateCoverage.Flash)
    (prior : ℝ) (complete : CandidateCoverage.PhysicalComplete schedule) :
    score fft ops (.toeplitz state) .none epochs (SelectionERPEvidence.indexedPhysicalMatrix schedule) prior=
      .ok (SelectionERPEvidence.readout (SelectionERPEvidence.physicalEvidence schedule
        (fun row => toeplitzLogit state (epochSeam fft .none (epochs row))) prior)) := by
  exact toeplitz_actual_candidate_score_is_derived_from_frozen_bin_affine_fields fft ops state epochs _ prior
    (by simpa only [Fintype.card_fin] using
      SelectionERPEvidence.indexed_complete_physical_matrix_passes_the_actual36_position_gate schedule complete)

structure PhysicalRecord (samples eventsMinusOne channels : ℕ) where
  geometry : InputGeometry samples eventsMinusOne
  raw : Matrix (Fin samples) (Fin channels) ℝ
  targetMask : Fin (eventsMinusOne+1) → Bool

def scorePhysicalRecord {n k c : ℕ} (fft : TraceFFT) (ops : NumericalOperations)
    (state : FamilyFields c) (baseline : Baseline) (filter : Matrix (Fin n) (Fin n) ℝ)
    (record : PhysicalRecord n k c) (membership : Fin (k+1) → Fin 36 → ℤ) (prior : ℝ) :=
  score fft ops state baseline (filteredEpochs record.geometry filter record.raw) membership prior

theorem actual_prediction_arguments_exclude_the_held_target_mask {n k c : ℕ}
    (fft : TraceFFT) (ops : NumericalOperations) (state : FamilyFields c) (baseline : Baseline)
    (filter : Matrix (Fin n) (Fin n) ℝ) (record : PhysicalRecord n k c)
    (membership : Fin (k+1) → Fin 36 → ℤ) (prior : ℝ) (changedTruth : Fin (k+1) → Bool) :
    scorePhysicalRecord fft ops state baseline filter {record with targetMask := changedTruth} membership prior=
      scorePhysicalRecord fft ops state baseline filter record membership prior := rfl

def trainingGroups (held : Fin 6) : Finset (Fin 6) := Finset.univ.erase held

theorem actual_e2_other_five_groups_exclude_the_held_record (held : Fin 6) :
    held ∉ trainingGroups held ∧ (trainingGroups held).card=5 := by
  simp [trainingGroups]

theorem actual_e2_complete_group_label_accounting (held : Fin 6) :
    (∑ _group ∈ trainingGroups held,(120:ℕ))=600 ∧
      (6*120:ℕ)=720 ∧ (800-720:ℕ)=80 := by
  have count := (actual_e2_other_five_groups_exclude_the_held_record held).2
  norm_num [count]

theorem actual_e2_complete_physical_training_target_count
    (schedule : Fin 6 → Fin 120 → CandidateCoverage.Flash)
    (complete : ∀ group,CandidateCoverage.PhysicalComplete (schedule group))
    (truth : Fin 6 → CandidateCoverage.Cell) (held : Fin 6) :
    (∑ group ∈ trainingGroups held,
      CandidateCoverage.columnCount (CandidateCoverage.physicalMatrix (schedule group)) (truth group))=100 := by
  have each : ∀ group,CandidateCoverage.columnCount
      (CandidateCoverage.physicalMatrix (schedule group)) (truth group)=(20:ℤ) :=
    fun group => CandidateCoverage.physical_complete_gives_twenty_per_cell _ (complete group) _
  simp_rw [each]
  rw [Finset.sum_const,nsmul_eq_mul,(actual_e2_other_five_groups_exclude_the_held_record held).2]
  norm_num

theorem actual_e2_fit_prior_ratio_uses100_targets_and500_nontargets :
    (600-100:ℕ)=500 ∧ (100/500:ℝ)=1/5 := by norm_num

def zeroToeplitz (channels : ℕ) (prior : ℝ) : ERPFeatures.Affine (channels*44) := ⟨fun _ => 0,prior⟩

theorem constructed_zero_contrast_toeplitz_has_literal_finite_binary_prior_logit
    (channels : ℕ) (prior : ℝ) (epoch : ERPFeatures.Epoch channels 178) :
    toeplitzLogit (zeroToeplitz channels prior) epoch=prior := by
  simp [toeplitzLogit,zeroToeplitz,ERPFeatures.evaluate]

theorem nonempty120_event36_choice_toeplitz_candidate_composition
    (fft : TraceFFT) (ops : NumericalOperations) (epochs : Fin 120 → ERPFeatures.Epoch 8 358)
    (prior : ℝ) :
    score fft ops (.toeplitz (zeroToeplitz 8 prior)) .none epochs
      (SelectionERPEvidence.indexedPhysicalMatrix SelectionERPEvidence.indexedCompleteSchedule) prior=
      .ok (SelectionERPEvidence.readout (fun _ => 0)) := by
  rw [toeplitz_actual_candidate_score_is_derived_from_frozen_bin_affine_fields fft ops (zeroToeplitz 8 prior)
    epochs _ prior SelectionERPEvidence.actual120_position_witness_passes_the_actual_fixed36_vocabulary_gate]
  simp only [constructed_zero_contrast_toeplitz_has_literal_finite_binary_prior_logit]
  have zero : SelectionERPEvidence.evidence
      (SelectionERPEvidence.indexedPhysicalMatrix SelectionERPEvidence.indexedCompleteSchedule)
      (fun _ => prior) prior=(fun _ => 0) := by
    funext candidate
    simp [SelectionERPEvidence.literal_llr_subtraction_then_membership_transpose_sum]
  rw [zero]

def constructedGeometry : InputGeometry 1429 119 where
  onset event := 51+9*event.val
  ordered := by
    intro a b less
    have ordered : a.val < b.val := less
    change 51+9*a.val < 51+9*b.val
    omega
  first := by norm_num
  last := by norm_num

theorem constructed_actual_selection_has120_complete_raw_windows :
    Fintype.card (Fin 120)=120 ∧
      ∀ event : Fin 120,(constructedGeometry.onset event+307)-(constructedGeometry.onset event-51)=358 := by
  exact ⟨rfl,raw_window_is_exactly_half_open_358 constructedGeometry⟩

abbrev PulseTimes := (Fin 16 × Fin 4) ⊕ Fin 114

def pulseTimePosition : PulseTimes ≃ Fin 178 :=
  (Equiv.sumCongr (finProdFinEquiv : Fin 16 × Fin 4 ≃ Fin 64) (Equiv.refl (Fin 114))).trans
    (finSumFinEquiv : Fin 64 ⊕ Fin 114 ≃ Fin 178)

def pulse (offset : Fin 4) : ℝ := ![8,-8,5,-5] offset

def pulseAugmented (row : Fin 16) (time : Fin 178) : ℝ :=
  match pulseTimePosition.symm time with
  | .inl coordinate => if coordinate.1=row then pulse coordinate.2 else 0
  | .inr _ => 0

theorem actual_four_pulse_pattern_has_zero_sum_and178_energy :
    (∑ offset : Fin 4,pulse offset)=0 ∧ (∑ offset : Fin 4,(pulse offset)^2)=178 := by
  norm_num [pulse,Fin.sum_univ_succ]

theorem constructed_augmented_rows_have_actual_zero_time_mean (row : Fin 16) :
    (∑ time,pulseAugmented row time)=0 := by
  rw [←Equiv.sum_comp pulseTimePosition]
  simp only [pulseAugmented,Equiv.symm_apply_apply]
  rw [Fintype.sum_sum_type,Fintype.sum_prod_type]
  simp only [Finset.sum_const_zero,add_zero]
  apply Finset.sum_eq_zero
  intro support _
  by_cases same : support=row
  · simp only [same,if_true]
    exact actual_four_pulse_pattern_has_zero_sum_and178_energy.1
  · simp [same]

theorem constructed_augmented_rows_have_actual_orthogonal178_cross_moments (a b : Fin 16) :
    (∑ time,pulseAugmented a time*pulseAugmented b time)=if a=b then 178 else 0 := by
  rw [←Equiv.sum_comp pulseTimePosition]
  simp only [pulseAugmented,Equiv.symm_apply_apply]
  rw [Fintype.sum_sum_type,Fintype.sum_prod_type]
  simp only [zero_mul,Finset.sum_const_zero,add_zero]
  by_cases equal : a=b
  · subst b
    have localSum (support : Fin 16) :
        (∑ offset : Fin 4,(if support=a then pulse offset else 0)*(if support=a then pulse offset else 0))=
          if support=a then 178 else 0 := by
      by_cases same : support=a
      · simp only [same,if_true,←pow_two]
        exact actual_four_pulse_pattern_has_zero_sum_and178_energy.2
      · simp [same]
    simp_rw [localSum]
    simp
  · have each (support : Fin 16) (offset : Fin 4) :
        (if support=a then pulse offset else 0)*(if support=b then pulse offset else 0)=0 := by
      by_cases left : support=a
      · simp [left,equal]
      · simp [left]
    simp_rw [each]
    simp [equal]

theorem constructed_augmented_empirical_covariance_is_actual_identity :
    empiricalCovariance pulseAugmented=1 := by
  have centered : timeCentered pulseAugmented=pulseAugmented := by
    funext row time
    simp [timeCentered,constructed_augmented_rows_have_actual_zero_time_mean]
  rw [empiricalCovariance,centered]
  ext a b
  simp only [Matrix.smul_apply,Matrix.mul_apply,Matrix.transpose_apply,smul_eq_mul,
    constructed_augmented_rows_have_actual_orthogonal178_cross_moments,Matrix.one_apply]
  by_cases equal : a=b <;> simp [equal]

theorem literal_oas_arithmetic_on_identity_is_identity :
    oasAlpha (1 : Matrix (Fin 16) (Fin 16) ℝ)=1/16 ∧
      oasMu (1 : Matrix (Fin 16) (Fin 16) ℝ)=1 ∧
      oasDenominator (1 : Matrix (Fin 16) (Fin 16) ℝ)=0 ∧
      oasRho (1 : Matrix (Fin 16) (Fin 16) ℝ)=1 := by
  have alpha : oasAlpha (1 : Matrix (Fin 16) (Fin 16) ℝ)=1/16 := by
    norm_num [oasAlpha,Matrix.one_apply]
  have mu : oasMu (1 : Matrix (Fin 16) (Fin 16) ℝ)=1 := by
    norm_num [oasMu]
  have denominator : oasDenominator (1 : Matrix (Fin 16) (Fin 16) ℝ)=0 := by
    norm_num [oasDenominator,alpha,mu]
  exact ⟨alpha,mu,denominator,actual_oas_zero_denominator_uses_full_spherical_shrinkage _ denominator⟩

theorem constructed_augmented_actual_oas_covariance_is_identity : oasCovariance pulseAugmented=1 := by
  unfold oasCovariance
  rw [constructed_augmented_empirical_covariance_is_actual_identity]
  change (1-oasRho (1 : Matrix (Fin 16) (Fin 16) ℝ)) • (1 : Matrix (Fin 16) (Fin 16) ℝ)+
    (oasRho (1 : Matrix (Fin 16) (Fin 16) ℝ)*oasMu (1 : Matrix (Fin 16) (Fin 16) ℝ)) • 1=1
  rw [literal_oas_arithmetic_on_identity_is_identity.2.2.2,literal_oas_arithmetic_on_identity_is_identity.2.1]
  simp

def pulseXDawnState : XDawnFields 8 where
  filters := 1
  prototypes row time := pulseAugmented (virtualRow (.inl row)) time
  reference := 1
  coefficient _ := 1
  intercept := 2

def pulseXDawnEpoch : ERPFeatures.Epoch 8 178 :=
  fun row time => pulseAugmented (virtualRow (.inr row)) time

theorem prototype_filter_augmentation_constructs_the_actual_fullrank_pulse_population :
    xdawnAugmented pulseXDawnState pulseXDawnEpoch=pulseAugmented := by
  have filtered : xdawnFiltered pulseXDawnState pulseXDawnEpoch=pulseXDawnEpoch := by
    simp [xdawnFiltered,pulseXDawnState]
  funext row time
  unfold xdawnAugmented
  rw [filtered]
  cases branch : virtualRow.symm row with
  | inl prototype =>
    have position : virtualRow (.inl prototype)=row := by rw [←branch]; exact virtualRow.apply_symm_apply row
    simp [pulseXDawnState,position]
  | inr observed =>
    have position : virtualRow (.inr observed)=row := by rw [←branch]; exact virtualRow.apply_symm_apply row
    simp [pulseXDawnEpoch,position]

theorem nonempty_xdawn_affine_output_from_independent_identity_cholesky_and_tangent_atoms
    (ops : NumericalOperations)
    (factor : ops.choleskyAdmission 1=.ok ())
    (matrixLogCoordinates : ops.riemannTangent 1 1=.ok (fun _ => 0)) :
    xdawnLogit ops pulseXDawnState pulseXDawnEpoch=.ok 2 := by
  unfold xdawnLogit
  rw [prototype_filter_augmentation_constructs_the_actual_fullrank_pulse_population,
    constructed_augmented_actual_oas_covariance_is_identity]
  simp [pulseXDawnState,factor,matrixLogCoordinates]

def zeroWaveform (channels : ℕ) (prior : ℝ) : WaveformFields channels :=
  ⟨fun _ => 0,fun _ => 1,fun _ => 0,prior,.float64⟩

theorem constructed_waveform_has_positive_scales_and_literal_binary_prior_logit
    (channels : ℕ) (prior : ℝ) (ops : NumericalOperations) (epoch : ERPFeatures.Epoch channels 178) :
    (∀ j,0<(zeroWaveform channels prior).scale j) ∧
      waveformLogit ops (zeroWaveform channels prior) epoch=prior := by
  simp [zeroWaveform,waveformLogit,coefficientCast]

theorem nonempty120_event36_choice_waveform_candidate_composition
    (fft : TraceFFT) (ops : NumericalOperations) (epochs : Fin 120 → ERPFeatures.Epoch 8 358)
    (prior : ℝ) :
    score fft ops (.waveform (zeroWaveform 8 prior)) .none epochs
      (SelectionERPEvidence.indexedPhysicalMatrix SelectionERPEvidence.indexedCompleteSchedule) prior=
      .ok (SelectionERPEvidence.readout (fun _ => 0)) := by
  rw [waveform_actual_candidate_score_is_derived_from_its_four_fields_and_cast_operations fft ops
    (zeroWaveform 8 prior) .none epochs _ prior
    SelectionERPEvidence.actual120_position_witness_passes_the_actual_fixed36_vocabulary_gate]
  have each (row : Fin 120) : waveformLogit ops (zeroWaveform 8 prior)
      (epochSeam fft .none (epochs row))=prior :=
    (constructed_waveform_has_positive_scales_and_literal_binary_prior_logit 8 prior ops _).2
  simp_rw [each]
  have zero : SelectionERPEvidence.evidence
      (SelectionERPEvidence.indexedPhysicalMatrix SelectionERPEvidence.indexedCompleteSchedule)
      (fun _ => prior) prior=(fun _ => 0) := by
    funext candidate
    simp [SelectionERPEvidence.literal_llr_subtraction_then_membership_transpose_sum]
  rw [zero]

theorem actual_time_constant_augmented_population_has_zero_oas_covariance :
    oasCovariance (0 : Matrix (Fin 16) (Fin 178) ℝ)=0 := by
  have centered : timeCentered (0 : Matrix (Fin 16) (Fin 178) ℝ)=0 := by
    ext row time
    simp [timeCentered]
  have covariance : empiricalCovariance (0 : Matrix (Fin 16) (Fin 178) ℝ)=0 := by
    simp [empiricalCovariance,centered]
  have mu : oasMu (0 : Matrix (Fin 16) (Fin 16) ℝ)=0 := by simp [oasMu]
  simp only [oasCovariance,covariance]
  simp [mu]

end
end N2P3.SelectionERPCaller
