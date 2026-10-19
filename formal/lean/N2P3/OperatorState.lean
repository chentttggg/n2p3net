import N2P3.PersonalAdmission
import N2P3.StateLedger
import N2P3.FrozenState
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.List.OfFn
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import N2P3.OperatorGeometry

/-!
# Literal primitive archive and constructor ledger for operator LDA

Source: operator_lda.py:47-96,153-212 and noise_model.py:82-134,193-210.
Five independently decoded primitive arrays and seven metadata fields are read.
Extra NPZ arrays are ignored; metadata extensions still enter the outer seal
before the constructor consumes its seven fields. Saved weights are never read.

This is an exact-real semantic layer after NumPy/JSON decoding. All array
elements are finite reals; ndarray layout, byte decoding/casts, JSON failures,
SHA implementations and physical I/O remain external. The covariance tolerance
validator and eigensolver are explicit numerical primitives. Passing their
source tolerances does not imply exact positive semidefiniteness or exact
contrast/feature support. No whole constructor/score roundtrip is a premise.

The joint covariance is normalized as (C+C.T)/2 before projection and operator
rebuilding. Raw sealed archive identity and the normalized constructor image
are separate. Hashes are arbitrary functions, never authentication axioms.
ArrayState represents actual ndarray contents only when its values length is
the product of its shape; this decoded-domain condition is not a new runtime
guard. Total getD reads outside that domain make no source-refinement claim.
Fields represent parsed dictionaries with unique keys; duplicated-key lists
are likewise outside the claimed Python/JSON mapping domain.
-/

namespace N2P3.OperatorState

open N2P3.FrozenState N2P3.StateLedger Matrix
open scoped BigOperators

noncomputable section

attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Primitives where
  classMeans : ArrayState
  jointMean : ArrayState
  jointCovariance : ArrayState
  rawOperator : ArrayState
  stateOperator : ArrayState

def primitiveList (a : Primitives) : List ArrayState :=
  [a.classMeans, a.jointMean, a.jointCovariance, a.rawOperator, a.stateOperator]

def encodePrimitives (a : Primitives) (extras : Fields := .nil) : Fields :=
  .cons "class_means" (encodeArray a.classMeans)
  (.cons "joint_mean" (encodeArray a.jointMean)
  (.cons "joint_covariance" (encodeArray a.jointCovariance)
  (.cons "raw_operator" (encodeArray a.rawOperator)
  (.cons "state_operator" (encodeArray a.stateOperator) extras))))

def readArray (fields : Fields) (name : String) : Check ArrayState := do
  let value ← required fields name
  match value.decodeArray with
  | some a => .ok a
  | none => .error (.fieldFailure ("array:" ++ name))

/-- Array extraction occurs inside the NPZ block, before schema/seal tests. -/
def decodePrimitives (fields : Fields) : Check Primitives := do
  let means ← readArray fields "class_means"
  let jointMean ← readArray fields "joint_mean"
  let jointCovariance ← readArray fields "joint_covariance"
  let rawOperator ← readArray fields "raw_operator"
  let stateOperator ← readArray fields "state_operator"
  pure ⟨means, jointMean, jointCovariance, rawOperator, stateOperator⟩

structure Input where
  arrays : Primitives
  ridgeRelative : Value
  inputIdentity : Value
  provenance : Value
  noiseRawDimension : Value
  noiseProvenance : Value

def metadata (ah : ArrayHash) (input : Input) : Fields := Fields.ofList
  [("schema", .text "n2p3_operator_aware_lda/1"),
   ("ridge_relative", input.ridgeRelative), ("input_identity", input.inputIdentity),
   ("provenance", input.provenance), ("noise_raw_dimension", input.noiseRawDimension),
   ("noise_provenance", input.noiseProvenance),
   ("arrays_sha256", .text (ah (primitiveList input.arrays)))]

def encodeInput (ah : ArrayHash) (input : Input) (arrayExtras : Fields := .nil) : ArchiveView :=
  ⟨encodePrimitives input.arrays arrayExtras, metadata ah input⟩

structure Envelope where
  arrays : Primitives
  unsealedMetadata : Fields

def loadEnvelope (ah : ArrayHash) (mh : FrozenState.MetadataHash)
    (archive : ArchiveView) : Check Envelope := do
  let arrays ← decodePrimitives archive.arrays
  ensure (fieldDefault archive.metadata "schema" = .text "n2p3_operator_aware_lda/1") "operator schema"
  let unsealed := archive.metadata.erase "artifact_sha256"
  ensure (fieldDefault archive.metadata "artifact_sha256" = .text (mh unsealed)) "operator metadata seal"
  ensure (fieldDefault unsealed "arrays_sha256" = .text (ah (primitiveList arrays))) "operator array seal"
  pure ⟨arrays, unsealed⟩

theorem five_arrays_decode_independently (arrays : Primitives) (extras : Fields) :
    decodePrimitives (encodePrimitives arrays extras) = .ok arrays := by
  cases arrays
  simp [decodePrimitives, encodePrimitives, readArray, required, Fields.get, decode_encode_array]

theorem erase_outer_seal (ah : ArrayHash) (input : Input) :
    (metadata ah input).erase "artifact_sha256" = metadata ah input := by
  simp [metadata, Fields.ofList, Fields.erase]

theorem encoded_primitives_reach_envelope (ah : ArrayHash) (mh : FrozenState.MetadataHash)
    (input : Input) (extras : Fields) :
    loadEnvelope ah mh (attachArtifactSeal mh (encodeInput ah input extras)) =
      .ok ⟨input.arrays, metadata ah input⟩ := by
  cases input
  simp [loadEnvelope, attachArtifactSeal, encodeInput, five_arrays_decode_independently,
    metadata, primitiveList, Fields.ofList, Fields.get, Fields.erase, fieldDefault,
    ensure]

theorem schema_rejected_before_seals (ah : ArrayHash) (mh : FrozenState.MetadataHash)
    (archive : ArchiveView) (a : Primitives)
    (ha : decodePrimitives archive.arrays = .ok a)
    (hs : fieldDefault archive.metadata "schema" ≠ .text "n2p3_operator_aware_lda/1") :
    loadEnvelope ah mh archive = .error (.sourceReject "operator schema") := by
  simp [loadEnvelope, ha, ensure, hs]

theorem metadata_seal_rejected_before_array_seal (ah : ArrayHash) (mh : FrozenState.MetadataHash)
    (archive : ArchiveView) (a : Primitives)
    (ha : decodePrimitives archive.arrays = .ok a)
    (hs : fieldDefault archive.metadata "schema" = .text "n2p3_operator_aware_lda/1")
    (hm : fieldDefault archive.metadata "artifact_sha256" ≠
      .text (mh (archive.metadata.erase "artifact_sha256"))) :
    loadEnvelope ah mh archive = .error (.sourceReject "operator metadata seal") := by
  simp [loadEnvelope, ha, ensure, hs, hm]

theorem array_seal_rejected_before_joint_constructor (ah : ArrayHash) (mh : FrozenState.MetadataHash)
    (archive : ArchiveView) (a : Primitives)
    (ha : decodePrimitives archive.arrays = .ok a)
    (hs : fieldDefault archive.metadata "schema" = .text "n2p3_operator_aware_lda/1")
    (hm : fieldDefault archive.metadata "artifact_sha256" =
      .text (mh (archive.metadata.erase "artifact_sha256")))
    (haSeal : fieldDefault (archive.metadata.erase "artifact_sha256") "arrays_sha256" ≠
      .text (ah (primitiveList a))) :
    loadEnvelope ah mh archive = .error (.sourceReject "operator array seal") := by
  simp [loadEnvelope, ha, ensure, hs, hm, haSeal]

def decodeTwoCounts : Value → Option (ℤ × ℤ)
  | .sequence (.cons (.integer a) (.cons (.integer b) .nil)) => some (a, b)
  | _ => none

def checkClassCounts (value : Value) (count : ℕ) : Check Unit :=
  match decodeTwoCounts value with
  | none => .error (.sourceReject "operator class counts")
  | some (a, b) => ensure (0 < a ∧ 0 < b ∧ a + b = (count : ℤ)) "operator class counts"

def checkOperatorLedger (noise : Fields) (provenance : Value) : Check Unit := by
  classical
  exact do
    let fields ← match provenance with
      | .object fields => .ok fields
      | _ => .error (.fieldFailure "operator provenance.get")
    ensure (fieldDefault fields "fit_scope" = .text "target_calibration" ∧
      PyTruthy (fieldDefault fields "scope_id") ∧
      PersonalAdmission.PyEq (fieldDefault fields "noise_estimation_scope") (.object noise)) "operator ledger header"
    let ids ← requireIdentifiers (fieldDefault fields "calibration_ids") "operator calibration IDs"
    checkClassCounts (fieldDefault fields "class_counts") ids.length
    requireDigest (fieldDefault fields "fit_sha256") "operator fit digest"
    requireDigest (fieldDefault fields "implementation_sha256") "operator implementation digest"
    if fieldDefault noise "fit_scope" = .text "target_calibration" then
      ensure (PersonalAdmission.PyEq (fieldDefault noise "scope_id")
        (fieldDefault fields "scope_id")) "operator target noise scope"
    else pure ()

def countValue (a b : ℤ) : Value :=
  .sequence (.cons (.integer a) (.cons (.integer b) .nil))

theorem exact_class_counts_accepted (a b : ℤ) (n : ℕ)
    (ha : 0 < a) (hb : 0 < b) (hn : a + b = (n : ℤ)) :
    checkClassCounts (countValue a b) n = .ok () := by
  simp [checkClassCounts, countValue, decodeTwoCounts, ensure, ha, hb, hn]

theorem boolean_class_count_rejected (b : Bool) :
    checkClassCounts (.sequence (.cons (.boolean b) (.cons (.integer 1) .nil))) 2 =
      .error (.sourceReject "operator class counts") := rfl

theorem operator_ledger_explicit_guards (noise fields : Fields) (ids : List String)
    (a b : ℤ) (fitSHA implementation : String)
    (hscope : fieldDefault fields "fit_scope" = .text "target_calibration")
    (hidentity : PyTruthy (fieldDefault fields "scope_id"))
    (hnoise : PersonalAdmission.PyEq (fieldDefault fields "noise_estimation_scope") (.object noise))
    (hids : fieldDefault fields "calibration_ids" = .sequence (encodeTexts ids))
    (hIdentifiers : Identifiers ids)
    (hcounts : fieldDefault fields "class_counts" = countValue a b)
    (ha : 0 < a) (hb : 0 < b) (hn : a + b = (ids.length : ℤ))
    (hfit : fieldDefault fields "fit_sha256" = .text fitSHA) (hfitFormat : LowerHex64 fitSHA)
    (himpl : fieldDefault fields "implementation_sha256" = .text implementation)
    (himplFormat : LowerHex64 implementation)
    (htarget : fieldDefault noise "fit_scope" = .text "target_calibration" →
      PersonalAdmission.PyEq (fieldDefault noise "scope_id") (fieldDefault fields "scope_id")) :
    checkOperatorLedger noise (.object fields) = .ok () := by
  classical
  have hi := (identifiers_ok_iff (.sequence (encodeTexts ids)) "operator calibration IDs" ids).mpr
    ⟨rfl, hIdentifiers⟩
  have hc := exact_class_counts_accepted a b ids.length ha hb hn
  by_cases ht : fieldDefault noise "fit_scope" = .text "target_calibration"
  · simp [checkOperatorLedger, ensure, hscope, hidentity, hnoise, hids, hi,
      hcounts, hc, hfit, requireDigest, hfitFormat, himpl, himplFormat, ht, htarget ht]
  · simp [checkOperatorLedger, ensure, hscope, hidentity, hnoise, hids, hi,
      hcounts, hc, hfit, requireDigest, hfitFormat, himpl, himplFormat, ht]

def matrixOfArray (rows cols : ℕ) (array : ArrayState) : Matrix (Fin rows) (Fin cols) ℝ :=
  fun i j => array.values.getD (i.val * cols + j.val) 0

def matrixArray {rows cols : ℕ} (matrix : Matrix (Fin rows) (Fin cols) ℝ) : ArrayState :=
  ⟨[rows, cols], List.ofFn (fun j : Fin (rows * cols) =>
    let pair := finProdFinEquiv.symm j
    matrix pair.1 pair.2)⟩

def symmetrize {d : ℕ} (matrix : Matrix (Fin d) (Fin d) ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  fun i j => (matrix i j + matrix j i) / 2

def normalizedJointCovariance (d : ℕ) (array : ArrayState) : ArrayState :=
  matrixArray (symmetrize (matrixOfArray d d array))

theorem symmetrized_entry (d : ℕ) (array : ArrayState) (i j : Fin d) :
    symmetrize (matrixOfArray d d array) i j =
      (covarianceEntry array d i.val j.val + covarianceEntry array d j.val i.val) / 2 := rfl

theorem symmetrize_is_symmetric {d : ℕ} (matrix : Matrix (Fin d) (Fin d) ℝ) :
    (symmetrize matrix)ᵀ = symmetrize matrix := by
  funext i j
  simp [Matrix.transpose_apply, symmetrize, add_comm]

theorem symmetrize_idempotent {d : ℕ} (matrix : Matrix (Fin d) (Fin d) ℝ) :
    symmetrize (symmetrize matrix) = symmetrize matrix := by
  funext i j
  simp only [symmetrize]
  ring

theorem matrix_array_recovers_entries {rows cols : ℕ}
    (matrix : Matrix (Fin rows) (Fin cols) ℝ) (i : Fin rows) (j : Fin cols) :
    matrixOfArray rows cols (matrixArray matrix) i j = matrix i j := by
  have hindex : i.val * cols + j.val = (finProdFinEquiv (i, j)).val := by
    simp [finProdFinEquiv, Nat.mul_comm, Nat.add_comm]
  simp only [matrixOfArray, matrixArray, hindex]
  rw [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
  split_ifs with h
  · simp only [Option.getD_some]
    have he : (⟨(finProdFinEquiv (i, j)).val, h⟩ : Fin (rows * cols)) =
        finProdFinEquiv (i, j) := rfl
    rw [he, Equiv.symm_apply_apply]
  · exact False.elim (h (finProdFinEquiv (i, j)).isLt)

theorem normalized_covariance_entry (d : ℕ) (array : ArrayState) (i j : Fin d) :
    matrixOfArray d d (normalizedJointCovariance d array) i j =
      (covarianceEntry array d i.val j.val + covarianceEntry array d j.val i.val) / 2 := by
  rw [normalizedJointCovariance, matrix_array_recovers_entries]
  rfl

structure Spectrum (d : ℕ) where
  values : Fin d → ℝ
  vectors : Matrix (Fin d) (Fin d) ℝ

/-- These external operations preserve their actual guard outcomes. In
particular true covarianceTolerancePSD does not assert exact PSD. -/
structure Numerics where
  covarianceTolerancePSD : ArrayState → Bool
  eigh : (d : ℕ) → Matrix (Fin d) (Fin d) ℝ → Option (Spectrum d)

structure NoiseImage where
  mean : ArrayState
  covariance : ArrayState
  rawDimension : ℕ
  provenance : Fields

def constructNoise (numerics : Numerics) (arrays : Primitives)
    (rawDimension provenance : Value) : Check NoiseImage := do
  let covariance := arrays.jointCovariance
  ensure (∃ n, covariance.shape = [n, n]) "joint covariance square"
  ensure (numerics.covarianceTolerancePSD covariance = true) "joint covariance tolerance/PSD"
  let n := arrays.jointMean.values.length
  ensure (arrays.jointMean.shape = [n] ∧ covariance.shape = [n, n]) "joint mean/covariance dimensions"
  let raw ← match rawDimension with
    | .integer z => do
      ensure (0 < z ∧ z ≤ (n : ℤ)) "joint raw dimension"
      pure z.toNat
    | _ => .error (.sourceReject "joint raw dimension")
  checkJoint provenance
  let fields ← match provenance with
    | .object fields => .ok fields
    | _ => .error (.sourceReject "joint_header")
  pure ⟨arrays.jointMean, normalizedJointCovariance n covariance, raw, fields⟩

def jointOperator (noise : NoiseImage) (arrays : Primitives) (d : ℕ) :
    Matrix (Fin d) (Fin noise.mean.values.length) ℝ := fun i j =>
  if j.val < noise.rawDimension then
    arrays.rawOperator.values.getD (i.val * noise.rawDimension + j.val) 0
  else
    arrays.stateOperator.values.getD
      (i.val * (noise.mean.values.length - noise.rawDimension) + (j.val - noise.rawDimension)) 0

def projectedCovariance (noise : NoiseImage) (arrays : Primitives) (d : ℕ) :
    Matrix (Fin d) (Fin d) ℝ :=
  let C := jointOperator noise arrays d
  symmetrize (C * matrixOfArray noise.mean.values.length noise.mean.values.length noise.covariance * Cᵀ)

def classMean (arrays : Primitives) (d : ℕ) (classIndex : Fin 2) : Fin d → ℝ :=
  fun j => arrays.classMeans.values.getD (classIndex.val * d + j.val) 0

def retainedColumns {d : ℕ} (spectrum : Spectrum d) (maximum : ℝ) :
    Matrix (Fin d) (OperatorGeometry.Retained spectrum.values maximum) ℝ :=
  OperatorGeometry.thinColumns spectrum.vectors spectrum.values maximum

def retainedValues {d : ℕ} (spectrum : Spectrum d) (maximum : ℝ) :
    OperatorGeometry.Retained spectrum.values maximum → ℝ := fun k => spectrum.values k.val

structure Image where
  input : Input
  noise : NoiseImage
  outputDimension : ℕ
  spectrum : Spectrum outputDimension
  maximum : ℝ
  ridge : ℝ
  combinedOperator : Matrix (Fin outputDimension) (Fin noise.mean.values.length) ℝ
  projected : Matrix (Fin outputDimension) (Fin outputDimension) ℝ
  covariance : Matrix (Fin outputDimension) (Fin outputDimension) ℝ
  identifiableProjector : Matrix (Fin outputDimension) (Fin outputDimension) ℝ
  weights : Fin outputDimension → ℝ
  intercept : ℝ

def canonicalImageInput (input : Input) (noise : NoiseImage) : Input :=
  ⟨⟨input.arrays.classMeans, noise.mean, noise.covariance,
      input.arrays.rawOperator, input.arrays.stateOperator⟩,
    input.ridgeRelative, input.inputIdentity, input.provenance,
    .integer noise.rawDimension, .object noise.provenance⟩

def rebuildImage (input : Input) (noise : NoiseImage) (d : ℕ)
    (spectrum : Spectrum d) (maximum ridge : ℝ) : Image := by
  classical
  let V := retainedColumns spectrum maximum
  let values := retainedValues spectrum maximum
  let mu0 := classMean input.arrays d 0
  let mu1 := classMean input.arrays d 1
  let w := OperatorGeometry.weights V values ridge (mu1 - mu0)
  exact ⟨canonicalImageInput input noise, noise, d, spectrum, maximum, ridge,
    jointOperator noise input.arrays d, projectedCovariance noise input.arrays d,
    OperatorGeometry.regularizedCovariance V values ridge,
    OperatorGeometry.projector V, w, OperatorGeometry.intercept mu0 mu1 w⟩

def constructOperatorAt (numerics : Numerics) (input : Input) (noise : NoiseImage) (d : ℕ) : Check Image := by
  classical
  exact do
    ensure (PyTruthy input.inputIdentity) "operator input identity"
    let ridge ← match numericScalar input.ridgeRelative with
      | some ridge => do
        ensure (0 ≤ ridge) "operator ridge"
        pure ridge
      | none => .error (.fieldFailure "operator ridge numeric")
    ensure (input.arrays.rawOperator.shape = [d, noise.rawDimension] ∧
      input.arrays.stateOperator.shape = [d, noise.mean.values.length - noise.rawDimension]) "operator dimensions"
    let covariance := projectedCovariance noise input.arrays d
    ensure (input.arrays.classMeans.shape = [2, d]) "operator class means"
    let spectrum ← match numerics.eigh d covariance with
      | some spectrum => .ok spectrum
      | none => .error (.fieldFailure "operator eigh")
    if hd : 0 < d then
      let top : Fin d := ⟨d - 1, by omega⟩
      let maximum := spectrum.values top
      ensure (0 < maximum) "operator no identifiable stochastic direction"
      let V := retainedColumns spectrum maximum
      let delta := classMean input.arrays d 1 - classMean input.arrays d 0
      ensure (OperatorGeometry.contrastAccepted V delta) "operator contrast outside support"
      checkOperatorLedger noise.provenance input.provenance
      pure (rebuildImage input noise d spectrum maximum ridge)
    else .error (.fieldFailure "eigh_maximum_index")

def constructOperator (numerics : Numerics) (input : Input) (noise : NoiseImage) : Check Image :=
  constructOperatorAt numerics input noise (input.arrays.rawOperator.shape.headD 0)

/-- JointNoiseModel is fully constructed before the source indexes ridge,
input_identity or operator provenance. Missing later fields cannot preempt it. -/
def load (numerics : Numerics) (ah : ArrayHash) (mh : FrozenState.MetadataHash)
    (archive : ArchiveView) : Check Image := do
  let envelope ← loadEnvelope ah mh archive
  let raw ← required envelope.unsealedMetadata "noise_raw_dimension"
  let noiseProvenance ← required envelope.unsealedMetadata "noise_provenance"
  let noise ← constructNoise numerics envelope.arrays raw noiseProvenance
  let ridge ← required envelope.unsealedMetadata "ridge_relative"
  let identity ← required envelope.unsealedMetadata "input_identity"
  let provenance ← required envelope.unsealedMetadata "provenance"
  constructOperator numerics ⟨envelope.arrays, ridge, identity, provenance, raw, noiseProvenance⟩ noise

def reconstruct (numerics : Numerics) (input : Input) : Check Image := do
  let noise ← constructNoise numerics input.arrays input.noiseRawDimension input.noiseProvenance
  constructOperator numerics input noise

theorem encoded_load_runs_literal_constructors (numerics : Numerics)
    (ah : ArrayHash) (mh : FrozenState.MetadataHash) (input : Input) (extras : Fields) :
    load numerics ah mh (attachArtifactSeal mh (encodeInput ah input extras)) = reconstruct numerics input := by
  simp [load, encoded_primitives_reach_envelope, metadata, Fields.ofList, required, Fields.get, reconstruct]

theorem joint_failure_precedes_operator_fields (numerics : Numerics)
    (ah : ArrayHash) (mh : FrozenState.MetadataHash) (archive : ArchiveView)
    (envelope : Envelope) (raw provenance : Value) (failure : LedgerFault)
    (he : loadEnvelope ah mh archive = .ok envelope)
    (hr : envelope.unsealedMetadata.get "noise_raw_dimension" = some raw)
    (hp : envelope.unsealedMetadata.get "noise_provenance" = some provenance)
    (hn : constructNoise numerics envelope.arrays raw provenance = .error failure) :
    load numerics ah mh archive = .error failure := by
  simp [load, he, required, hr, hp, hn]

theorem rebuilt_weights_are_from_retained_geometry (input : Input) (noise : NoiseImage)
    (d : ℕ) (spectrum : Spectrum d) (maximum ridge : ℝ) :
    (rebuildImage input noise d spectrum maximum ridge).weights =
      OperatorGeometry.weights (retainedColumns spectrum maximum) (retainedValues spectrum maximum)
        ridge (classMean input.arrays d 1 - classMean input.arrays d 0) := by
  classical
  rfl

theorem rebuilt_covariance_is_retained_ridge (input : Input) (noise : NoiseImage)
    (d : ℕ) (spectrum : Spectrum d) (maximum ridge : ℝ) :
    (rebuildImage input noise d spectrum maximum ridge).covariance =
      OperatorGeometry.regularizedCovariance (retainedColumns spectrum maximum)
        (retainedValues spectrum maximum) ridge := by
  classical
  rfl

theorem rebuilt_score_is_retained_quadratic_contrast (input : Input) (noise : NoiseImage)
    (d : ℕ) (spectrum : Spectrum d) (maximum ridge : ℝ) (x : Fin d → ℝ) :
    OperatorGeometry.logit x (rebuildImage input noise d spectrum maximum ridge).weights
      (rebuildImage input noise d spectrum maximum ridge).intercept =
      (OperatorGeometry.retainedEnergy (retainedColumns spectrum maximum) (retainedValues spectrum maximum)
        ridge x (classMean input.arrays d 0) -
       OperatorGeometry.retainedEnergy (retainedColumns spectrum maximum) (retainedValues spectrum maximum)
        ridge x (classMean input.arrays d 1)) / 2 := by
  classical
  exact OperatorGeometry.source_logit_is_retained_quadratic_contrast
    (retainedColumns spectrum maximum) (retainedValues spectrum maximum) ridge
    (classMean input.arrays d 0) (classMean input.arrays d 1) x

theorem construct_noise_explicit_guards (numerics : Numerics) (arrays : Primitives)
    (raw : ℕ) (ledger : Fields)
    (_hsquare : ∃ n, arrays.jointCovariance.shape = [n, n])
    (htolerance : numerics.covarianceTolerancePSD arrays.jointCovariance = true)
    (hshapes : arrays.jointMean.shape = [arrays.jointMean.values.length] ∧
      arrays.jointCovariance.shape = [arrays.jointMean.values.length, arrays.jointMean.values.length])
    (hraw : 0 < raw ∧ raw ≤ arrays.jointMean.values.length)
    (hledger : checkJoint (.object ledger) = .ok ()) :
    constructNoise numerics arrays (.integer raw) (.object ledger) = .ok
      ⟨arrays.jointMean, normalizedJointCovariance arrays.jointMean.values.length arrays.jointCovariance,
        raw, ledger⟩ := by
  have hr : (0 : ℤ) < raw ∧ (raw : ℤ) ≤ arrays.jointMean.values.length := by omega
  simp [constructNoise, ensure, htolerance, hshapes, hr, hraw.1, hledger]

theorem construct_operator_explicit_guards (numerics : Numerics) (input : Input)
    (noise : NoiseImage) (d : ℕ) (spectrum : Spectrum d) (ridge : ℝ)
    (hidentity : PyTruthy input.inputIdentity)
    (hridge : numericScalar input.ridgeRelative = some ridge) (hridgeNonnegative : 0 ≤ ridge)
    (hrow : input.arrays.rawOperator.shape.headD 0 = d)
    (hshapes : input.arrays.rawOperator.shape = [d, noise.rawDimension] ∧
      input.arrays.stateOperator.shape = [d, noise.mean.values.length - noise.rawDimension])
    (hmeans : input.arrays.classMeans.shape = [2, d])
    (heigh : numerics.eigh d (projectedCovariance noise input.arrays d) = some spectrum)
    (hd : 0 < d)
    (hmaximum : 0 < spectrum.values ⟨d - 1, by omega⟩)
    (hcontrast : OperatorGeometry.contrastAccepted
      (retainedColumns spectrum (spectrum.values ⟨d - 1, by omega⟩))
      (classMean input.arrays d 1 - classMean input.arrays d 0))
    (hledger : checkOperatorLedger noise.provenance input.provenance = .ok ()) :
    constructOperator numerics input noise =
      .ok (rebuildImage input noise d spectrum (spectrum.values ⟨d - 1, by omega⟩) ridge) := by
  rw [constructOperator, hrow]
  simp [constructOperatorAt, ensure, hidentity, hridge, hridgeNonnegative,
    hshapes, hmeans, heigh, hd, hmaximum, hcontrast, hledger]

theorem zero_output_reaches_actual_maximum_index_failure (numerics : Numerics)
    (input : Input) (noise : NoiseImage) (spectrum : Spectrum 0) (ridge : ℝ)
    (hidentity : PyTruthy input.inputIdentity)
    (hridge : numericScalar input.ridgeRelative = some ridge) (hridgeNonnegative : 0 ≤ ridge)
    (hrow : input.arrays.rawOperator.shape.headD 0 = 0)
    (hshapes : input.arrays.rawOperator.shape = [0, noise.rawDimension] ∧
      input.arrays.stateOperator.shape = [0, noise.mean.values.length - noise.rawDimension])
    (hmeans : input.arrays.classMeans.shape = [2, 0])
    (heigh : numerics.eigh 0 (projectedCovariance noise input.arrays 0) = some spectrum) :
    constructOperator numerics input noise = .error (.fieldFailure "eigh_maximum_index") := by
  rw [constructOperator, hrow]
  simp [constructOperatorAt, ensure, hidentity, hridge, hridgeNonnegative,
    hshapes, hmeans, heigh]

theorem rebuilt_projector_and_intercept (input : Input) (noise : NoiseImage)
    (d : ℕ) (spectrum : Spectrum d) (maximum ridge : ℝ) :
    (rebuildImage input noise d spectrum maximum ridge).identifiableProjector =
      OperatorGeometry.projector (retainedColumns spectrum maximum) ∧
    (rebuildImage input noise d spectrum maximum ridge).intercept =
      OperatorGeometry.intercept (classMean input.arrays d 0) (classMean input.arrays d 1)
        (OperatorGeometry.weights (retainedColumns spectrum maximum) (retainedValues spectrum maximum)
          ridge (classMean input.arrays d 1 - classMean input.arrays d 0)) := by
  classical
  exact ⟨rfl, rfl⟩

theorem rebuilt_keeps_four_primitives_and_normalized_noise (input : Input) (noise : NoiseImage)
    (d : ℕ) (spectrum : Spectrum d) (maximum ridge : ℝ) :
    (rebuildImage input noise d spectrum maximum ridge).input.arrays.classMeans = input.arrays.classMeans ∧
    (rebuildImage input noise d spectrum maximum ridge).input.arrays.jointMean = noise.mean ∧
    (rebuildImage input noise d spectrum maximum ridge).input.arrays.jointCovariance = noise.covariance ∧
    (rebuildImage input noise d spectrum maximum ridge).input.arrays.rawOperator = input.arrays.rawOperator ∧
    (rebuildImage input noise d spectrum maximum ridge).input.arrays.stateOperator = input.arrays.stateOperator := by
  classical
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

theorem extra_npz_arrays_are_ignored (numerics : Numerics) (ah : ArrayHash)
    (mh : FrozenState.MetadataHash) (input : Input) (extra1 extra2 : Fields) :
    load numerics ah mh (attachArtifactSeal mh (encodeInput ah input extra1)) =
      load numerics ah mh (attachArtifactSeal mh (encodeInput ah input extra2)) := by
  rw [encoded_load_runs_literal_constructors, encoded_load_runs_literal_constructors]

/-- Source prediction after the separate runtime-integrity assertion. The
admission norm is over the complete centered matrix, including all batch rows. -/
def predictAfterIntegrity (image : Image) (identity : Value) (features : ArrayState) : Check ArrayState := do
  ensure (PersonalAdmission.PyEq identity image.input.inputIdentity) "operator prediction identity"
  let rows := features.shape.headD 0
  ensure (features.shape = [rows, image.outputDimension]) "operator prediction feature shape"
  let X := matrixOfArray rows image.outputDimension features
  let center := fun j => (classMean image.input.arrays image.outputDimension 0 j +
    classMean image.input.arrays image.outputDimension 1 j) / 2
  let centered := fun i j => X i j - center j
  ensure (OperatorGeometry.batchAccepted (retainedColumns image.spectrum image.maximum) centered)
    "operator prediction batch outside support"
  pure ⟨[rows], List.ofFn (fun i => OperatorGeometry.logit (X i) image.weights image.intercept)⟩

theorem prediction_uses_full_centered_batch (image : Image) (identity : Value)
    (features : ArrayState)
    (hid : PersonalAdmission.PyEq identity image.input.inputIdentity)
    (hshape : features.shape = [features.shape.headD 0, image.outputDimension])
    (hbatch : OperatorGeometry.batchAccepted (retainedColumns image.spectrum image.maximum)
      (fun i j => matrixOfArray (features.shape.headD 0) image.outputDimension features i j -
        (classMean image.input.arrays image.outputDimension 0 j +
         classMean image.input.arrays image.outputDimension 1 j) / 2)) :
    predictAfterIntegrity image identity features = .ok
      ⟨[features.shape.headD 0], List.ofFn (fun i => OperatorGeometry.logit
        (matrixOfArray (features.shape.headD 0) image.outputDimension features i) image.weights image.intercept)⟩ := by
  simp only [predictAfterIntegrity, ensure, if_pos hid, Except.bind, Bind.bind,
    if_pos hshape, if_pos hbatch, Pure.pure, Except.pure]

theorem prediction_identity_rejected_before_feature_checks (image : Image) (identity : Value)
    (features : ArrayState) (hid : ¬PersonalAdmission.PyEq identity image.input.inputIdentity) :
    predictAfterIntegrity image identity features = .error (.sourceReject "operator prediction identity") := by
  simp [predictAfterIntegrity, ensure, hid]

def fixtureNoiseLedger (scope : FitScope) : Fields := Fields.ofList
  [("fit_scope", .text scope.text), ("scope_id", .text "noise-scope"),
   ("draw_ids", .sequence (encodeTexts ["draw-0", " draw-0", "draw-2"])),
   ("n_draws", .integer 3), ("input_sha256", .text zeroDigest),
   ("shrinkage", .number 0), ("implementation_sha256", .text zeroDigest),
   ("estimator", .text "joint_standardized_covariance_with_cross_terms")]

theorem fixture_joint_ledger_passes (scope : FitScope) :
    checkJoint (.object (fixtureNoiseLedger scope)) = .ok () := by
  change checkJoint (producedJoint scope "noise-scope" ["draw-0", " draw-0", "draw-2"]
    zeroDigest zeroDigest 0) = .ok ()
  apply produced_joint_contract_accepted
  · rfl
  · exact draw_identifiers_valid
  · decide
  · exact zero_digest_has_lowercase_hex_format
  · exact zero_digest_has_lowercase_hex_format
  · norm_num

def fixtureCalibrationIds : List String := ["calibration-0", "calibration-1"]

theorem fixture_calibration_ids_valid : Identifiers fixtureCalibrationIds := by
  have h0 : Nonblank "calibration-0" := rfl
  have h1 : Nonblank "calibration-1" := rfl
  refine ⟨by decide, ?_, by decide⟩
  intro s hs
  rcases List.mem_cons.mp hs with h | h
  · subst s; exact h0
  · have he := List.mem_singleton.mp h
    subst s
    exact h1

def fixtureOperatorFields (noise : Fields) (scopeIdentity : Value) : Fields := Fields.ofList
  [("fit_scope", .text "target_calibration"), ("scope_id", scopeIdentity),
   ("noise_estimation_scope", .object noise),
   ("calibration_ids", .sequence (encodeTexts fixtureCalibrationIds)),
   ("class_counts", countValue 1 1), ("fit_sha256", .text zeroDigest),
   ("implementation_sha256", .text zeroDigest), ("class_prior", .text "equal"),
   ("oracle_state_at_predict", .boolean false)]

theorem source_noise_allows_distinct_truthy_scope :
    checkOperatorLedger (fixtureNoiseLedger .sourceTraining)
      (.object (fixtureOperatorFields (fixtureNoiseLedger .sourceTraining) (.boolean true))) = .ok () := by
  have hi := (identifiers_ok_iff (.sequence (encodeTexts fixtureCalibrationIds))
    "operator calibration IDs" fixtureCalibrationIds).mpr ⟨rfl, fixture_calibration_ids_valid⟩
  dsimp [fixtureCalibrationIds] at hi
  simp [checkOperatorLedger, fixtureOperatorFields, fixtureNoiseLedger, fieldDefault,
    Fields.ofList, Fields.get, FitScope.text, PyTruthy, PersonalAdmission.PyEq, ensure,
    hi, checkClassCounts, countValue, decodeTwoCounts, fixtureCalibrationIds, requireDigest,
    zero_digest_has_lowercase_hex_format]

theorem target_noise_rejects_distinct_truthy_scope :
    checkOperatorLedger (fixtureNoiseLedger .targetCalibration)
      (.object (fixtureOperatorFields (fixtureNoiseLedger .targetCalibration) (.boolean true))) =
      .error (.sourceReject "operator target noise scope") := by
  have hi := (identifiers_ok_iff (.sequence (encodeTexts fixtureCalibrationIds))
    "operator calibration IDs" fixtureCalibrationIds).mpr ⟨rfl, fixture_calibration_ids_valid⟩
  dsimp [fixtureCalibrationIds] at hi
  simp [checkOperatorLedger, fixtureOperatorFields, fixtureNoiseLedger, fieldDefault,
    Fields.ofList, Fields.get, FitScope.text, PyTruthy, PersonalAdmission.PyEq,
    PersonalAdmission.normalizeValue, ensure, hi, checkClassCounts, countValue, decodeTwoCounts,
    fixtureCalibrationIds, requireDigest, zero_digest_has_lowercase_hex_format]

def fixtureArrays : Primitives :=
  ⟨⟨[2, 1], [0, 2]⟩, ⟨[1], [0]⟩, ⟨[1, 1], [1]⟩, ⟨[1, 1], [1]⟩, ⟨[1, 0], []⟩⟩

def fixtureInput : Input :=
  ⟨fixtureArrays, .boolean false, .boolean true,
    .object (fixtureOperatorFields (fixtureNoiseLedger .sourceTraining) (.boolean true)),
    .integer 1, .object (fixtureNoiseLedger .sourceTraining)⟩

/-- This fixture's projected covariance is identity; the supplied numerical
output below is its genuine exact identity eigensystem. No general eigensolver
correctness is inferred from the constant fixture implementation. -/
def identitySpectrum (d : ℕ) : Spectrum d := ⟨fun _ => 1, 1⟩

def fixtureNumerics : Numerics := ⟨fun _ => true, fun d _ => some (identitySpectrum d)⟩

def fixtureNoise : NoiseImage :=
  ⟨fixtureArrays.jointMean, normalizedJointCovariance 1 fixtureArrays.jointCovariance,
    1, fixtureNoiseLedger .sourceTraining⟩

def oneRetained : OperatorGeometry.Retained (fun _ : Fin 1 => (1 : ℝ)) 1 :=
  ⟨0, by norm_num [OperatorGeometry.eigenTolerance]⟩

theorem one_retained_unique (k : OperatorGeometry.Retained (fun _ : Fin 1 => (1 : ℝ)) 1) :
    k = oneRetained := by
  apply Subtype.ext
  exact Subsingleton.elim _ _

theorem identity_fixture_projector :
    OperatorGeometry.projector (retainedColumns (identitySpectrum 1) 1) = 1 := by
  classical
  letI : Unique (OperatorGeometry.Retained (fun _ : Fin 1 => (1 : ℝ)) 1) :=
    ⟨⟨oneRetained⟩, one_retained_unique⟩
  have hval : (default : OperatorGeometry.Retained (fun _ : Fin 1 => (1 : ℝ)) 1).val = 0 :=
    Subsingleton.elim _ _
  ext i j
  fin_cases i
  fin_cases j
  simp [OperatorGeometry.projector, retainedColumns, OperatorGeometry.thinColumns,
    identitySpectrum, Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply, hval]

theorem fixture_contrast_passes :
    OperatorGeometry.contrastAccepted (retainedColumns (identitySpectrum 1) 1)
      (classMean fixtureArrays 1 1 - classMean fixtureArrays 1 0) := by
  apply OperatorGeometry.exact_support_passes_source_contrast_gate
  rw [identity_fixture_projector, Matrix.one_mulVec]

theorem fixture_noise_constructs :
    constructNoise fixtureNumerics fixtureArrays (.integer 1)
      (.object (fixtureNoiseLedger .sourceTraining)) = .ok fixtureNoise := by
  apply construct_noise_explicit_guards
  · exact ⟨1, rfl⟩
  · rfl
  · exact ⟨rfl, rfl⟩
  · norm_num [fixtureArrays]
  · exact fixture_joint_ledger_passes .sourceTraining

theorem fixture_operator_constructs :
    constructOperator fixtureNumerics fixtureInput fixtureNoise =
      .ok (rebuildImage fixtureInput fixtureNoise 1 (identitySpectrum 1) 1 0) := by
  exact construct_operator_explicit_guards fixtureNumerics fixtureInput fixtureNoise 1
    (identitySpectrum 1) 0 rfl rfl (by norm_num) rfl ⟨rfl, rfl⟩ rfl rfl (by decide)
    (by norm_num [identitySpectrum]) fixture_contrast_passes source_noise_allows_distinct_truthy_scope

theorem nonempty_operator_sealed_load_constructs (ah : ArrayHash) (mh : FrozenState.MetadataHash)
    (extras : Fields) :
    load fixtureNumerics ah mh (attachArtifactSeal mh (encodeInput ah fixtureInput extras)) =
      .ok (rebuildImage fixtureInput fixtureNoise 1 (identitySpectrum 1) 1 0) := by
  rw [encoded_load_runs_literal_constructors]
  simp only [reconstruct, fixtureInput, fixture_noise_constructs]
  exact fixture_operator_constructs

theorem fixture_numerical_input_is_identity :
    projectedCovariance fixtureNoise fixtureArrays 1 = 1 := by
  have hC : jointOperator fixtureNoise fixtureArrays 1 = (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
    ext i j
    fin_cases i
    fin_cases j
    norm_num [jointOperator, fixtureNoise, fixtureArrays]
  have hN : matrixOfArray 1 1 fixtureNoise.covariance = (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
    ext i j
    fin_cases i
    fin_cases j
    change matrixOfArray 1 1 (normalizedJointCovariance 1 fixtureArrays.jointCovariance) 0 0 = 1
    rw [normalized_covariance_entry]
    norm_num [covarianceEntry, fixtureArrays]
  unfold projectedCovariance
  rw [hC]
  change symmetrize ((1 : Matrix (Fin 1) (Fin 1) ℝ) *
    matrixOfArray 1 1 fixtureNoise.covariance * (1 : Matrix (Fin 1) (Fin 1) ℝ)ᵀ) = 1
  rw [hN]
  ext i j
  fin_cases i
  fin_cases j
  norm_num [symmetrize]

theorem fixture_eigh_is_actual_exact_spectral_data :
    (identitySpectrum 1).vectorsᵀ * (identitySpectrum 1).vectors = 1 ∧
    projectedCovariance fixtureNoise fixtureArrays 1 =
      (identitySpectrum 1).vectors * Matrix.diagonal (identitySpectrum 1).values *
        (identitySpectrum 1).vectorsᵀ := by
  simp [identitySpectrum, fixture_numerical_input_is_identity]

theorem fixture_basis_entry (i : Fin 1)
    (k : OperatorGeometry.Retained (fun _ : Fin 1 => (1 : ℝ)) 1) :
    retainedColumns (identitySpectrum 1) 1 i k = 1 := by
  simp [retainedColumns, OperatorGeometry.thinColumns, identitySpectrum,
    Subsingleton.elim i k.val]

theorem fixture_rebuilt_weight_is_two (i : Fin 1) :
    (rebuildImage fixtureInput fixtureNoise 1 (identitySpectrum 1) 1 0).weights i = 2 := by
  classical
  letI : Unique (OperatorGeometry.Retained (fun _ : Fin 1 => (1 : ℝ)) 1) :=
    ⟨⟨oneRetained⟩, one_retained_unique⟩
  rw [rebuilt_weights_are_from_retained_geometry]
  simp only [OperatorGeometry.weights, Matrix.mulVec, dotProduct, Matrix.transpose_apply]
  simp only [fixture_basis_entry]
  simp [OperatorGeometry.regularized, retainedValues, identitySpectrum,
    classMean, fixtureInput, fixtureArrays]

theorem fixture_rebuilt_score_is_four :
    OperatorGeometry.logit (fun _ => 3)
      (rebuildImage fixtureInput fixtureNoise 1 (identitySpectrum 1) 1 0).weights
      (rebuildImage fixtureInput fixtureNoise 1 (identitySpectrum 1) 1 0).intercept = 4 := by
  have hw : (rebuildImage fixtureInput fixtureNoise 1 (identitySpectrum 1) 1 0).weights =
      fun _ => 2 := funext fixture_rebuilt_weight_is_two
  have hb : (rebuildImage fixtureInput fixtureNoise 1 (identitySpectrum 1) 1 0).intercept = -2 := by
    change OperatorGeometry.intercept (classMean fixtureInput.arrays 1 0)
      (classMean fixtureInput.arrays 1 1)
      (rebuildImage fixtureInput fixtureNoise 1 (identitySpectrum 1) 1 0).weights = -2
    rw [hw]
    norm_num [OperatorGeometry.intercept, classMean, fixtureInput, fixtureArrays, dotProduct, Fin.sum_univ_one]
  rw [OperatorGeometry.logit, hw, hb]
  change (∑ _ : Fin 1, (3 : ℝ) * 2) + -2 = 4
  norm_num [dotProduct, Fin.sum_univ_one]

theorem nonempty_loaded_operator_scores_four (ah : ArrayHash) (mh : FrozenState.MetadataHash)
    (extras : Fields) :
    (load fixtureNumerics ah mh (attachArtifactSeal mh (encodeInput ah fixtureInput extras))).map
      (fun image => OperatorGeometry.logit (fun _ => 3) image.weights image.intercept) = .ok 4 := by
  rw [nonempty_operator_sealed_load_constructs]
  exact congrArg Except.ok fixture_rebuilt_score_is_four

theorem raw_covariance_entry_changes_under_normalization :
    matrixOfArray 2 2 (normalizedJointCovariance 2 slightlyAsymmetricCovariance) 0 1 ≠
      covarianceEntry slightlyAsymmetricCovariance 2 0 1 := by
  rw [normalized_covariance_entry]
  norm_num [covarianceEntry, slightlyAsymmetricCovariance]

end

end N2P3.OperatorState
