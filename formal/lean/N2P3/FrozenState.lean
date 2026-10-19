import Mathlib.Data.Real.Basic
import Mathlib.Data.List.Nodup
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# Full-field semantic codecs and the shared sealed-state protocol

Source: continuous_decision.py:122-188,335-373 and noise_model.py:22-78,
100-134,190-241. `ArchiveView` is an independently constructed formal codec
for already decoded named arrays and JSON-object metadata. No refinement of
Python json, NumPy NPZ/compression/dtype coercion, little-endian float bytes,
SHA256, physical I/O or exclusive file creation is asserted. Object fields
stand for unique-key parsed dictionaries; hash functions may canonicalize them
but are otherwise arbitrary, with no injectivity/authentication assumption.

All ten continuous and all four joint-noise semantic fields are represented.
Numerical arrays include shape and ordered values; provenance is a recursive
value tree, never omitted from metadata. Integers/reals/booleans are distinct
semantic tags; there are no IEEE NaN/infinity values in the real domain.

The returned value is `readyInput`, an input to the numerical constructor.
Explicit core checks cover dimensions, payload lengths, phi, sampling rate,
channel/candidate identity lists, raw_dimension and provenance object shape.
Not covered: tolerance-based PSD/symmetry acceptance, eigenvalue/Cholesky
validation, floating-point normalization, continuous analytic/training ledger
rules and censor counts, noise ledger exact keys/draw IDs/count/digests/shrinkage.
The real constructors also symmetrize covariance. A separate exact-real lemma
shows entry preservation on symmetric normal forms; hash success alone never
implies that constructor normalization leaves arbitrary covariance unchanged.

Recursive copying proves value preservation in this immutable mathematical
ADT. It does not prove Python object aliasing, write protection or immutable
bytes ownership; those source behaviors require independent execution probes.
-/

namespace N2P3.FrozenState

noncomputable section

attribute [local instance] Classical.propDecidable

mutual
inductive Value where
  | null
  | boolean (b : Bool)
  | integer (z : ℤ)
  | number (r : ℝ)
  | text (s : String)
  | sequence (items : Values)
  | object (fields : Fields)
inductive Values where
  | nil
  | cons (head : Value) (tail : Values)
inductive Fields where
  | nil
  | cons (key : String) (value : Value) (tail : Fields)
end

namespace Fields

def ofList : List (String × Value) → Fields
  | [] => .nil
  | (k, v) :: tail => .cons k v (ofList tail)

def get (key : String) : Fields → Option Value
  | .nil => none
  | .cons k v tail => if key = k then some v else get key tail

def erase (key : String) : Fields → Fields
  | .nil => .nil
  | .cons k v tail => if key = k then erase key tail else .cons k v (erase key tail)

end Fields

mutual
def copyValue : Value → Value
  | .null => .null
  | .boolean b => .boolean b
  | .integer z => .integer z
  | .number r => .number r
  | .text s => .text s
  | .sequence xs => .sequence (copyValues xs)
  | .object fs => .object (copyFields fs)
def copyValues : Values → Values
  | .nil => .nil
  | .cons v tail => .cons (copyValue v) (copyValues tail)
def copyFields : Fields → Fields
  | .nil => .nil
  | .cons k v tail => .cons k (copyValue v) (copyFields tail)
end

mutual
theorem copyValue_preserves : ∀ v : Value, copyValue v = v
  | .null => rfl
  | .boolean _ => rfl
  | .integer _ => rfl
  | .number _ => rfl
  | .text _ => rfl
  | .sequence xs => by rw [copyValue, copyValues_preserves xs]
  | .object fs => by rw [copyValue, copyFields_preserves fs]
theorem copyValues_preserves : ∀ xs : Values, copyValues xs = xs
  | .nil => rfl
  | .cons v tail => by rw [copyValues, copyValue_preserves v, copyValues_preserves tail]
theorem copyFields_preserves : ∀ fs : Fields, copyFields fs = fs
  | .nil => rfl
  | .cons k v tail => by rw [copyFields, copyValue_preserves v, copyFields_preserves tail]
end

def encodeNats : List ℕ → Values
  | [] => .nil
  | n :: tail => .cons (.integer n) (encodeNats tail)

def decodeNats : Values → Option (List ℕ)
  | .nil => some []
  | .cons (.integer n) tail => if 0 ≤ n then (decodeNats tail).map (n.toNat :: ·) else none
  | _ => none

def encodeReals : List ℝ → Values
  | [] => .nil
  | n :: tail => .cons (.number n) (encodeReals tail)

def decodeReals : Values → Option (List ℝ)
  | .nil => some []
  | .cons (.number n) tail => (decodeReals tail).map (n :: ·)
  | _ => none

def encodeTexts : List String → Values
  | [] => .nil
  | s :: tail => .cons (.text s) (encodeTexts tail)

def decodeTexts : Values → Option (List String)
  | .nil => some []
  | .cons (.text s) tail => (decodeTexts tail).map (s :: ·)
  | _ => none

theorem decode_encode_nats (xs : List ℕ) : decodeNats (encodeNats xs) = some xs := by
  induction xs with
  | nil => rfl
  | cons h tail ih => simp [encodeNats, decodeNats, ih]

theorem decode_encode_reals (xs : List ℝ) : decodeReals (encodeReals xs) = some xs := by
  induction xs with
  | nil => rfl
  | cons h tail ih => simp [encodeReals, decodeReals, ih]

theorem decode_encode_texts (xs : List String) : decodeTexts (encodeTexts xs) = some xs := by
  induction xs with
  | nil => rfl
  | cons h tail ih => simp [encodeTexts, decodeTexts, ih]

structure ArrayState where
  shape : List ℕ
  values : List ℝ

def encodeArray (a : ArrayState) : Value :=
  .object (Fields.ofList [("shape", .sequence (encodeNats a.shape)),
    ("values", .sequence (encodeReals a.values))])

namespace Value

def readSequence : Value → Option Values
  | .sequence xs => some xs
  | _ => none

def readText : Value → Option String
  | .text s => some s
  | _ => none

def readReal : Value → Option ℝ
  | .number r => some r
  | _ => none

def readNat : Value → Option ℕ
  | .integer n => if 0 ≤ n then some n.toNat else none
  | _ => none

def decodeArray : Value → Option ArrayState
  | .object fs => do
    let shape ← (← fs.get "shape").readSequence
    let shape ← decodeNats shape
    let values ← (← fs.get "values").readSequence
    let values ← decodeReals values
    pure ⟨shape, values⟩
  | _ => none

end Value

open Value

theorem decode_encode_array (a : ArrayState) : decodeArray (encodeArray a) = some a := by
  cases a
  simp [encodeArray, decodeArray, Fields.ofList, Fields.get, readSequence,
    decode_encode_nats, decode_encode_reals]

structure ContinuousState where
  basis : ArrayState
  kernelMean : ArrayState
  kernelRowCovariance : ArrayState
  innovationCovariance : ArrayState
  phi : ℝ
  sfreq : ℝ
  channels : List String
  candidateIds : List String
  operatorIdentity : String
  provenance : Value

structure JointNoiseState where
  mean : ArrayState
  covariance : ArrayState
  rawDimension : ℕ
  provenance : Value

structure ArchiveView where
  arrays : Fields
  metadata : Fields

def continuousArrays (s : ContinuousState) : List ArrayState :=
  [s.basis, s.kernelMean, s.kernelRowCovariance, s.innovationCovariance]

def jointArrays (s : JointNoiseState) : List ArrayState := [s.mean, s.covariance]

abbrev ArrayHash := List ArrayState → String
abbrev MetadataHash := Fields → String

def continuousMetadata (arrayHash : ArrayHash) (s : ContinuousState) : Fields :=
  Fields.ofList [("schema", .text "n2p3_continuous_decision_model/2"),
    ("phi", .number s.phi), ("sfreq", .number s.sfreq),
    ("channels", .sequence (encodeTexts s.channels)),
    ("candidate_ids", .sequence (encodeTexts s.candidateIds)),
    ("operator_identity", .text s.operatorIdentity), ("provenance", copyValue s.provenance),
    ("state_sha256", .text (arrayHash (continuousArrays s)))]

def jointMetadata (arrayHash : ArrayHash) (s : JointNoiseState) : Fields :=
  Fields.ofList [("schema", .text "n2p3_joint_noise/1"),
    ("raw_dimension", .integer s.rawDimension), ("provenance", copyValue s.provenance),
    ("state_sha256", .text (arrayHash (jointArrays s)))]

def encodeContinuous (arrayHash : ArrayHash) (s : ContinuousState) : ArchiveView :=
  ⟨Fields.ofList [("basis", encodeArray s.basis), ("kernel_mean", encodeArray s.kernelMean),
    ("kernel_row_covariance", encodeArray s.kernelRowCovariance),
    ("innovation_covariance", encodeArray s.innovationCovariance)], continuousMetadata arrayHash s⟩

def encodeJoint (arrayHash : ArrayHash) (s : JointNoiseState) : ArchiveView :=
  ⟨Fields.ofList [("mean", encodeArray s.mean), ("covariance", encodeArray s.covariance)],
    jointMetadata arrayHash s⟩

def continuousArrayInputs (fs : Fields) : Option (ArrayState × ArrayState × ArrayState × ArrayState) := do
  let basis ← (← fs.get "basis").decodeArray
  let mean ← (← fs.get "kernel_mean").decodeArray
  let row ← (← fs.get "kernel_row_covariance").decodeArray
  let noise ← (← fs.get "innovation_covariance").decodeArray
  pure (basis, mean, row, noise)

def jointArrayInputs (fs : Fields) : Option (ArrayState × ArrayState) := do
  let mean ← (← fs.get "mean").decodeArray
  let covariance ← (← fs.get "covariance").decodeArray
  pure (mean, covariance)

def assembleContinuous (a : ArrayState × ArrayState × ArrayState × ArrayState)
    (m : Fields) : Option ContinuousState := do
  let phi ← (← m.get "phi").readReal
  let sfreq ← (← m.get "sfreq").readReal
  let channels ← decodeTexts (← (← m.get "channels").readSequence)
  let candidateIds ← decodeTexts (← (← m.get "candidate_ids").readSequence)
  let operatorIdentity ← (← m.get "operator_identity").readText
  let provenance ← m.get "provenance"
  pure ⟨a.1, a.2.1, a.2.2.1, a.2.2.2, phi, sfreq, channels, candidateIds, operatorIdentity, provenance⟩

def assembleJoint (a : ArrayState × ArrayState) (m : Fields) : Option JointNoiseState := do
  let raw ← (← m.get "raw_dimension").readNat
  let provenance ← m.get "provenance"
  pure ⟨a.1, a.2, raw, provenance⟩

def decodeContinuous (v : ArchiveView) : Option ContinuousState := do
  assembleContinuous (← continuousArrayInputs v.arrays) v.metadata

def decodeJoint (v : ArchiveView) : Option JointNoiseState := do
  assembleJoint (← jointArrayInputs v.arrays) v.metadata

theorem continuous_array_inputs_roundtrip (hash : ArrayHash) (s : ContinuousState) :
    continuousArrayInputs (encodeContinuous hash s).arrays =
      some (s.basis, s.kernelMean, s.kernelRowCovariance, s.innovationCovariance) := by
  simp [continuousArrayInputs, encodeContinuous, Fields.ofList, Fields.get, decode_encode_array]

theorem joint_array_inputs_roundtrip (hash : ArrayHash) (s : JointNoiseState) :
    jointArrayInputs (encodeJoint hash s).arrays = some (s.mean, s.covariance) := by
  simp [jointArrayInputs, encodeJoint, Fields.ofList, Fields.get, decode_encode_array]

/-- All fields, including full nested provenance, are independently decoded. -/
theorem continuous_codec_roundtrip (hash : ArrayHash) (s : ContinuousState) :
    decodeContinuous (encodeContinuous hash s) = some s := by
  rw [decodeContinuous, continuous_array_inputs_roundtrip]
  cases s
  simp [assembleContinuous, encodeContinuous, continuousMetadata, Fields.ofList,
    Fields.get, readReal, readText, readSequence, decode_encode_texts, copyValue_preserves]

theorem joint_codec_roundtrip (hash : ArrayHash) (s : JointNoiseState) :
    decodeJoint (encodeJoint hash s) = some s := by
  rw [decodeJoint, joint_array_inputs_roundtrip]
  cases s
  simp [assembleJoint, encodeJoint, jointMetadata, Fields.ofList, Fields.get,
    readNat, copyValue_preserves]

def provenanceIsObject : Value → Prop
  | .object _ => True
  | _ => False

def arrayLayout (a : ArrayState) : Prop := a.values.length = a.shape.foldr (· * ·) 1

def continuousCore (s : ContinuousState) : Prop :=
  (∃ width rank : ℕ, 0 < width ∧ 0 < rank ∧ s.basis.shape = [width, rank] ∧
    s.kernelMean.shape = [2 * rank, s.channels.length] ∧
    s.kernelRowCovariance.shape = [2 * rank, 2 * rank] ∧
    s.innovationCovariance.shape = [s.channels.length, s.channels.length]) ∧
  arrayLayout s.basis ∧ arrayLayout s.kernelMean ∧ arrayLayout s.kernelRowCovariance ∧
  arrayLayout s.innovationCovariance ∧ |s.phi| < 1 ∧ 0 < s.sfreq ∧
  s.channels ≠ [] ∧ s.channels.Nodup ∧ (∀ ch ∈ s.channels, ch ≠ "") ∧
  2 ≤ s.candidateIds.length ∧ s.candidateIds.Nodup ∧
  (∀ candidate ∈ s.candidateIds, candidate ≠ "") ∧ s.operatorIdentity ≠ "" ∧
  provenanceIsObject s.provenance

def jointCore (s : JointNoiseState) : Prop :=
  (∃ n : ℕ, s.mean.shape = [n] ∧ s.covariance.shape = [n, n] ∧
    0 < s.rawDimension ∧ s.rawDimension ≤ n) ∧
  arrayLayout s.mean ∧ arrayLayout s.covariance ∧ provenanceIsObject s.provenance

def covarianceEntry (a : ArrayState) (n i j : ℕ) : ℝ := a.values.getD (i * n + j) 0

def covarianceNormalForm (a : ArrayState) (n : ℕ) : Prop :=
  a.shape = [n, n] ∧ a.values.length = n * n ∧
    ∀ i < n, ∀ j < n, covarianceEntry a n i j = covarianceEntry a n j i

/-- The source's `(cov + cov.T)/2` element rule, over exact reals only. -/
def reconstructedCovarianceEntry (a : ArrayState) (n i j : ℕ) : ℝ :=
  (covarianceEntry a n i j + covarianceEntry a n j i) / 2

theorem normal_form_preserves_constructor_covariance_entry (a : ArrayState) (n i j : ℕ)
    (ha : covarianceNormalForm a n) (hi : i < n) (hj : j < n) :
    reconstructedCovarianceEntry a n i j = covarianceEntry a n i j := by
  unfold reconstructedCovarianceEntry
  rw [← ha.2.2 i hi j hj]
  ring

structure Sealed (S : Type*) where
  state : S
  integrity : String

def sealContinuous (ah : ArrayHash) (mh : MetadataHash) (s : ContinuousState) : Sealed ContinuousState :=
  ⟨s, mh (continuousMetadata ah s)⟩

def sealJoint (ah : ArrayHash) (mh : MetadataHash) (s : JointNoiseState) : Sealed JointNoiseState :=
  ⟨s, mh (jointMetadata ah s)⟩

def attachArtifactSeal (mh : MetadataHash) (v : ArchiveView) : ArchiveView :=
  ⟨v.arrays, .cons "artifact_sha256" (.text (mh v.metadata)) v.metadata⟩

def saveContinuous (ah : ArrayHash) (mh : MetadataHash) (s : Sealed ContinuousState) :
    Option ArchiveView :=
  if mh (continuousMetadata ah s.state) = s.integrity then
    some (attachArtifactSeal mh (encodeContinuous ah s.state)) else none

def saveJoint (ah : ArrayHash) (mh : MetadataHash) (s : Sealed JointNoiseState) : Option ArchiveView :=
  if mh (jointMetadata ah s.state) = s.integrity then
    some (attachArtifactSeal mh (encodeJoint ah s.state)) else none

inductive LoadFailure where
  | arrayDecode
  | schema
  | metadataDigest
  | stateDigest
  | metadataFields
  | semanticCore
  deriving DecidableEq

inductive LoadResult (S : Type*) where
  | rejected (reason : LoadFailure)
  | readyInput (state : S)

def loadContinuous (ah : ArrayHash) (mh : MetadataHash) (v : ArchiveView) : LoadResult ContinuousState := by
  classical
  exact match continuousArrayInputs v.arrays with
  | none => .rejected .arrayDecode
  | some a =>
    if v.metadata.get "schema" ≠ some (.text "n2p3_continuous_decision_model/2") then .rejected .schema
    else if v.metadata.get "artifact_sha256" ≠
        some (.text (mh (v.metadata.erase "artifact_sha256"))) then .rejected .metadataDigest
    else if v.metadata.get "state_sha256" ≠ some (.text (ah [a.1, a.2.1, a.2.2.1, a.2.2.2]))
      then .rejected .stateDigest
    else match assembleContinuous a v.metadata with
      | none => .rejected .metadataFields
      | some s => if continuousCore s then .readyInput s else .rejected .semanticCore

def loadJoint (ah : ArrayHash) (mh : MetadataHash) (v : ArchiveView) : LoadResult JointNoiseState := by
  classical
  exact match jointArrayInputs v.arrays with
  | none => .rejected .arrayDecode
  | some a =>
    if v.metadata.get "schema" ≠ some (.text "n2p3_joint_noise/1") then .rejected .schema
    else if v.metadata.get "artifact_sha256" ≠
        some (.text (mh (v.metadata.erase "artifact_sha256"))) then .rejected .metadataDigest
    else if v.metadata.get "state_sha256" ≠ some (.text (ah [a.1, a.2])) then .rejected .stateDigest
    else match assembleJoint a v.metadata with
      | none => .rejected .metadataFields
      | some s => if jointCore s then .readyInput s else .rejected .semanticCore

theorem save_continuous_mismatch_rejected (ah : ArrayHash) (mh : MetadataHash)
    (s : Sealed ContinuousState) (h : mh (continuousMetadata ah s.state) ≠ s.integrity) :
    saveContinuous ah mh s = none := by simp [saveContinuous, h]

theorem save_joint_mismatch_rejected (ah : ArrayHash) (mh : MetadataHash)
    (s : Sealed JointNoiseState) (h : mh (jointMetadata ah s.state) ≠ s.integrity) :
    saveJoint ah mh s = none := by simp [saveJoint, h]

theorem freshly_sealed_continuous_save (ah : ArrayHash) (mh : MetadataHash) (s : ContinuousState) :
    saveContinuous ah mh (sealContinuous ah mh s) =
      some (attachArtifactSeal mh (encodeContinuous ah s)) := by simp [saveContinuous, sealContinuous]

theorem freshly_sealed_joint_save (ah : ArrayHash) (mh : MetadataHash) (s : JointNoiseState) :
    saveJoint ah mh (sealJoint ah mh s) = some (attachArtifactSeal mh (encodeJoint ah s)) := by
  simp [saveJoint, sealJoint]

theorem erase_continuous_artifact_field (ah : ArrayHash) (s : ContinuousState) :
    (continuousMetadata ah s).erase "artifact_sha256" = continuousMetadata ah s := by
  simp [continuousMetadata, Fields.ofList, Fields.erase]

theorem erase_joint_artifact_field (ah : ArrayHash) (s : JointNoiseState) :
    (jointMetadata ah s).erase "artifact_sha256" = jointMetadata ah s := by
  simp [jointMetadata, Fields.ofList, Fields.erase]

theorem load_encoded_continuous (ah : ArrayHash) (mh : MetadataHash) (s : ContinuousState) :
    loadContinuous ah mh (attachArtifactSeal mh (encodeContinuous ah s)) =
      if continuousCore s then .readyInput s else .rejected .semanticCore := by
  classical
  cases s
  simp [loadContinuous, attachArtifactSeal, encodeContinuous, continuousMetadata,
    continuousArrayInputs, decode_encode_array, continuousArrays, Fields.get, Fields.erase,
    Fields.ofList, assembleContinuous, readReal, readSequence, readText,
    decode_encode_texts, copyValue_preserves]

theorem load_encoded_joint (ah : ArrayHash) (mh : MetadataHash) (s : JointNoiseState) :
    loadJoint ah mh (attachArtifactSeal mh (encodeJoint ah s)) =
      if jointCore s then .readyInput s else .rejected .semanticCore := by
  classical
  cases s
  simp [loadJoint, attachArtifactSeal, encodeJoint, jointMetadata, jointArrayInputs,
    decode_encode_array, jointArrays, Fields.get, Fields.erase, Fields.ofList,
    assembleJoint, readNat, copyValue_preserves]

theorem continuous_protocol_roundtrip (ah : ArrayHash) (mh : MetadataHash) (s : ContinuousState)
    (hcore : continuousCore s) :
    loadContinuous ah mh (attachArtifactSeal mh (encodeContinuous ah s)) = .readyInput s := by
  rw [load_encoded_continuous, if_pos hcore]

theorem joint_protocol_roundtrip (ah : ArrayHash) (mh : MetadataHash) (s : JointNoiseState)
    (hcore : jointCore s) :
    loadJoint ah mh (attachArtifactSeal mh (encodeJoint ah s)) = .readyInput s := by
  rw [load_encoded_joint, if_pos hcore]

theorem continuous_schema_first (ah : ArrayHash) (mh : MetadataHash) (v : ArchiveView)
    (a : ArrayState × ArrayState × ArrayState × ArrayState)
    (ha : continuousArrayInputs v.arrays = some a)
    (hs : v.metadata.get "schema" ≠ some (.text "n2p3_continuous_decision_model/2")) :
    loadContinuous ah mh v = .rejected .schema := by simp [loadContinuous, ha, hs]

theorem continuous_metadata_mismatch (ah : ArrayHash) (mh : MetadataHash) (v : ArchiveView)
    (a : ArrayState × ArrayState × ArrayState × ArrayState)
    (ha : continuousArrayInputs v.arrays = some a)
    (hs : v.metadata.get "schema" = some (.text "n2p3_continuous_decision_model/2"))
    (hm : v.metadata.get "artifact_sha256" ≠ some (.text (mh (v.metadata.erase "artifact_sha256")))) :
    loadContinuous ah mh v = .rejected .metadataDigest := by simp [loadContinuous, ha, hs, hm]

theorem continuous_state_mismatch (ah : ArrayHash) (mh : MetadataHash) (v : ArchiveView)
    (a : ArrayState × ArrayState × ArrayState × ArrayState)
    (ha : continuousArrayInputs v.arrays = some a)
    (hs : v.metadata.get "schema" = some (.text "n2p3_continuous_decision_model/2"))
    (hm : v.metadata.get "artifact_sha256" = some (.text (mh (v.metadata.erase "artifact_sha256"))))
    (hst : v.metadata.get "state_sha256" ≠ some (.text (ah [a.1, a.2.1, a.2.2.1, a.2.2.2]))) :
    loadContinuous ah mh v = .rejected .stateDigest := by simp [loadContinuous, ha, hs, hm, hst]

theorem joint_schema_first (ah : ArrayHash) (mh : MetadataHash) (v : ArchiveView)
    (a : ArrayState × ArrayState) (ha : jointArrayInputs v.arrays = some a)
    (hs : v.metadata.get "schema" ≠ some (.text "n2p3_joint_noise/1")) :
    loadJoint ah mh v = .rejected .schema := by simp [loadJoint, ha, hs]

theorem joint_metadata_mismatch (ah : ArrayHash) (mh : MetadataHash) (v : ArchiveView)
    (a : ArrayState × ArrayState) (ha : jointArrayInputs v.arrays = some a)
    (hs : v.metadata.get "schema" = some (.text "n2p3_joint_noise/1"))
    (hm : v.metadata.get "artifact_sha256" ≠ some (.text (mh (v.metadata.erase "artifact_sha256")))) :
    loadJoint ah mh v = .rejected .metadataDigest := by simp [loadJoint, ha, hs, hm]

theorem joint_state_mismatch (ah : ArrayHash) (mh : MetadataHash) (v : ArchiveView)
    (a : ArrayState × ArrayState) (ha : jointArrayInputs v.arrays = some a)
    (hs : v.metadata.get "schema" = some (.text "n2p3_joint_noise/1"))
    (hm : v.metadata.get "artifact_sha256" = some (.text (mh (v.metadata.erase "artifact_sha256"))))
    (hst : v.metadata.get "state_sha256" ≠ some (.text (ah [a.1, a.2]))) :
    loadJoint ah mh v = .rejected .stateDigest := by simp [loadJoint, ha, hs, hm, hst]

/-- Exact source AR innovation, before the separately modeled nuisance QR. -/
def arInnovation (s : ContinuousState) (previous current : ℝ) : ℝ := current - s.phi * previous

theorem decoded_continuous_ar_innovation (ah : ArrayHash) (s : ContinuousState) (x y : ℝ) :
    (decodeContinuous (encodeContinuous ah s)).map (fun t => arInnovation t x y) =
      some (arInnovation s x y) := by rw [continuous_codec_roundtrip]; rfl

def referenceProvenance : Value := .object (Fields.ofList
  [("fit_scope", .text "analytic_reference"), ("reference_id", .text "formal-codec-reference"),
    ("records", .sequence .nil)])

def referenceContinuous : ContinuousState where
  basis := ⟨[3, 1], [1, 1, 1]⟩
  kernelMean := ⟨[2, 1], [0, 0]⟩
  kernelRowCovariance := ⟨[2, 2], [1, 0, 0, 1]⟩
  innovationCovariance := ⟨[1, 1], [1]⟩
  phi := 0
  sfreq := 256
  channels := ["Cz"]
  candidateIds := ["candidate-a", "candidate-b"]
  operatorIdentity := "operator-a"
  provenance := referenceProvenance

def changedMetadata : ContinuousState :=
  { referenceContinuous with phi := 1 / 2, operatorIdentity := "operator-b" }

def invalidPhi : ContinuousState := { referenceContinuous with phi := 2 }

theorem reference_continuous_core : continuousCore referenceContinuous := by
  refine ⟨⟨3, 1, by norm_num [referenceContinuous]⟩, ?_⟩
  norm_num [continuousCore, referenceContinuous, arrayLayout, referenceProvenance, provenanceIsObject]
  decide

theorem changed_metadata_core : continuousCore changedMetadata := by
  refine ⟨⟨3, 1, by norm_num [changedMetadata, referenceContinuous]⟩, ?_⟩
  norm_num [continuousCore, changedMetadata, referenceContinuous, arrayLayout,
    referenceProvenance, provenanceIsObject]
  decide

theorem invalid_phi_fails_core : ¬continuousCore invalidPhi := by
  rintro ⟨_, _, _, _, _, hphi, _⟩
  norm_num [invalidPhi, referenceContinuous] at hphi

theorem invalid_phi_rejected_despite_matching_seals (ah : ArrayHash) (mh : MetadataHash) :
    loadContinuous ah mh (attachArtifactSeal mh (encodeContinuous ah invalidPhi)) =
      .rejected .semanticCore := by
  rw [load_encoded_continuous, if_neg invalid_phi_fails_core]

theorem changed_metadata_changes_state : referenceContinuous ≠ changedMetadata := by
  intro h
  have hp := congrArg ContinuousState.phi h
  norm_num [referenceContinuous, changedMetadata] at hp

/-- A seal over arrays alone omits two source metadata fields here, for EVERY
array hash function. This is not a SHA256 collision argument. -/
theorem arrays_only_omit_metadata (ah : ArrayHash) :
    ah (continuousArrays referenceContinuous) = ah (continuousArrays changedMetadata) ∧
      referenceContinuous ≠ changedMetadata := by
  exact ⟨rfl, changed_metadata_changes_state⟩

theorem changed_metadata_changes_ar_step :
    arInnovation referenceContinuous 1 0 ≠ arInnovation changedMetadata 1 0 := by
  norm_num [arInnovation, referenceContinuous, changedMetadata]

def constantMetadataHash : MetadataHash := fun _ => "constant-digest"

/-- Equal seals do not imply equal states when hashes are arbitrary. -/
theorem seal_collision_does_not_identify_state (ah : ArrayHash) :
    (sealContinuous ah constantMetadataHash referenceContinuous).integrity =
      (sealContinuous ah constantMetadataHash changedMetadata).integrity ∧
        referenceContinuous ≠ changedMetadata := by
  exact ⟨rfl, changed_metadata_changes_state⟩

/-- Each self-sealed record is accepted as its OWN constructor input. No
external trusted producer is authenticated by this checksum protocol. -/
theorem self_seals_do_not_authenticate_producer (ah : ArrayHash) (mh : MetadataHash) :
    loadContinuous ah mh (attachArtifactSeal mh (encodeContinuous ah referenceContinuous)) =
      .readyInput referenceContinuous ∧
    loadContinuous ah mh (attachArtifactSeal mh (encodeContinuous ah changedMetadata)) =
      .readyInput changedMetadata :=
  ⟨continuous_protocol_roundtrip ah mh _ reference_continuous_core,
    continuous_protocol_roundtrip ah mh _ changed_metadata_core⟩

def zeroDigest : String := "0000000000000000000000000000000000000000000000000000000000000000"

def referenceJoint : JointNoiseState where
  mean := ⟨[2], [0, 0]⟩
  covariance := ⟨[2, 2], [1, 0, 0, 1]⟩
  rawDimension := 1
  provenance := .object (Fields.ofList [
    ("fit_scope", .text "target_calibration"), ("scope_id", .text "formal-codec-reference"),
    ("draw_ids", .sequence (encodeTexts ["draw-0", "draw-1", "draw-2"])),
    ("n_draws", .integer 3), ("input_sha256", .text zeroDigest),
    ("shrinkage", .number 0), ("implementation_sha256", .text zeroDigest),
    ("estimator", .text "joint_standardized_covariance_with_cross_terms")])

theorem reference_joint_core : jointCore referenceJoint := by
  refine ⟨⟨2, by norm_num [referenceJoint]⟩, ?_⟩
  norm_num [jointCore, referenceJoint, arrayLayout, provenanceIsObject]

theorem reference_joint_ready_input (ah : ArrayHash) (mh : MetadataHash) :
    loadJoint ah mh (attachArtifactSeal mh (encodeJoint ah referenceJoint)) = .readyInput referenceJoint :=
  joint_protocol_roundtrip ah mh _ reference_joint_core

def slightlyAsymmetricCovariance : ArrayState := ⟨[2, 2], [1, 1 / 1000000000000, 0, 1]⟩

theorem arbitrary_covariance_normalization_can_change_entry :
    reconstructedCovarianceEntry slightlyAsymmetricCovariance 2 0 1 ≠
      covarianceEntry slightlyAsymmetricCovariance 2 0 1 := by
  norm_num [reconstructedCovarianceEntry, covarianceEntry, slightlyAsymmetricCovariance]

end

end N2P3.FrozenState
