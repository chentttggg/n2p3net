import N2P3.ERPTemporalCacheRecord

/-!
# Attested comparison, stored/fresh decoded digest and loader return gates

Source: epochs.py:1287-1305,1515-1555,817-856. This computes the expected
record from raw arrays, performs JSON dump/load, normalizes only the stated
legacy sidecar case, then compares every expected field using Python JSON
equality. It does not call full dataset.validate or assume its success.

The NPZ stored decoded hash bypasses ALL current content-invariant hashing.
When absent, every current array/table/identity/fingerprint digest is computed
in literal order. Required labels, identity exclusions and physical-coordinate
return gates follow attestation construction. Raw wrong-length/rank datasets
remain raw; no EpochDataset N/C/T view is fabricated here.
-/

namespace N2P3.ERPTemporalAttestedLoad

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open N2P3.ERPTemporalCache N2P3.ERPTemporalSnapshot N2P3.ERPTemporalNPZ
open N2P3.ERPTemporalJSON N2P3.ERPTemporalCacheRecord

noncomputable section

attribute [local instance] Classical.propDecidable

structure Operations (World : Type) where
  record : ERPTemporalCacheRecord.Operations World
  tableToJSON : TableImage → EStateM Fault World String

def fieldList : JFields → List (String × JValue)
  | .nil => []
  | .cons key value tail => (key,value) :: fieldList tail

def setField (fields : JFields) (key : String) (value : JValue) : JFields :=
  .cons key value (ERPTemporalRawFields.erase fields key)

def normalizeLegacySidecar (record : JFields) : JFields :=
  match record.get "preprocessing" with
  | some (.object preprocessing) =>
    if preprocessing.get "causal_iir_initial_state"=none ∧
        pythonEqual (ERPTemporalRawFields.fieldDefault preprocessing "filter_phase") (.text "zero") then
      setField record "preprocessing" (.object (setField preprocessing "causal_iir_initial_state" (.text "not_applicable")))
    else record
  | _ => record

def mismatchedFields (stored expected : JFields) : List String :=
  ((fieldList expected).filter (fun item =>
    !decide (pythonEqual (ERPTemporalRawFields.fieldDefault stored item.1) item.2))).map Prod.fst

def checkAttestedRecord {World : Type} (p : Operations World) (stored : JFields)
    (dataset : ERPTemporalRawCacheDecode.Dataset) : TraceM World JFields := do
  let expected ← roundtripRecord p.record dataset
  let expected ← checked (ERPTemporalRawFields.objectAt expected "expected record.items")
  let stored := normalizeLegacySidecar stored
  checked (ensure (mismatchedFields stored expected=[]) "attested sidecar fields differ from decoded record")
  pure stored

def semanticHash {World : Type} (p : Operations World) (value : JValue) : TraceM World String := do
  sourceHash p.record.hash (← small (p.record.hash.strictSemanticASCII value))

def contentInvariants {World : Type} (p : Operations World)
    (dataset : ERPTemporalRawCacheDecode.Dataset) : TraceM World JValue := do
  checkStoredIdentityCoverage p.record dataset
  let X ← arrayHash p.record dataset.X.1
  let labels ← match dataset.labels with
    | none => pure JValue.null
    | some labels => JValue.text <$> arrayHash p.record labels.1
  let subjects ← small (p.record.decode.npz.asArray dataset.subjects.1)
  let subjects ← small (p.record.astype .string subjects.1)
  let subjectHash ← arrayHash p.record subjects.1
  let positions ← arrayHash p.record dataset.positions.1
  let staticMask ← arrayHash p.record dataset.staticMask.1
  let trialMask ← match dataset.trialMask with
    | none => pure JValue.null
    | some mask => JValue.text <$> arrayHash p.record mask.1
  let fingerprint ← timelineFingerprint p.record dataset.timeline
  let metadataText ← small (p.tableToJSON dataset.metadata)
  let metadata ← small (p.record.decode.decodeJSON metadataText)
  let metadataHash ← semanticHash p metadata
  let identity ← checked (p.record.decode.input.execution.semanticHash
    (.object (PersonalERPInput.identityCanonicalPayload dataset.identity)))
  let relative ← arrayHash p.record dataset.quality.relativePTP.1
  let standardDeviation ← arrayHash p.record dataset.quality.standardDeviation.1
  let scale ← arrayHash p.record dataset.quality.epochScale.1
  let observed ← arrayHash p.record dataset.quality.observed.1
  let quality := fields [("schema",.text dataset.quality.schema),("relative_ptp_sha256",.text relative),
    ("channel_std_v_sha256",.text standardDeviation),("epoch_scale_v_sha256",.text scale),
    ("observed_mask_sha256",.text observed)]
  pure (fields [("X_sha256",.text X),("y_sha256",labels),("subject_ids_sha256",.text subjectHash),
    ("channel_names",texts dataset.channelNames),("channel_positions_sha256",.text positions),
    ("channel_mask_sha256",.text staticMask),("trial_channel_mask_sha256",trialMask),
    ("event_timeline_fingerprint",.text fingerprint),("metadata_sha256",.text metadataHash),
    ("identity_digest",.text identity),("qc_features",quality)])

def decodedDatasetHash {World : Type} (p : Operations World)
    (dataset : ERPTemporalRawCacheDecode.Dataset) : TraceM World String := do
  let record ← record p.record dataset
  let invariants ← contentInvariants p dataset
  semanticHash p (fields [("record",record),("content_invariants",invariants)])

def chooseDecodedHash {World : Type} (p : Operations World)
    (dataset : ERPTemporalRawCacheDecode.Dataset) (stored : Option String) : TraceM World String :=
  match stored with
  | some digest => pure digest
  | none => decodedDatasetHash p dataset

def verifiedAttestation (attestation : Fields) (decoded : String) : Fields :=
  Fields.ofList [("schema",.text "n2p3_verified_epoch_cache_load/1"),
    ("cache_attestation_schema",N2P3.StateLedger.fieldDefault attestation "schema"),
    ("full_contract_validated",.boolean true),("sha256",N2P3.StateLedger.fieldDefault attestation "sha256"),
    ("byte_size",N2P3.StateLedger.fieldDefault attestation "byte_size"),("decoded_dataset_sha256",.text decoded)]

def physicalReturnGate (dataset : ERPTemporalRawCacheDecode.Dataset) : Check Unit :=
  match dataset.provenance with
  | .object fields => ensure (ERPTemporalRawFields.fieldDefault fields "data_alignment"=.null)
      "physical loader refuses data alignment coordinates"
  | _ => .ok ()

/-- The physical attested branch only. expectedPreprocessing, when supplied,
is the complete field image of an already constructed PreprocessingSpec,
not an unparsed partial JSON kwargs object. The CLI explicitly requests labels;
the underlying public epoch loader's default is false. -/
def load {World : Type} (p : Operations World) (path : String) (requireLabels : Bool := false)
    (expectedPreprocessing : Option JFields := none) : TraceM World ERPTemporalRawCacheDecode.Dataset := do
  let decoded ← ERPTemporalRawCacheDecode.acquireAndDecode p.record.decode path
  let dataset := decoded.1
  let preparation := decoded.2.1
  let snapshot := decoded.2.2
  let _ ← checkAttestedRecord p snapshot.record dataset
  let digest ← chooseDecodedHash p dataset preparation.arrays.storedDecodedDigest
  let dataset := {dataset with attestation := some (verifiedAttestation snapshot.attestation digest)}
  checked (ensure (requireLabels=false ∨ dataset.labels.isSome) "operation requires labels")
  match expectedPreprocessing with
  | none => pure ()
  | some expected => checked (ensure (pythonEqual (.object dataset.preprocessing) (.object expected))
      "loader preprocessing contract differs")
  checked (PersonalERPInput.checkIdentityExclusions p.record.decode.input dataset.identity)
  checked (physicalReturnGate dataset)
  pure dataset

end
end N2P3.ERPTemporalAttestedLoad
