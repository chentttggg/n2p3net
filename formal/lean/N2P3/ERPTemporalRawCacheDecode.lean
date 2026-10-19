import N2P3.ERPTemporalCacheDecode
import N2P3.ERPTemporalRawSnapshot
import N2P3.ERPTemporalRawIdentity
import N2P3.ERPTemporalRawLineage

/-! Full raw JSON arguments on the NPZ construction path. This extends the
separate finite-JSON component without projecting unknown/nonfinite spec or
provenance fields. Raw arrays keep independent shapes. No N/C/T Context is
built here, so malformed datasets can survive into record/CLI/actual guards.
-/

namespace N2P3.ERPTemporalRawCacheDecode

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open N2P3.ERPTemporalCache N2P3.ERPTemporalSnapshot N2P3.ERPTemporalNPZ
open N2P3.ERPTemporalJSON

noncomputable section

structure Operations (World : Type) where
  npz : ERPTemporalNPZ.Operations World
  input : PersonalERPInput.Primitives
  decodeJSON : String → EStateM Fault World JValue
  pythonString : JValue → Check String
  lineageLower : String → String
  lineageHash : N2P3.FrozenState.Value → Check String

def snapshotOperations {World : Type} (p : Operations World) : ERPTemporalRawSnapshot.Operations World :=
  ⟨p.npz.snapshot,p.decodeJSON⟩

def identityOperations {World : Type} (p : Operations World) : ERPTemporalRawIdentity.Primitives :=
  ⟨p.input,p.pythonString⟩

def lineageOperations {World : Type} (p : Operations World) : ERPTemporalRawLineage.Primitives :=
  ⟨p.pythonString,p.input.execution.strip,p.lineageLower,p.lineageHash⟩

def readJSON {World : Type} (p : Operations World) (archive : ℕ) (name : String) : TraceM World JValue := do
  small (p.decodeJSON (← readString p.npz archive name))

structure Prefix where
  arrays : ArrayPrefix
  preprocessing : JFields
  hasLabels : Bool
  metadataJSON : String

def readPrefix {World : Type} (p : Operations World) (archive : ℕ) : TraceM World Prefix := do
  let arrays ← readArrayPrefix p.npz archive
  let preprocessing ← checked (ERPTemporalRawFields.preprocessingArgs
    (← readJSON p archive "preprocessing_json"))
  let hasLabels ← readBool p.npz archive "has_labels"
  let metadata ← metadataText p.npz (← readArray p.npz archive "metadata_json")
  pure ⟨arrays,preprocessing,hasLabels,metadata⟩

structure Dataset where
  name : String
  X : {image : RawArray // image.kind=.floating}
  labels : Option {image : RawArray // image.kind=.signed}
  subjects : {image : RawArray // image.kind=.unicode}
  channelNames : List String
  positions : {image : RawArray // image.kind=.floating}
  staticMask : {image : RawArray // image.kind=.boolean}
  preprocessing : JFields
  timeline : ERPTemporalNPZ.Timeline
  metadata : TableImage
  provenance : JValue
  trialMask : Option {image : RawArray // image.kind=.boolean}
  quality : ERPTemporalCacheDecode.Quality
  identity : List PersonalERPInput.IdentityRecord
  lineage : ERPTemporalLineage.Lineage
  attestation : Option Fields := none

def readQuality {World : Type} (p : Operations World) (archive : ℕ) : TraceM World ERPTemporalCacheDecode.Quality := do
  let relative ← readCast p.npz archive "qc_relative_ptp" .float32
  let standardDeviation ← readCast p.npz archive "qc_channel_std_v" .float32
  let scale ← readCast p.npz archive "qc_epoch_scale_v" .float32
  let observed ← readCast p.npz archive "qc_observed_mask" .boolean
  let schema ← readString p.npz archive "qc_feature_schema"
  pure ⟨relative,standardDeviation,scale,observed,schema⟩

def constructDataset {World : Type} (p : Operations World) (archive : ℕ) (preparation : Prefix) :
    TraceM World Dataset := do
  let quality ← readQuality p archive
  let name ← readString p.npz archive "name"
  let X ← small (p.npz.cast .float32 preparation.arrays.retainedX)
  let labels ← if preparation.hasLabels then some <$> readCast p.npz archive "y" .int64 else pure none
  let subjects ← readCast p.npz archive "subject_ids" .string
  let channels ← ERPTemporalCacheDecode.readChannelNames p.npz archive
  let positions ← readCast p.npz archive "channel_positions_m" .float32
  let mask ← readCast p.npz archive "channel_mask" .boolean
  let timeline ← constructTimeline p.npz (← readTimelineKeywords p.npz archive)
  let metadata ← small (p.npz.readTableJSON preparation.metadataJSON)
  let provenance ← readJSON p archive "provenance_json"
  let trial ← ERPTemporalCacheDecode.readTrialMask p.npz archive preparation.arrays.members
  let identity ← checked (ERPTemporalRawIdentity.readTable (identityOperations p)
    (← readJSON p archive "identity_json"))
  let lineage ← checked (ERPTemporalRawLineage.readLineage (lineageOperations p)
    (← readJSON p archive "lineage_json"))
  pure ⟨name,⟨X.1,X.2.1⟩,labels,subjects,channels,positions,mask,preparation.preprocessing,timeline,metadata,
    provenance,trial,quality,identity,lineage,none⟩

def decodeArchive {World : Type} (p : Operations World) (archive : ℕ) : TraceM World (Dataset × Prefix) := do
  let preparation ← readPrefix p archive
  let dataset ← constructDataset p archive preparation
  pure (dataset,preparation)

/-- Both contexts close before attested record comparison. If np.load entry
fails, the already-entered temporary snapshot context still closes. -/
def acquireAndDecode {World : Type} (p : Operations World) (path : String) :
    TraceM World (Dataset × Prefix × SnapshotPacket JFields) := do
  let snapshot ← ERPTemporalRawSnapshot.prepareSnapshot (snapshotOperations p) path
  closeFinally (p.npz.snapshot.closeSnapshot snapshot.handle) do
    let archive ← small (p.npz.openNPZNoPickle snapshot.handle)
    closeFinally (p.npz.closeNPZ archive) do
      let decoded ← decodeArchive p archive
      pure (decoded.1,decoded.2,snapshot)

end
end N2P3.ERPTemporalRawCacheDecode
