import N2P3.ERPTemporalLineage

/-!
# Actual NPZ-to-raw EpochDataset construction

Source: epochs.py:1436-1513. The result retains every independently decoded
raw array, including arbitrary shape and nonfinite cells. No full validation
is inserted, and no primitive returns an EpochDataset or admitted scope.
Identity and lineage readers are expanded imported field programs. The later
attested-record comparison, returned identity/physical gates, conversion into
the typed Input representation and CLI prep remain separate pending stages.
-/

namespace N2P3.ERPTemporalCacheDecode

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.ERPTemporalCache N2P3.ERPTemporalSnapshot N2P3.ERPTemporalNPZ

noncomputable section

structure Operations (World : Type) where
  npz : ERPTemporalNPZ.Operations World
  identity : PersonalERPInput.Primitives
  lineage : ERPTemporalLineage.Primitives

structure Quality where
  relativePTP : {image : RawArray // image.kind=.floating}
  standardDeviation : {image : RawArray // image.kind=.floating}
  epochScale : {image : RawArray // image.kind=.floating}
  observed : {image : RawArray // image.kind=.boolean}
  schema : String

structure Dataset where
  name : String
  X : {image : RawArray // image.kind=.floating}
  labels : Option {image : RawArray // image.kind=.signed}
  subjects : {image : RawArray // image.kind=.unicode}
  channelNames : List String
  positions : {image : RawArray // image.kind=.floating}
  staticMask : {image : RawArray // image.kind=.boolean}
  preprocessing : Fields
  timeline : ERPTemporalNPZ.Timeline
  metadata : TableImage
  provenance : Value
  trialMask : Option {image : RawArray // image.kind=.boolean}
  quality : Quality
  identity : List PersonalERPInput.IdentityRecord
  lineage : ERPTemporalLineage.Lineage
  attestation : Option Fields := none

def readQuality {World : Type} (p : Operations World) (archive : ℕ) : TraceM World Quality := do
  let relative ← readCast p.npz archive "qc_relative_ptp" .float32
  let standardDeviation ← readCast p.npz archive "qc_channel_std_v" .float32
  let epochScale ← readCast p.npz archive "qc_epoch_scale_v" .float32
  let observed ← readCast p.npz archive "qc_observed_mask" .boolean
  let schema ← readString p.npz archive "qc_feature_schema"
  pure ⟨relative,standardDeviation,epochScale,observed,schema⟩

def pythonStrings {World : Type} (p : ERPTemporalNPZ.Operations World) :
    List NativeValue → TraceM World (List String)
  | [] => pure []
  | value :: tail => do
    let string ← small (p.pythonString value)
    let rest ← pythonStrings p tail
    pure (string :: rest)

def readChannelNames {World : Type} (p : ERPTemporalNPZ.Operations World) (archive : ℕ) :
    TraceM World (List String) := do
  let array ← small (p.readMember archive "channel_names")
  pythonStrings p (← small (p.iterateFirstAxis array))

def readTrialMask {World : Type} (p : ERPTemporalNPZ.Operations World) (archive : ℕ)
    (members : List String) : TraceM World (Option {image : RawArray // image.kind=.boolean}) := do
  if "trial_channel_mask" ∈ members then
    let array ← readArray p archive "trial_channel_mask"
    if 0 < array.cells.length then some <$> readCast p archive "trial_channel_mask" .boolean
    else pure none
  else pure none

def constructDataset {World : Type} (p : Operations World) (archive : ℕ) (preparation : Prefix) :
    TraceM World Dataset := do
  let quality ← readQuality p archive
  let name ← readString p.npz archive "name"
  let X ← small (p.npz.cast .float32 preparation.retainedX)
  let labels ← if preparation.hasLabels then some <$> readCast p.npz archive "y" .int64 else pure none
  let subjects ← readCast p.npz archive "subject_ids" .string
  let channels ← readChannelNames p.npz archive
  let positions ← readCast p.npz archive "channel_positions_m" .float32
  let mask ← readCast p.npz archive "channel_mask" .boolean
  let timeline ← constructTimeline p.npz (← readTimelineKeywords p.npz archive)
  let metadata ← small (p.npz.readTableJSON preparation.metadataJSON)
  let provenance ← readJSON p.npz archive "provenance_json"
  let trial ← readTrialMask p.npz archive preparation.members
  let identity ← checked (PersonalERPInput.readIdentityPayload p.identity
    (← readJSON p.npz archive "identity_json"))
  let lineage ← checked (ERPTemporalLineage.readLineage p.lineage
    (← readJSON p.npz archive "lineage_json"))
  pure ⟨name,⟨X.1,X.2.1⟩,labels,subjects,channels,positions,mask,preparation.preprocessing,timeline,metadata,
    provenance,trial,quality,identity,lineage,none⟩

def decodeArchive {World : Type} (p : Operations World) (archive : ℕ) : TraceM World (Dataset × Prefix) := do
  let preparation ← readPrefix p.npz archive
  let dataset ← constructDataset p archive preparation
  pure (dataset,preparation)

end
end N2P3.ERPTemporalCacheDecode
