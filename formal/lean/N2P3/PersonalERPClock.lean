import N2P3.PersonalERPBids

/-!
# Finalized recording clock and journal validation

Source: recording_clock.py:9-100. Contained path lookup, raw-byte IO, SHA,
JSON parsing and Decimal scalar arithmetic are separate named operations.
Counts, per-batch cursors, exact switches, segment coverage and containment
are expanded; no clock or recording validity oracle is supplied. P06 WIP.
-/

namespace N2P3.PersonalERPClock

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPExecution N2P3.PersonalERPRaw N2P3.PersonalERPModels

noncomputable section

attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Primitives where
  bids : PersonalERPBids.Primitives
  readBytes : String → Check (List UInt8)
  shaBytes : List UInt8 → String
  splitLines : List UInt8 → List (List UInt8)
  stripBytes : List UInt8 → List UInt8
  jsonBytes : List UInt8 → Check Value
  decimalFromFloat : ℝ → Check ℝ
  decimalMultiply : ℝ → ℝ → Check ℝ
  scalarAdd : Value → Value → Check Atom

def lengthJSON : Value → Check ℕ
  | .sequence values => .ok (valuesToList values).length
  | .object fields => .ok (fieldKeys fields).length
  | .text value => .ok value.toList.length
  | _ => .error (.fieldFailure "len recording clock counter")

def metadataPaths (p : Primitives) (root : String) (clock : Fields) : List String → Check (List (String × String))
  | [] => .ok []
  | key :: tail => do
    let value ← text (fieldDefault clock key) "clock path string"
    ensure (value ≠ "") "clock nonempty metadata journal paths"
    let path ← p.bids.path root value
    ensure (path.contained=true ∧ path.file=true) "clock contained existing metadata journal files"
    let rest ← metadataPaths p root clock tail
    pure ((key,path.resolved)::rest)

def withoutPaths (fields : Fields) : Fields := (fields.erase "metadata_path").erase "journal_path"

structure Counts where
  stored : ℤ
  acquired : ℤ
  serialized : ℤ
  padding : ℤ

def readCounts (metadata : Fields) : Check Counts := do
  let stored ← integer (fieldDefault metadata "stored_samples") "clock exact integer stored samples"
  let acquired ← integer (fieldDefault metadata "acquired_samples") "clock exact integer acquired samples"
  let serialized ← integer (fieldDefault metadata "edf_samples") "clock exact integer serialized samples"
  let padding ← integer (fieldDefault metadata "padding_samples") "clock exact integer padding samples"
  let voltage ← integer (fieldDefault metadata "voltage_samples") "clock exact integer voltage samples"
  ensure (0 < stored ∧ acquired=stored ∧ voltage=stored ∧ acquired ≤ serialized ∧ padding=serialized-acquired)
    "clock acquired voltage stored serialized padding counts"
  pure ⟨stored,acquired,serialized,padding⟩

def parseJournal (p : Primitives) : List (List UInt8) → Check (List Fields)
  | [] => .ok []
  | line :: tail => do
    let value ← p.jsonBytes line
    let row ← objectAt value "clock journal row mapping"
    let rest ← parseJournal p tail
    pure (row :: rest)

structure Batch where
  fields : Fields
  received : ℤ
  accepted : ℤ
  storedStart : ℤ

structure JournalState where
  received : ℤ := 0
  stored : ℤ := 0
  batches : List Batch := []

def checkJournal : List Fields → ℕ → JournalState → Check JournalState
  | [],_,state => .ok state
  | row :: tail,index,state => do
    ensure (PersonalAdmission.PyEq (fieldDefault row "batch_id") (.integer index) ∧
      PersonalAdmission.PyEq (fieldDefault row "received_start") (.integer state.received)) "clock journal batch cursor identity"
    let received ← integer (fieldDefault row "n_received") "clock exact integer received rows"
    let accepted ← integer (fieldDefault row "n_stored") "clock exact integer accepted rows"
    ensure (0 ≤ received ∧ (accepted=0 ∨ accepted=received)) "clock nonnegative all or none accepted rows"
    if accepted ≠ 0 then
      ensure (PersonalAdmission.PyEq (fieldDefault row "stored_start") (.integer state.stored) ∧
        fieldDefault row "sequence_valid"=.boolean true) "clock accepted batch cursor valid sequence"
    let seqLength ← lengthJSON ((row.get "seq_num").getD (.sequence .nil))
    ensure ((seqLength : ℤ)=received) "clock sequence counter length"
    let deltaLength ← lengthJSON ((row.get "delta_time_raw").getD (.sequence .nil))
    ensure ((deltaLength : ℤ)=received) "clock delta counter length"
    let batch : Batch := ⟨row,received,accepted,state.stored⟩
    checkJournal tail (index+1) ⟨state.received+received,state.stored+accepted,state.batches ++ [batch]⟩

structure Segment where
  fields : Fields
  start : ℤ
  stop : ℤ

def checkSegments : List Value → ℕ → ℤ → Check (List Segment × ℤ)
  | [],_,cursor => .ok ([],cursor)
  | value :: tail,index,cursor => do
    let row ← objectAt value "clock acquisition segment mapping"
    ensure (PersonalAdmission.PyEq (fieldDefault row "segment_id") (.integer index) ∧
      PersonalAdmission.PyEq (fieldDefault row "stored_start") (.integer cursor)) "clock contiguous segment index start"
    let stop ← integer (fieldDefault row "stored_stop") "clock exact integer segment stop"
    ensure (cursor ≤ stop) "clock nondecreasing segment stop"
    let rest ← checkSegments tail (index+1) stop
    pure (⟨row,cursor,stop⟩::rest.1,rest.2)

def matchingSegment (batch : Batch) : List Segment → Option Segment
  | [] => none
  | segment :: tail =>
    if PersonalAdmission.PyEq (fieldDefault segment.fields "segment_id") (fieldDefault batch.fields "segment_id") then
      some segment else matchingSegment batch tail

def checkBatchSegments (p : Primitives) (segments : List Segment) : List Batch → Check Unit
  | [] => .ok ()
  | batch :: tail => do
    if batch.accepted ≠ 0 then
      let segment ← match matchingSegment batch segments with
        | none => .error (.sourceReject "clock accepted batch has acquisition segment")
        | some segment => .ok segment
      -- Source accesses the stored_start value again; the earlier Python
      -- equality allows numeric bool/float aliases, so use numeric conversion.
      let start ← numeric (fieldDefault batch.fields "stored_start") "clock comparable stored batch start"
      let stop ← p.scalarAdd (fieldDefault batch.fields "stored_start") (.integer batch.accepted)
      ensure ((segment.start : ℝ) ≤ start ∧ numberLE stop (.integer segment.stop)) "clock stored batch inside acquisition segment"
    checkBatchSegments p segments tail

structure Result where
  metadata : Fields
  counts : Option Counts
  journal : List Batch
  segments : List Segment

def validate (p : Primitives) (root : String) (recording : Fields) (rate duration : ℝ) : Check Result := do
  let clockValue := fieldDefault recording "clock"
  if clockValue=.null then
    pure ⟨Fields.ofList [("alignment_level",.text "legacy_host_only"),("absolute_offset_bound_seconds",.null)],none,[],[]⟩
  else
    let clock ← objectAt clockValue "recording clock mapping"
    ensure (fieldDefault clock "schema"=.text "brainsync-sample-clock/1") "recording clock schema one"
    let paths ← metadataPaths p root clock ["metadata_path","journal_path"]
    let metadataPath ← PersonalERPBids.getPath paths "metadata_path"
    let metadata ← PersonalERPBids.readObject p.bids metadataPath
    ensure (PersonalAdmission.PyEq (.object (withoutPaths clock)) (.object (withoutPaths metadata)))
      "clock finalized sidecar matches manifest fields"
    ensure (fieldDefault metadata "finalized"=.boolean true ∧ fieldDefault metadata "integrity_failed"=.boolean false ∧
      PersonalAdmission.PyEq (fieldDefault metadata "sample_rate_hz") (.number rate) ∧
      fieldDefault metadata "alignment_level"=.text "delivery_anchored" ∧
      fieldDefault metadata "absolute_offset_bound_seconds"=.null) "clock finalized integrity rate alignment declaration"
    let counts ← readCounts metadata
    let decimalDuration ← p.decimalFromFloat duration
    let decimalRate ← p.decimalFromFloat rate
    ensure (0 < decimalDuration ∧ 0 < decimalRate) "clock positive decimal duration rate"
    let product ← p.decimalMultiply decimalDuration decimalRate
    ensure (product=counts.serialized) "clock exact Decimal serialized sample product"
    let journalPath ← PersonalERPBids.getPath paths "journal_path"
    let contents ← p.readBytes journalPath
    ensure (fieldDefault metadata "journal_sha256"=.text (p.shaBytes contents)) "clock actual journal byte checksum"
    let rows ← parseJournal p ((p.splitLines contents).filter (fun line => !(p.stripBytes line).isEmpty))
    let state ← checkJournal rows 0 {}
    ensure (state.stored=counts.stored ∧
      PersonalAdmission.PyEq (.integer state.received) (fieldDefault metadata "received_samples") ∧
      PersonalAdmission.PyEq (.integer rows.length) (fieldDefault metadata "batch_count")) "clock finalized journal totals"
    let segmentValues ← tupleValue ((metadata.get "segments").getD (.sequence .nil))
    let (segments,cursor) ← checkSegments segmentValues 0 0
    ensure (cursor=counts.stored) "clock segments cover all stored samples"
    checkBatchSegments p segments state.batches
    pure ⟨metadata,some counts,state.batches,segments⟩

theorem absent_clock_is_explicit_legacy (p : Primitives) (root : String) (recording : Fields) (rate duration : ℝ)
    (h : recording.get "clock"=none) :
    validate p root recording rate duration=.ok ⟨Fields.ofList
      [("alignment_level",.text "legacy_host_only"),("absolute_offset_bound_seconds",.null)],none,[],[]⟩ := by
  simp [validate,fieldDefault,h]

theorem partial_batch_is_not_an_accepted_clock_batch (received accepted : ℤ)
    (hzero : accepted ≠ 0) (hfull : accepted ≠ received) :
    ensure (0 ≤ received ∧ (accepted=0 ∨ accepted=received)) "clock nonnegative all or none accepted rows"=
      .error (.sourceReject "clock nonnegative all or none accepted rows") := by simp [ensure,hzero,hfull]

end

end N2P3.PersonalERPClock
