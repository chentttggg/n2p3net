import N2P3.PersonalERPRawOperators

/-!
# Native continuous QC and dependency-ledger projection

Source: continuous_qc.py:187-243,354-482,484-511. Threshold comparison,
interval clustering, exact-constant runs, projected parent dependencies,
merging, acquisition boundaries and epoch-state waits are sequenced explicitly.
NumPy/SciPy array expressions and the separately named state-bound numerical
solvers remain unrefined operation boundaries; their physical validity and
floating implementation are not certified by these control-flow definitions.
No whole QC detector, raw builder or validator is an input primitive. P06 WIP.
-/

namespace N2P3.PersonalERPContinuous

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPExecution N2P3.PersonalERPRaw N2P3.PersonalERPModels
open N2P3.PersonalERPRawOperators

noncomputable section

attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Interval where
  start : ℤ
  stop : ℤ
  detected : ℤ
  confirmed : ℤ
  channels : List String
  reason : String
  reset : Bool

structure Result where
  samples : ℕ
  rate : ℝ
  intervals : List Interval
  modelDigest : String
  channels : List String
  localMode : Bool
  lookahead : ℤ

structure Primitives where
  raw : PersonalERPRawOperators.Primitives
  segments : Raw → Check (List PersonalERPRawOperators.Span)
  slice : ArrayImage Atom → PersonalERPRawOperators.Span → Check (ArrayImage Atom)
  difference : ArrayImage Atom → Check (ArrayImage Atom)
  absolute : ArrayImage Atom → Check (ArrayImage Atom)
  greater : ArrayImage Atom → ArrayImage Atom → Check (ArrayImage Bool)
  padLeft : ArrayImage Bool → Check (ArrayImage Bool)
  maximumFilter : ArrayImage Atom → ℤ → ℤ → String → Check (ArrayImage Atom)
  minimumFilter : ArrayImage Atom → ℤ → ℤ → String → Check (ArrayImage Atom)
  clearBefore : ArrayImage Bool → ℤ → Check (ArrayImage Bool)
  dilate : ArrayImage Bool → ℤ → ℤ → Check (ArrayImage Bool)
  unionFlags : ArrayImage Bool → ArrayImage Bool → Check (ArrayImage Bool)
  rowFlags : ArrayImage Bool → ℕ → Check (List Bool)
  rowValues : ArrayImage Atom → ℕ → Check (List Atom)
  equalSamples : Atom → Atom → Check Bool
  arrayRow : ArrayImage Atom → ℕ → Check (List Atom)
  stateBoxWait : StateBox → Check (Option ℕ)
  epochWaits : EpochRecovery → Check (List ℕ)

def flagAt (image : ArrayImage Bool) (row column : ℕ) : Check Bool := do
  let width ← match image.shape with
    | [_,width] => .ok width
    | _ => .error (.fieldFailure "continuous flag matrix geometry")
  match image.values[row*width+column]? with
  | none => .error (.fieldFailure "continuous flag matrix index")
  | some value => .ok value

def flaggedIndices (values : List Bool) : List ℤ :=
  ((List.range values.length).filter (fun i => values[i]?.getD false)).map Int.ofNat

def gatherCluster (settle : ℤ) : List ℤ → ℤ → List ℤ × List ℤ × ℤ
  | [],last => ([],[],last)
  | value :: tail,last =>
    if value ≤ last+settle then
      let (members,remaining,last) := gatherCluster settle tail value
      (value::members,remaining,last)
    else ([],value::tail,last)

def clustersFuel (endSample settle lookahead : ℤ) (names : List String) (reason : String) :
    ℕ → List ℤ → Check (List Interval)
  | _,[] => .ok []
  | 0,_::_ => .error (.fieldFailure "cluster recursion consumes each index")
  | fuel+1,first :: tail => do
    let (_,remaining,last) := gatherCluster settle tail first
    let stop := min endSample (last+settle+1)
    let row : Interval := ⟨first,stop,min (endSample-1) (first+lookahead),
      min (endSample-1) (stop-1+lookahead),names,reason,true⟩
    let rest ← clustersFuel endSample settle lookahead names reason fuel remaining
    pure (row::rest)

def cluster (indices : List ℤ) (endSample settle : ℤ) (names : List String) (reason : String) (lookahead : ℤ) :
    Check (List Interval) := clustersFuel endSample settle lookahead names reason indices.length indices

def runBoundaries (p : Primitives) : List Atom → ℕ → Check (List ℤ)
  | [],_ => .ok []
  | [_],_ => .ok []
  | value :: next :: tail,index => do
    let same ← p.equalSamples value next
    let rest ← runBoundaries p (next::tail) (index+1)
    pure (if same then rest else ((index : ℤ)+1) :: rest)

def exactConstantIntervals (p : Primitives) (values : List Atom) (offset minimum : ℤ) (name : String) :
    Check (List Interval) := do
  let cuts ← runBoundaries p values 0
  let boundaries := [0] ++ cuts ++ [(values.length : ℤ)]
  let pairs := boundaries.zip (boundaries.drop 1)
  pure ((pairs.filter (fun pair => decide (minimum ≤ pair.2-pair.1))).map fun (start,stop) =>
    ⟨offset+start,offset+stop,offset+start+minimum-1,offset+min stop ((values.length : ℤ)-1),
      [name],"raw_exact_constant",true⟩)

def trailingRange (p : Primitives) (values : ArrayImage Atom) (width : ℤ) : Check (ArrayImage Atom) := do
  let high ← p.maximumFilter values width ((width-1)/2) "nearest"
  let low ← p.minimumFilter values width ((width-1)/2) "nearest"
  p.raw.subtractArrays high low

def localChannelIntervals (p : Primitives) (model : ContinuousQC) (segment : PersonalERPRawOperators.Span)
    (values : ArrayImage Atom) (flags : ArrayImage Bool) (settle lookahead : ℤ) :
    List (ℕ × String) → Check (List Interval)
  | [] => .ok []
  | (channel,name) :: tail => do
    let rowFlags ← p.rowFlags flags channel
    let indices := (flaggedIndices rowFlags).map (fun index => index+(segment.start : ℤ))
    let reason := if model.spec.rangeWindow.isSome ∧ fieldDefault model.spec.normalized "range_action"=.text "reject" then
      "raw_difference_or_range_outlier" else "raw_difference_outlier"
    let detected ← cluster indices segment.stop settle [name] reason lookahead
    let guarded := detected.map (fun row => {row with start:=max segment.start (row.start-model.spec.preGuard)})
    let constants ← match model.spec.flatlineDuration with
      | none => .ok []
      | some duration => do
        let minimum ← p.raw.scope.modal.ceilProduct duration model.rate
        let rowValues ← p.rowValues values channel
        exactConstantIntervals p rowValues segment.start (minimum+1) name
    let rest ← localChannelIntervals p model segment values flags settle lookahead tail
    pure (guarded ++ constants ++ rest)

def globalFlagIndices (bad : ArrayImage Bool) (channels width : ℕ) : Check (List ℤ) := do
  let columns ← (List.range width).mapM fun column => do
    let flags ← (List.range channels).mapM (fun channel => flagAt bad channel column)
    pure (flags.any id)
  pure (flaggedIndices columns)

def affectedNames (bad : ArrayImage Bool) (names : List String) (members : List ℤ) (start : ℕ) : Check (List String) := do
  let selected ← ((List.range names.length).zip names).mapM fun (channel,name) => do
    let flags ← members.mapM (fun sample => flagAt bad channel (sample-(start : ℤ)-1).toNat)
    pure (name,flags.any id)
  pure ((selected.filter Prod.snd).map Prod.fst)

def globalIntervalsFuel (bad : ArrayImage Bool) (names : List String) (start stop settle : ℤ)
    (wait : Option ℕ) : ℕ → List ℤ → Check (List Interval)
  | _,[] => .ok []
  | 0,_::_ => .error (.fieldFailure "global cluster recursion consumes each index")
  | fuel+1,first :: tail => do
    let (members,remaining,last) := gatherCluster settle tail first
    let affected ← affectedNames bad names (first::members) start.toNat
    let confirmed := min (last+settle) (stop-1)
    let endSample := confirmed+1
    let event : Interval := ⟨first,endSample,first,confirmed,affected,"raw_difference_outlier",true⟩
    let recoveryEnd := match wait with | none => stop | some wait => min stop (endSample+wait)
    let recovery := if endSample < stop ∧ endSample < recoveryEnd then
      [Interval.mk endSample recoveryEnd first confirmed affected
        (if wait=none then "filter_state_uncertainty_unbounded" else "filter_state_bounded_recovery") false] else []
    let rest ← globalIntervalsFuel bad names start stop settle wait fuel remaining
    pure ([event] ++ recovery ++ rest)

def detectSegments (p : Primitives) (model : ContinuousQC) (values center limit : ArrayImage Atom)
    (names : List String) (settle : ℤ) (wait : Option ℕ) : List PersonalERPRawOperators.Span → Check (List Interval)
  | [] => .ok []
  | segment :: tail => do
    let segmentValues ← p.slice values segment
    let differences ← p.difference segmentValues
    let centered ← p.raw.subtractArrays differences center
    let magnitude ← p.absolute centered
    let bad ← p.greater magnitude limit
    let intervals ← if model.spec.channelLocal then do
      let flags ← p.padLeft bad
      let width := model.spec.rangeWindow
      let flags ← match width with
        | none => .ok flags
        | some width => do
          if fieldDefault model.spec.normalized "range_action"=.text "reject" ∧ width ≤ (segment.stop : ℤ)-segment.start then
            let ranges ← trailingRange p segmentValues width
            let thresholds ← p.raw.scope.modal.recovery.genericArray (model.ranges.getD .null)
            let thresholds ← p.raw.columnBroadcast thresholds
            let rangeBad ← p.greater ranges thresholds
            let rangeBad ← p.clearBefore rangeBad (width-1)
            let rangeBad ← p.dilate rangeBad width (-(width/2))
            p.unionFlags flags rangeBad
          else pure flags
      let lookahead := match width with
        | none => 0
        | some width => if fieldDefault model.spec.normalized "range_action"=.text "diagnostic" then 0 else width-1
      localChannelIntervals p model segment segmentValues flags settle lookahead ((List.range names.length).zip names)
      else do
        let width ← match bad.shape with
          | [_,width] => .ok width
          | _ => .error (.fieldFailure "global difference flag geometry")
        let indices ← globalFlagIndices bad names.length width
        let indices := indices.map (fun index => index+(segment.start : ℤ)+1)
        let events ← globalIntervalsFuel bad names segment.start segment.stop settle wait indices.length indices
        let recoveryEnd := match wait with | none => (segment.stop : ℤ) | some wait => min segment.stop (segment.start+wait)
        let initial := if 0 < segment.start ∧ (segment.start : ℤ) < recoveryEnd then
          [Interval.mk segment.start recoveryEnd segment.start segment.start names
            (if wait=none then "filter_state_uncertainty_unbounded" else "filter_state_bounded_recovery") false] else []
        pure (events ++ initial)
    let rest ← detectSegments p model values center limit names settle wait tail
    pure (intervals ++ rest)

def projectEmpirical (p : Primitives) (model : EpochRecovery) (mixing : ArrayImage Atom)
    (channels : List String) : Check EpochRecovery := do
  ensure (mixing.shape=[channels.length,model.channels.length] ∧ finiteArray mixing)
    "empirical recovery finite named mixing geometry"
  let magnitude ← p.absolute mixing
  let envelope ← p.raw.matrixMultiply magnitude model.envelope
  let nativeRecord ← epochRecoveryRecord p.raw.scope.modal model
  let nativeDigest ← p.raw.scope.modal.recovery.input.execution.semanticHash (.object nativeRecord)
  let mixingValue ← p.raw.scope.modal.arrayJSON mixing
  let sosValue ← p.raw.scope.modal.arrayJSON model.sos
  let envelopeValue ← p.raw.scope.modal.arrayJSON envelope
  let provenance := Fields.cons "projection" mixingValue (.cons "native_recovery_digest" (.text nativeDigest)
    ((model.provenance.erase "projection").erase "native_recovery_digest"))
  readEpochRecovery p.raw.scope.modal (.object (Fields.ofList
    [("schema",.text (if model.bidirectional then "n2p3_bidirectional_state_recovery/1" else "n2p3_epoch_state_recovery/2")),
     ("sos",sosValue),("sample_rate_hz",.number model.rate),("preprocessing",.object (preprocessingRecord model.preprocessing)),
     ("channel_names",.sequence (encodeTexts channels)),("modal_envelope",envelopeValue),
     ("tolerance_v",.number model.tolerance),("fit_provenance",.object provenance),
     ("mode_identity",model.serializedIdentity.getD .null)]))

def mergeReset (exactConstant : Bool) (boundaries : List ℤ) (name : String) (rows : List Interval) : List Interval → List Interval
  | [] => rows
  | row :: tail =>
    let crosses := exactConstant && decide (row.start ∈ boundaries)
    let rows := match rows.reverse with
      | previous :: rest =>
        if row.start ≤ previous.stop ∧ crosses=false then
          rest.reverse ++ [⟨previous.start,max previous.stop row.stop,min previous.detected row.detected,
            max previous.confirmed row.confirmed,[name],"raw_dependency_outlier",true⟩]
        else rows ++ [⟨row.start,row.stop,row.detected,row.confirmed,[name],"raw_dependency_outlier",true⟩]
      | [] => [⟨row.start,row.stop,row.detected,row.confirmed,[name],"raw_dependency_outlier",true⟩]
    mergeReset exactConstant boundaries name rows tail

def dependentParents (names : List String) (row : List Atom) : List String :=
  ((names.zip row).filter (fun pair => !(zeroNumber pair.2))).map Prod.fst

def projectedChannels (p : Primitives) (model : ContinuousQC) (result : Result)
    (mixing : ArrayImage Atom) (channels native : List String) (boundaries : List ℤ)
    (waits : Option (List ℕ)) (bidirectional : Bool) : List (ℕ × String) → Check (List Interval)
  | [] => .ok []
  | (index,name) :: tail => do
    let row ← p.arrayRow mixing index
    let intervals ← if row.all zeroNumber then
      pure [Interval.mk 0 result.samples 0 0 [name] "unfitted_signal_dependency" true]
      else do
        let parents := dependentParents native row
        let raw := (result.intervals.filter (fun interval => interval.reset &&
          decide (parents.toFinset ∩ interval.channels.toFinset ≠ ∅))).mergeSort (fun a b => (a.start,a.stop) ≤ (b.start,b.stop))
        let exactConstant := model.spec.flatlineDuration.isSome
        let evidence := raw.map (fun row => Interval.mk row.start row.stop row.detected row.confirmed [name]
          (if row.reason="raw_exact_constant" then "raw_exact_constant" else "raw_dependency_outlier") true)
        let merged := mergeReset exactConstant boundaries name [] raw
        let events := if exactConstant then evidence else merged
        if bidirectional then pure events
        else
          let resets := merged.map (fun row => (row.stop,row.confirmed)) ++ boundaries.map (fun sample => (sample,sample))
          let releases ← resets.mapM fun (start,confirmation) => do
            let stop ← match waits with
              | none => .ok (result.samples : ℤ)
              | some waits => match waits[index]? with
                | none => .error (.fieldFailure "projected recovery wait channel index")
                | some wait => .ok (min (result.samples : ℤ) (start+wait))
            let stop := if exactConstant then min stop ((boundaries.find? (fun sample => decide (start < sample))).getD result.samples) else stop
            pure (if start < stop then [Interval.mk start stop start confirmation [name]
              (if waits.isSome then "empirical_epoch_state_recovery" else "filter_state_uncertainty_unbounded") false] else [])
          pure (events ++ releases.flatten)
    let rest ← projectedChannels p model result mixing channels native boundaries waits bidirectional tail
    pure (intervals ++ rest)

def projectResult (p : Primitives) (model : ContinuousQC) (result : Result) (mixing : ArrayImage Atom)
    (channels : List String) (boundaries : List ℤ) : Check Result := do
  let native ← model.channels.mapM (fun value => text value "continuous native channel string")
  ensure (mixing.shape=[channels.length,native.length] ∧ finiteArray mixing) "finite QC dependency projection geometry"
  let recovery ← match model.epochRecovery with
    | none => .ok none
    | some recovery => do
      let projected ← projectEmpirical p recovery mixing channels
      pure (some projected)
  let bidirectional := recovery.any (fun recovery => decide (fieldDefault recovery.preprocessing "filter_phase"=.text "zero"))
  let waits ← match recovery with
    | none => .ok none
    | some recovery => if bidirectional then .ok none else (p.epochWaits recovery).map some
  let boundaries := boundaries.dedup.mergeSort (· ≤ ·)
  let intervals ← projectedChannels p model result mixing channels native boundaries waits bidirectional ((List.range channels.length).zip channels)
  pure ⟨result.samples,result.rate,intervals,result.modelDigest,channels,true,result.lookahead⟩

def identityMixing (channels : ℕ) : ArrayImage Atom :=
  ⟨.floating,[channels,channels],((List.range channels).map (fun i => (List.range channels).map
    (fun j => Atom.real (if i=j then 1 else 0)))).flatten,by simp [Function.comp_def]⟩

def detect (p : Primitives) (model : ContinuousQC) (raw : Raw) : Check Result := do
  let rate ← strictReal raw.rate
  ensure (rate=model.rate) "continuous QC raw fitted sample rate"
  let names ← model.channels.mapM (fun value => text value "continuous QC raw pick channel string")
  ensure (names ⊆ raw.names) "continuous QC measured raw channels"
  let values ← p.raw.getData raw names
  ensure (finiteArray values) "nonfinite raw rejected before continuous QC"
  let center ← p.raw.scope.modal.recovery.genericArray model.center
  let center ← p.raw.columnBroadcast center
  let scale ← p.raw.scope.modal.recovery.genericArray model.scale
  let scale ← p.raw.columnBroadcast scale
  let differenceZ ← numeric (fieldDefault model.spec.normalized "difference_z") "continuous difference multiplier"
  let limit ← p.raw.scalarMultiply differenceZ scale
  let wait ← match model.recovery with | none => .ok none | some box => p.stateBoxWait box
  let segments ← p.segments raw
  let settle ← integer (fieldDefault model.spec.normalized "settle_samples") "continuous settle samples integer"
  let intervals ← detectSegments p model values center limit names settle wait segments
  let record ← continuousQCRecord p.raw.scope.modal model
  let digest ← p.raw.scope.modal.recovery.input.execution.semanticHash (.object record)
  let lookahead ← continuousLookahead p.raw.scope.modal model
  let result : Result := ⟨raw.samples,model.rate,intervals,digest,names,model.spec.channelLocal,lookahead⟩
  if model.spec.channelLocal then
    let boundaries := ((segments.filter (fun span => decide (0 < span.start))).map (fun span => (span.start : ℤ)))
    projectResult p model result (identityMixing names.length) names boundaries
  else pure result

theorem empty_reset_clusters_produce_no_intervals (endSample settle lookahead : ℤ) (names : List String) (reason : String) :
    cluster [] endSample settle names reason lookahead=.ok [] := rfl

theorem no_dependency_row_has_no_parents (names : List String) :
    dependentParents names (List.replicate names.length (Atom.real 0))=[] := by
  unfold dependentParents
  suffices h : (names.zip (List.replicate names.length (Atom.real 0))).filter
      (fun pair => !(zeroNumber pair.2))=[] by rw [h]; rfl
  apply List.filter_eq_nil_iff.mpr
  rintro ⟨a,b⟩ h
  have hb : b=Atom.real 0 := List.eq_of_mem_replicate (List.of_mem_zip h).2
  simp [hb,zeroNumber]

end

end N2P3.PersonalERPContinuous
