import N2P3.PersonalERPRawInputTail

/-!
# Source-fixed nonempty nine-row public-array witness

The field choices are fixed from the one-channel/two-sample R01 control plan:
250 Hz, [0,8) ms, no filtering/notch/QC/reference transform, nine candidates,
unit [1,0] readout. This is a receipt-free public-constructor slice, not proof
of the raw EDF builder/receipt producer, a fitted estimator or EEG efficacy.
Each partial native operation below implements one named primitive on the
small supported literal domain; other operations return explicit unrefined.
There is no default-valid model or whole-validator implementation leaf.
-/

namespace N2P3.PersonalERPNineInputWitness
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open N2P3.PersonalERPData N2P3.PersonalERPScalar N2P3.PersonalERPModelRuntime
open N2P3.ERPTemporalNPZ N2P3.PersonalERPArrayIngress

noncomputable section
set_option maxRecDepth 8192
set_option maxHeartbeats 2000000
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def texts (values : List String) : RawArray where
  kind := .unicode
  dtype := "unicode"
  shape := [values.length]
  cells := values.map Cell.text
  storageGeometry := by simp
  dtypeContents := by
    intro cell member
    obtain ⟨value,_,rfl⟩ := List.mem_map.mp member
    exact ⟨value,rfl⟩

def integers (values : List ℤ) : RawArray where
  kind := .signed
  dtype := "int64"
  shape := [values.length]
  cells := values.map (fun z => .atom (.integer z))
  storageGeometry := by simp
  dtypeContents := by
    intro cell member
    obtain ⟨value,_,rfl⟩ := List.mem_map.mp member
    exact ⟨value,rfl⟩

def floats (shape : List ℕ) (values : List ℝ) (length : values.length=shape.prod) : RawArray where
  kind := .floating
  dtype := "float64"
  shape := shape
  cells := values.map (fun r => .atom (.real r))
  storageGeometry := by simpa using length
  dtypeContents := by
    intro cell member
    obtain ⟨value,_,rfl⟩ := List.mem_map.mp member
    exact ⟨.real value,rfl,Or.inl ⟨value,rfl⟩⟩

def booleans (shape : List ℕ) (values : List Bool) (length : values.length=shape.prod) : RawArray where
  kind := .boolean
  dtype := "bool"
  shape := shape
  cells := values.map (fun b => .atom (.boolean b))
  storageGeometry := by simpa using length
  dtypeContents := by
    intro cell member
    obtain ⟨value,_,rfl⟩ := List.mem_map.mp member
    exact ⟨value,rfl⟩

def unsupported {α : Type} (site : String) : R α := .error (.unrefined site)

def nativeCast (target : Cast) (raw : RawArray) : R (CastResult target raw) :=
  match target with
  | .string =>
    if h : raw.kind=.unicode then .ok ⟨raw,h,rfl⟩
    else if raw.kind=.object then
      if cells : ∀ cell ∈ raw.cells, ∃ text, cell=.text text then
        let image : RawArray :=
          { raw with
            kind := .unicode
            dtype := "unicode"
            dtypeContents := by intro cell member; exact cells cell member }
        .ok ⟨image,rfl,rfl⟩
      else unsupported "object-array string cast outside fixed text cells"
    else unsupported "string cast outside fixed string arrays"
  | .float64 =>
    if h : raw.kind=.floating then
      if raw.dtype="float64" then .ok ⟨raw,h,rfl⟩ else unsupported "float64 cast outside fixed dtype"
    else unsupported "float64 cast outside fixed floating arrays"
  | .float32 => unsupported "float32 cast unused by nine-row public input"
  | .int64 =>
    if h : raw.kind=.signed then
      if raw.dtype="int64" then .ok ⟨raw,h,rfl⟩ else unsupported "int64 cast outside fixed dtype"
    else unsupported "int64 cast outside fixed signed arrays"
  | .boolean =>
    if h : raw.kind=.boolean then .ok ⟨raw,h,rfl⟩ else unsupported "boolean cast outside fixed Boolean arrays"

def nativeLess (left right : Cell) : R Bool :=
  match atomNumber (cellAtom left),atomNumber (cellAtom right) with
  | some a,some b => .ok (decide (a < b))
  | _,_ => unsupported "fixed finite scalar less-than"

def nativeLessEqual (left right : Cell) : R Bool :=
  match atomNumber (cellAtom left),atomNumber (cellAtom right) with
  | some a,some b => .ok (decide (a ≤ b))
  | _,_ => unsupported "fixed finite scalar less-equal"

def nativeGreaterEqual (left right : Cell) : R Bool := nativeLessEqual right left

def nativeEqual (left right : Cell) : R Bool :=
  match left,right with
  | .text a,.text b => .ok (decide (a=b))
  | _,_ => match atomNumber (cellAtom left),atomNumber (cellAtom right) with
    | some a,some b => .ok (decide (a=b))
    | _,_ => unsupported "fixed finite scalar equality"

def nativeSubtract (_dtype : String) (left right : Cell) : R Cell :=
  match atomNumber (cellAtom left),atomNumber (cellAtom right) with
  | some a,some b => .ok (.atom (.real (a-b)))
  | _,_ => unsupported "fixed finite scalar subtraction"

def nativeAdd (left right : Cell) : R Cell :=
  match left,right with
  | .atom (.integer a),.atom (.integer b) =>
    if a=0 ∧ b=0 then .ok (.atom (.integer 0)) else unsupported "fixed zero int64 addition"
  | _,_ => unsupported "fixed int64 addition tags"

def nativeFinite (raw : RawArray) : R Bool :=
  if raw.kind=.floating then .ok (decide (∀ cell ∈ raw.cells, finiteNumber (cellAtom cell)=true))
  else unsupported "fixed floating isfinite all"

def nativeFiniteRows (dtype : String) (rows : List (List Cell)) : R Bool :=
  if dtype="float64" then .ok (decide (∀ row ∈ rows, ∀ cell ∈ row, finiteNumber (cellAtom cell)=true))
  else unsupported "fixed floating coordinate isfinite all"

def nativeNorm (dtype : String) (row : List Cell) : R Cell :=
  if dtype="float64" ∧ row=[.atom (.real 0),.atom (.real 0),.atom (.real (1/10))] then
    .ok (.atom (.real (1/10)))
  else unsupported "fixed coordinate norm atom"

def nativeIntegers (kind : Kind) (dtype : String) (cells : List Cell) :
    R {values : List ℤ // values.length=cells.length} :=
  if kind=.signed ∧ dtype="int64" then .ok ⟨cells.map cellInteger,by simp⟩
  else unsupported "fixed int64 ndarray conversion"

def operations (input : PersonalERPExecutionRuntime.Primitives) : Operations :=
  ⟨input,nativeCast,nativeLess,nativeLessEqual,nativeGreaterEqual,nativeEqual,nativeSubtract,
    nativeAdd,nativeNorm,nativeFinite,nativeFiniteRows,nativeIntegers⟩

def preprocessing : Fields := Fields.ofList
  [("name",.text "lean_actual_raw_witness"),("sfreq",.number 250),("l_freq",.null),("h_freq",.null),
   ("tmin_ms",.number 0),("tmax_ms",.number 8),("n_times",.integer 2),("baseline_mode",.text "none"),
   ("zero_phase_edge_policy",.text "mne_reflect_limited")]

def timelineArrays : List RawArray :=
  [texts ["e0","e1","e2","e3","e4","e5","e6","e7","e8"],
   texts (List.replicate 9 "lean_p06_future"),texts (List.replicate 9 "P001"),
   integers [1,2,3,4,5,6,7,8,9],integers [0,250,500,750,1000,1250,1500,1750,2000],
   floats [9] [0,1,2,3,4,5,6,7,8] rfl,floats [9] (List.replicate 9 9) rfl,
   integers [0,1,2,3,4,5,6,7,8],texts (List.replicate 9 "available"),texts (List.replicate 9 ""),
   texts (List.replicate 9 "BrainSync-GTN"),texts (List.replicate 9 "lean_p06_future"),
   texts (List.replicate 9 "lean_p06_future"),texts (List.replicate 9 "lean_p06_future"),
   texts ["1","2","3","4","5","6","7","8","9"],texts (List.replicate 9 ""),
   integers (List.replicate 9 0)]

def raw : RawDataset where
  historical := false
  coordinates := .text "physical_electrode_voltages"
  name := .text "BrainSync-GTN"
  x := floats [9,1,2] (List.replicate 18 0) rfl
  channelNames := ["FZ"]
  positions := floats [1,3] [0,0,1/10] rfl
  staticMask := booleans [1] [true] rfl
  trialMask := none
  subjects := texts (List.replicate 9 "P001")
  timeline := ⟨fun j => timelineArrays.get ⟨j.val,j.isLt⟩,true,false⟩
  labels := none
  quality := none
  metadataIsDataFrame := true
  metadataRows := 9
  metadataColumns := 1
  preprocessingArgs := preprocessing
  provenance := Fields.ofList [("source_reference",.text "left earlobe"),("simulation_status",.text "declared_mock")]
  provenanceIsMapping := true
  provenanceIsDict := true
  dictImpliesMapping := by simp
  identityCache := none
  lineage := .absent
  attestation := none

def literalStrings : List String :=
  ["","lean_actual_raw_witness","BrainSync-GTN","P001","left earlobe","lean_p06_future",
    "1","2","3","4","5","6","7","8","9"]

structure Leaves (p : PersonalERPExecutionRuntime.Primitives) : Prop where
  strip : ∀ value ∈ literalStrings, p.data.input.original.execution.strip value=value
  canonical : p.data.input.original.canonicalChannel "FZ"=.ok "FZ"
  mathQuality : p.data.mathFinite (.number 30)=.ok true
  tupleFinite : p.data.numpyTupleFiniteAll [.number 250,.number 0,.number 8]=.ok true
  floor : p.data.floorSamples (.number 0) (.number 8) (.number 250)=.ok 2
  divide : p.data.divide (.number 250) (.number 2)=.ok 125
  quoteSource : p.data.input.original.quoteURI "BrainSync-GTN"="BrainSync-GTN"
  quoteSubject : p.data.input.original.quoteURI "P001"="P001"

def resolvedPreprocessing : Fields := Fields.ofList
  [("name",.text "lean_actual_raw_witness"),("sfreq",.number 250),("l_freq",.null),("h_freq",.null),
   ("tmin_ms",.number 0),("tmax_ms",.number 8),("n_times",.integer 2),("baseline_mode",.text "none"),
   ("signal_unit",.text "V"),("filter_method",.text "iir"),("filter_order",.integer 4),("filter_phase",.text "zero"),
   ("causal_iir_initial_state",.text "not_applicable"),("notch_freqs_hz",.sequence .nil),
   ("notch_quality_factor",.number 30),("zero_phase_edge_policy",.text "mne_reflect_limited"),
   ("resample_domain",.text "epoched"),("resample_method",.text "fft"),("resample_npad",.text "auto"),
   ("resample_window",.text "auto"),("resample_pad",.text "edge"),("trial_reference_window_ms",.null),
   ("trial_reference_center",.text "mean"),("trial_reference_scale",.text "none"),("reject_threshold_v",.null)]

theorem preprocessing_fields_are_known :
    readPreprocessingArgs preprocessing=.ok resolvedPreprocessing := by
  have known : fieldKeys preprocessing ⊆ preprocessingKeywords := by
    simp [preprocessing,preprocessingKeywords,preprocessingDefaults,fieldKeys,Fields.ofList]
  unfold readPreprocessingArgs
  rw [show ensure (fieldKeys preprocessing ⊆ preprocessingKeywords) "unknown preprocessing keyword"=.ok () by
    simp only [ensure,if_pos known]]
  rfl

theorem resolved_preprocessing_is_idempotent :
    readPreprocessingArgs resolvedPreprocessing=.ok resolvedPreprocessing := by
  have keys : fieldKeys resolvedPreprocessing=preprocessingKeywords := rfl
  have known : fieldKeys resolvedPreprocessing ⊆ preprocessingKeywords := by
    intro key member
    simpa only [keys] using member
  unfold readPreprocessingArgs
  rw [show ensure (fieldKeys resolvedPreprocessing ⊆ preprocessingKeywords) "unknown preprocessing keyword"=.ok () by
    simp only [ensure,if_pos known]]
  rfl

theorem preprocessing_from_individual_scalar_atoms (p : PersonalERPExecutionRuntime.Primitives) (h : Leaves p) :
    PersonalERPDataRuntime.checkPreprocessing p.data resolvedPreprocessing=
      .ok ⟨250,0,8,2,none,none,none,[],30⟩ := by
  have stripName := h.strip "lean_actual_raw_witness" (by simp [literalStrings])
  unfold PersonalERPDataRuntime.checkPreprocessing
  rw [resolved_preprocessing_is_idempotent]
  simp [resolvedPreprocessing,fieldDefault,Fields.ofList,Fields.get,valuesToList,
    PersonalERPDataRuntime.checkNotches,PersonalERPDataRuntime.notchValues,PersonalERPDataRuntime.realType,
    PersonalERPDataRuntime.mathematicalFinite,PersonalERPDataRuntime.tupleFinite,PersonalERPDataRuntime.checkEdge,
    PersonalERPDataRuntime.checkWindow,PersonalERPDataRuntime.optionalType,PersonalERPDataRuntime.optionalPositive,
    PersonalERPModelRuntime.numeric,PersonalERPModels.numeric,PersonalERPModelRuntime.integer,PersonalERPExecution.integer,
    numericScalar,numericNonBool,PersonalAdmission.PyEq,PersonalAdmission.normalizeValue,
    require,liftCheck,setOption,setContains,builtinHashable,stripName,h.mathQuality,h.tupleFinite,h.floor,h.divide]

end
end N2P3.PersonalERPNineInputWitness
