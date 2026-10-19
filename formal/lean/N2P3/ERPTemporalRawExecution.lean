import N2P3.ERPTemporalRawPreprocessing
import N2P3.PersonalERPExecutionRuntime

/-!
# Raw JSON receipt prefix, bundles and execution binding

Source: epochs.py:523-679. The individual receipt's actual strict semantic
hash precedes declaration parsing. The parent bundle stays raw until its
children finish and execution_binding hashes it. No blanket finite view is
inserted before either source hash. Expected name is removed at the same
asdict comparison as source and is not projected or supplied to expectedGrid.

After the actual gates, finite logical views reuse the individually expanded
source-grid/output/support/filter programs. Projection inconsistency after a
claimed strict encoder success stays explicitly unrefined, not source-rejected.
This is not a proof of native encoders, signed-zero numerical argument traces,
or a full raw Dataset validator. The original raw record is retained for the
final hash, rather than rehashing the logical finite projection.
-/

namespace N2P3.ERPTemporalRawExecution

open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalERPData
open N2P3.PersonalERPScalar N2P3.ERPTemporalJSON

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

abbrev R := PersonalERPScalar.Result

structure Operations where
  declaration : ERPTemporalRawPreprocessing.Operations
  native : PersonalERPExecutionRuntime.Primitives
  semanticHash : JValue → R String

def dictionary (value : JValue) (site : String) : R JFields :=
  match value with
  | .object fields => .ok fields
  | _ => .error (.sourceValueError site)

def listValue (value : JValue) (site : String) : R (List JValue) :=
  match value with
  | .sequence values => .ok values.toList
  | _ => .error (.sourceValueError site)

def exactInteger (value : JValue) (site : String) : R ℤ :=
  match value with
  | .integer value => .ok value
  | _ => .error (.sourceValueError site)

/-- A logical bridge after actual source guards, not a source admission
rule. Its unresolved native consistency is visible in the fault carrier. -/
def finiteAfterSource (fields : JFields) (site : String) : R Fields :=
  match finiteFields fields with
  | some fields => .ok fields
  | none => .error (.unrefined (site ++ " strict-hash/equality finite-view consistency"))

def completeDeclarationArgs (fields : JFields) : R JFields := do
  require (completePreprocessingKeys ⊆ fields.keys) "complete preprocessing declaration"
  require ((fields.get "notch_freqs_hz").isSome=(fields.get "notch_quality_factor").isSome)
    "explicit notch frequencies and quality pair"
  liftCheck (ERPTemporalRawFields.preprocessingArgs (.object fields))

structure Receipt where
  original : JFields
  declaration : JFields
  checked : PersonalERPExecutionRuntime.Receipt

def checkIndividual (p : Operations) (expectedArgs fields : JFields) : R Receipt := do
  let value := ERPTemporalRawFields.fieldDefault fields
  require (pythonEqual (value "schema") (.text "n2p3_preprocessing_execution/1")) "individual receipt schema"
  let digest ← p.semanticHash (.object (ERPTemporalRawFields.erase fields "receipt_sha256"))
  require (pythonEqual (value "receipt_sha256") (.text digest)) "receipt actual semantic digest"
  let finite ← finiteAfterSource fields "individual receipt"
  let declared ← dictionary (value "declaration") "receipt preprocessing declaration mapping"
  let declared ← completeDeclarationArgs declared
  let declaration ← ERPTemporalRawPreprocessing.validate p.declaration declared
  require (pythonEqual (.object (ERPTemporalRawFields.erase expectedArgs "name"))
    (.object (ERPTemporalRawFields.erase declaration.original "name"))) "receipt executed asdict physical declaration"
  let declaredFinite ← finiteAfterSource declaration.original "receipt executed declaration"
  -- Source removes expected.name; the reused grid expression only reads
  -- tmin_ms/tmax_ms/sfreq and uses the checked scalar count. No name is added.
  let expectedFinite ← finiteAfterSource (ERPTemporalRawFields.erase expectedArgs "name") "expected declaration without name"
  require (PersonalAdmission.PyEq (N2P3.StateLedger.fieldDefault finite "semantic_identity")
    PersonalERPExecution.operatorIdentity) "receipt supported operator identity"
  liftCheck (PersonalERPExecution.checkImplementation finite)
  let context ← PersonalERPModelRuntime.objectAt (N2P3.StateLedger.fieldDefault finite "source_context") "receipt source context"
  let grid ← PersonalERPModelRuntime.objectAt (N2P3.StateLedger.fieldDefault finite "epoch_grid") "receipt epoch grid"
  let output ← PersonalERPModelRuntime.objectAt (N2P3.StateLedger.fieldDefault finite "output") "receipt output mapping"
  let filtering ← PersonalERPModelRuntime.objectAt (N2P3.StateLedger.fieldDefault finite "filter") "receipt filter mapping"
  let source ← PersonalERPExecutionRuntime.sourceGrid p.native context
  PersonalERPExecutionRuntime.sourceBands p.native declaredFinite declaration.scalars context
  let expected ← PersonalERPExecutionRuntime.expectedGrid p.native expectedFinite context declaration.scalars
  require (PersonalAdmission.PyEq (.object grid) (.object expected.fields)) "actual source epoch grid equality"
  require (declaration.scalars.nTimes ≤ expected.resampled) "resampled output coverage"
  let geometry ← PersonalERPExecutionRuntime.outputGeometry output declaration.scalars
  PersonalERPExecutionRuntime.rawScopeContext p.native context output source geometry
  PersonalERPExecutionRuntime.filterSupports context source geometry
  PersonalERPExecutionRuntime.filterPolicy p.native filtering (rawPreprocessing declaredFinite) declaration.scalars
  let _ ← liftCheck (PersonalERPExecution.nonemptyText
    (N2P3.StateLedger.fieldDefault finite "timing_source") "execution nonempty timing source")
  pure ⟨fields,declaration.original,⟨finite,declaredFinite,source,geometry⟩⟩

def children (p : Operations) (expectedArgs : JFields) : List JValue → R (List Receipt)
  | [] => .ok []
  | value :: tail => do
    let fields ← dictionary value "bundle child dictionary"
    require (pythonEqual (ERPTemporalRawFields.fieldDefault fields "schema")
      (.text "n2p3_preprocessing_execution/1")) "bundle contains individual receipts only"
    let child ← checkIndividual p expectedArgs fields
    let rest ← children p expectedArgs tail
    pure (child :: rest)

structure CheckedRecord where
  original : JFields
  receipts : List Receipt
  unknown : JValue

def checkRecord (p : Operations) (expectedArgs : JFields) (raw : JValue) : R CheckedRecord := do
  let fields ← dictionary raw "execution receipt dictionary"
  let value := ERPTemporalRawFields.fieldDefault fields
  if pythonEqual (value "schema") (.text "n2p3_preprocessing_execution_bundle/1") then
    let values ← listValue (value "recordings") "execution bundle recording list"
    let unknown ← exactInteger (value "unverified_inputs") "execution bundle unknown exact integer"
    require (0 ≤ unknown ∧ (values ≠ [] ∨ unknown ≠ 0)) "nonnegative unknown nonempty bundle"
    let receipts ← children p expectedArgs values
    pure ⟨fields,receipts,.integer unknown⟩
  else
    let receipt ← checkIndividual p expectedArgs fields
    pure ⟨fields,[receipt],(fields.get "unverified_inputs").getD (.integer 0)⟩

def executionBinding (p : Operations) (expectedArgs : JFields) (value : Option JValue) :
    R (Option PersonalERPExecution.Binding) := do
  match value with
  | none | some .null => pure none
  | some value =>
    let record ← checkRecord p expectedArgs value
    let identity := if ERPTemporalRawFields.truth record.unknown ∨ record.receipts=[] then none
      else some PersonalERPExecution.operatorIdentity
    let digest ← p.semanticHash (.object record.original)
    pure (some ⟨identity,digest⟩)

theorem absent_record_skips_all_declaration_and_hash_operations (p : Operations) (expectedArgs : JFields) :
    executionBinding p expectedArgs none=.ok none := rfl

theorem null_record_skips_all_declaration_and_hash_operations (p : Operations) (expectedArgs : JFields) :
    executionBinding p expectedArgs (some .null)=.ok none := rfl

def rawUnknownBundle : JFields := JFields.ofList
  [("schema",.text "n2p3_preprocessing_execution_bundle/1"),
   ("recordings",.sequence .nil),("unverified_inputs",.integer 1),("unused",.floating .nan)]

/-- This is the actual header/empty-child program, not an assumption of
complete receipt or Dataset validation. A later binding hash still runs. -/
theorem bundle_headers_do_not_imply_all_fields_finite (p : Operations) (expectedArgs : JFields) :
    checkRecord p expectedArgs (.object rawUnknownBundle)=.ok ⟨rawUnknownBundle,[],.integer 1⟩ ∧
      finiteFields rawUnknownBundle=none := by
  constructor
  · simp [checkRecord,rawUnknownBundle,dictionary,listValue,exactInteger,children,
      ERPTemporalRawFields.fieldDefault,JFields.ofList,JFields.get,JValues.toList,
      pythonEqual,scalarNaN,require]
  · simp [rawUnknownBundle,JFields.ofList,finiteFields,finiteView,finiteValues]

theorem bundle_nonfinite_data_reaches_actual_final_hash (p : Operations) (expectedArgs : JFields)
    (fault : Fault) (failed : p.semanticHash (.object rawUnknownBundle)=.error fault) :
    executionBinding p expectedArgs (some (.object rawUnknownBundle))=.error fault := by
  simp [executionBinding,(bundle_headers_do_not_imply_all_fields_finite p expectedArgs).1,failed]

end
end N2P3.ERPTemporalRawExecution
