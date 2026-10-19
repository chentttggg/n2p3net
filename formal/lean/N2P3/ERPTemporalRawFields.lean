import N2P3.ERPTemporalJSON
import N2P3.PersonalERPInput

/-! Raw JSON field operations. A PreprocessingSpec dataclass construction
keeps NaN/Infinity and arbitrary declared values; it does not validate them.
Lineage's _canonical routine rejects nonfinite numbers at its own actual
call site. finiteView is never installed as an early source rejection gate. -/

namespace N2P3.ERPTemporalRawFields

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.ERPTemporalJSON

noncomputable section

attribute [local instance] Classical.propDecidable

attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def objectAt (raw : JValue) (site : String) : Check JFields :=
  match raw with
  | .object fields => .ok fields
  | _ => .error (.fieldFailure site)

def fieldDefault (fields : JFields) (key : String) : JValue :=
  (fields.get key).getD .null

def required (fields : JFields) (key : String) : Check JValue :=
  match fields.get key with
  | some value => .ok value
  | none => .error (.fieldFailure ("missing field:" ++ key))

def listAt (raw : JValue) (site : String) : Check (List JValue) :=
  match raw with
  | .sequence values => .ok values.toList
  | _ => .error (.fieldFailure site)

def stringAt (raw : JValue) (site : String) : Check String :=
  match raw with
  | .text value => .ok value
  | _ => .error (.sourceReject site)

def stringList (raw : JValue) (site : String) : Check (List String) := do
  let values ← match raw with
    | .sequence values => .ok values.toList
    | _ => .error (.sourceReject site)
  values.mapM (fun value => stringAt value site)

def exactFields (fields : JFields) (keys : List String) : Prop :=
  fields.keys ⊆ keys ∧ keys ⊆ fields.keys

def erase (fields : JFields) (key : String) : JFields :=
  match fields with
  | .nil => .nil
  | .cons name value tail => if name=key then erase tail key else .cons name value (erase tail key)

def truth : JValue → Bool
  | .null => false
  | .boolean value => value
  | .integer value => decide (value ≠ 0)
  | .floating (.finite value) => decide (value ≠ 0)
  | .negativeZero => false
  | .floating _ => true
  | .text value => !value.isEmpty
  | .sequence .nil | .object .nil => false
  | _ => true

def iterableValues : JValue → Check (List JValue)
  | .sequence values => .ok values.toList
  | .object fields => .ok (fields.keys.map JValue.text)
  | .text text => .ok (text.toList.map (fun char => .text (String.singleton char)))
  | _ => .error (.fieldFailure "list() of noniterable notch declaration")

def preprocessingArgs (raw : JValue) : Check JFields := do
  let fields ← objectAt raw "PreprocessingSpec kwargs mapping"
  if fields.keys ⊆ preprocessingKeywords then
    let defaults := fromFields preprocessingDefaults
    pure (JFields.ofList (preprocessingKeywords.map fun key =>
      (key,(fields.get key).getD (fieldDefault defaults key))))
  else .error (.fieldFailure "PreprocessingSpec unexpected keyword")

def preprocessingRecord (args : JFields) : Check JFields := do
  let args := if pythonEqual (fieldDefault args "zero_phase_edge_policy") (.text "legacy") then
      erase args "zero_phase_edge_policy" else args
  if !truth (fieldDefault args "notch_freqs_hz") ∧
      pythonEqual (fieldDefault args "notch_quality_factor") (.floating (.finite 30)) then
    pure (erase (erase args "notch_freqs_hz") "notch_quality_factor")
  else
    let notches ← iterableValues (fieldDefault args "notch_freqs_hz")
    pure (.cons "notch_freqs_hz" (.sequence (JValues.ofList notches)) (erase args "notch_freqs_hz"))

mutual
def canonicalLineageValue : JValue → Check N2P3.FrozenState.Value
  | .null => .ok .null
  | .boolean value => .ok (.boolean value)
  | .integer value => .ok (.integer value)
  | .floating (.finite value) => .ok (.number value)
  | .negativeZero => .ok (.number 0)
  | .floating _ => .error (.sourceReject "lineage parameters cannot contain NaN or infinity")
  | .text value => .ok (.text value)
  | .sequence values => .sequence <$> canonicalLineageValues values
  | .object fields => .object <$> canonicalLineageFields fields
def canonicalLineageValues : JValues → Check N2P3.FrozenState.Values
  | .nil => .ok .nil
  | .cons value tail => do
    let value ← canonicalLineageValue value
    let tail ← canonicalLineageValues tail
    pure (.cons value tail)
def canonicalLineageFields : JFields → Check N2P3.FrozenState.Fields
  | .nil => .ok .nil
  | .cons key value tail => do
    ensure (key ≠ "") "lineage parameter keys nonempty strings"
    let value ← canonicalLineageValue value
    let tail ← canonicalLineageFields tail
    pure (.cons key value tail)
end

/-- Exactly the four cache-attestation guards, then reconstruction of their
canonical finite output. Unknown raw sidecar fields remain outside this view. -/
def cacheAttestation (raw : JValue) : Check N2P3.FrozenState.Fields := do
  let fields ← objectAt raw "cache attestation mapping"
  ensure (fieldDefault fields "schema"=.text "n2p3net_epoch_cache_attestation/1") "cache attestation schema"
  ensure (fieldDefault fields "full_contract_validated"=.boolean true) "cache full validation flag is True"
  let bytes ← match fieldDefault fields "byte_size" with
    | .integer value => .ok value
    | _ => .error (.sourceReject "cache positive nonbool integer byte size")
  ensure (0 < bytes) "cache positive nonbool integer byte size"
  let digest ← stringAt (fieldDefault fields "sha256") "cache SHA256 string"
  ensure (LowerHex64 digest) "cache lowercase SHA256"
  pure (N2P3.FrozenState.Fields.ofList [
    ("schema",.text "n2p3net_epoch_cache_attestation/1"),("full_contract_validated",.boolean true),
    ("sha256",.text digest),("byte_size",.integer bytes)])

theorem dataclass_reconstruction_keeps_declared_nan :
    ((preprocessingArgs (.object (.cons "sfreq" (.floating .nan) .nil))).map
      (fun fields => fields.get "sfreq"))=.ok (some (.floating .nan)) := by
  have known : (JFields.cons "sfreq" (.floating .nan) .nil).keys ⊆ preprocessingKeywords := by decide
  simp only [preprocessingArgs,objectAt,known,ite_true,Except.bind,Except.pure,
    Bind.bind,Pure.pure,Except.map]
  rfl

theorem lineage_canonicalization_rejects_the_same_nan :
    canonicalLineageValue (.floating .nan)=
      .error (.sourceReject "lineage parameters cannot contain NaN or infinity") := rfl

end
end N2P3.ERPTemporalRawFields
