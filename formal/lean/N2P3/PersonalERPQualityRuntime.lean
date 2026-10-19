import N2P3.PersonalERPModelRuntime

/-!
# Runtime scalar and actual-set refinement of personal/legacy epoch QC

The frozen decoded reader remains unchanged. This layer places each actual
NumPy finite test at its source position and retains atomic TypeError and
OverflowError distinctions. Ordinary comparison-only fields, including
cv_splits, keep their wider Python integer domain. The personal reader's
TypeError/ValueError catch is explicit for these refined faults; inherited
structural/conversion faults retain a named catch-class boundary, not an
invented Python class. No complete validator or default model is a premise.
-/

namespace N2P3.PersonalERPQualityRuntime
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPExecution N2P3.PersonalERPRaw N2P3.PersonalERPModels
open N2P3.PersonalERPQuality N2P3.PersonalERPScalar
open N2P3.PersonalERPModelRuntime

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Primitives where
  original : PersonalERPQuality.Primitives
  finite : PersonalERPModelRuntime.Finite
  arrays : PersonalERPModelRuntime.Arrays

def Primitives.models (p : Primitives) : PersonalERPModelRuntime.Modal := ⟨p.original.models,p.finite,p.arrays⟩
def Primitives.pythonInt (p : Primitives) (value : Value) : R ℤ := liftCheck (p.original.pythonInt value)

def readFiniteFloat (p : PersonalERPModelRuntime.Recovery) (value : Value) : R ℝ :=
  liftCheck (PersonalERPModels.readFiniteFloat p.original value)
def originalNulls (value : Value) : R (List Bool) := liftCheck (PersonalERPQuality.originalNulls value)
def maskedThresholds (ptp flat mask : List Atom) (a b : List Bool) : R (List Atom × List Atom) :=
  liftCheck (PersonalERPQuality.maskedThresholds ptp flat mask a b)
def checkDiagnosticCells (counts lower upper : List Atom) : R Unit :=
  liftCheck (PersonalERPQuality.checkDiagnosticCells counts lower upper)

def readPersonalPolicy (scalar : PersonalERPModelRuntime.Finite) (value : Value) : R Fields := do
  let fields ← objectAt value "personal epoch quality policy mapping"
  require (fieldKeys fields ⊆ fieldKeys personalPolicyDefaults) "personal epoch quality known policy kwargs"
  let fields := Fields.ofList ((fieldKeys personalPolicyDefaults).map fun key =>
    (key,(fields.get key).getD (fieldDefault personalPolicyDefaults key)))
  require (fieldDefault fields "method"=.text "personal_epoch_signal_evidence") "personal policy method"
  let multiplier ← numpyNumeric scalar (fieldDefault fields "diagnostic_mad_z") "finite personal diagnostic multiplier"
  require (0 < multiplier) "positive personal diagnostic multiplier"
  let minimum ← integer (fieldDefault fields "minimum_group_epochs") "personal exact integer minimum group epochs"
  require (2 ≤ minimum) "personal minimum group epochs at least two"
  let fraction ← numpyNumeric scalar (fieldDefault fields "max_bad_channel_fraction") "finite personal bad channel fraction"
  require (0 ≤ fraction ∧ fraction < 1) "personal bad channel fraction interval"
  pure fields

def constructPersonalQuality (p : Primitives) (value : Value) : R PersonalQuality := do
  let fields ← objectAt value "personal epoch quality record mapping"
  require (fieldDefault fields "schema"=.text "n2p3_personal_epoch_quality/1" ∧
    fieldDefault fields "method"=.text "personal_epoch_signal_evidence") "personal epoch quality schema method"
  -- Source constructs the policy here but invokes its validator in record().
  let policyValue ← required fields "policy"
  let policyFields ← objectAt policyValue "personal epoch quality policy mapping"
  require (fieldKeys policyFields ⊆ fieldKeys personalPolicyDefaults) "personal epoch quality constructor kwargs"
  let lowerValue ← required fields "diagnostic_log_lower"
  let lower ← p.models.recovery.input.execution.floatArray lowerValue
  let upperValue ← required fields "diagnostic_log_upper"
  let upper ← p.models.recovery.input.execution.floatArray upperValue
  let countsValue ← required fields "diagnostic_group_counts"
  let counts ← p.models.recovery.genericArray countsValue
  let observedValue ← required fields "fit_observed_channel_mask"
  let observed ← p.models.recovery.genericArray observedValue
  let fitEpochs ← required fields "fit_n_epochs"
  let groupsValue ← required fields "fit_groups"
  let groups ← tupleJSON groupsValue
  let subjectsValue ← required fields "fit_subject_ids"
  let subjects ← tupleJSON subjectsValue
  let groupSource ← required fields "fit_group_source"
  let fitDigest ← required fields "fit_data_digest"
  let provenance ← required fields "fit_provenance"
  let policy ← readPersonalPolicy p.finite policyValue
  let channels ← match observed.shape with
    | [] => .error (.inherited (.fieldFailure "len personal observed mask"))
    | channels :: _ => .ok channels
  let fitEpochCount ← numeric fitEpochs "personal comparable fit epoch count"
  require (0 < channels ∧ 2 ≤ fitEpochCount ∧ observed.dtype=.boolean ∧ observed.shape=[channels] ∧
    lower.shape=[channels,3] ∧ upper.shape=[channels,3] ∧ counts.shape=[channels,3] ∧ counts.dtype=.integer)
    "personal quality geometry and support dtype"
  require (∀ count ∈ counts.values, ∃ z, count=.integer z ∧ 0 ≤ z) "nonnegative personal group support"
  checkDiagnosticCells counts.values lower.values upper.values
  require (PyTruthy groups ∧ PyTruthy subjects ∧ PyTruthy fitDigest) "personal quality nonempty fit provenance"
  setOption "personal quality fit group source" [.text "subject_ids",.text "explicit_fit_groups"] groupSource
  pure ⟨policy,lower,upper,counts,observed,channels,fitEpochs,groups,subjects,groupSource,fitDigest,provenance⟩

def personalQualityRecord (p : Primitives) (model : PersonalQuality) : R Fields := do
  let counts ← p.models.arrayJSON model.counts
  let observed ← p.models.arrayJSON model.observed
  let provenance ← objectAt model.provenance "personal quality exported provenance mapping"
  pure (Fields.ofList [("schema",.text "n2p3_personal_epoch_quality/1"),
    ("method",.text "personal_epoch_signal_evidence"),("policy",.object model.policy),
    ("n_channels",.integer model.channels),("feature_names",.sequence (encodeTexts
      ["log_ptp_v","log_std_v","log_normalized_difference_rms"])),
    ("diagnostic_log_lower",.sequence (valuesOfList (rowsOfThree model.lower.values))),
    ("diagnostic_log_upper",.sequence (valuesOfList (rowsOfThree model.upper.values))),
    ("diagnostic_group_counts",counts),("diagnostic_scope",.text "calibration_group_envelope_no_false_positive_rate_claim"),
    ("hard_rejection_basis",.text "exact_constant_epoch_signal"),
    ("raw_quality_requirement",.text "preserve_upstream_observation_masks_and_recovery"),
    ("distribution_diagnostics_are_artifact_proof",.boolean false),("fit_observed_channel_mask",observed),
    ("fit_n_epochs",model.fitEpochs),("fit_groups",model.fitGroups),("fit_subject_ids",model.fitSubjects),
    ("fit_group_source",model.groupSource),("fit_data_digest",model.fitDigest),("fit_provenance",.object provenance)])

def wrapPersonal {α : Type} (result : R α) : R α :=
  match result with
  | .ok value => .ok value
  | .error (.atomic _ _ .typeError) => .error (.sourceValueError "invalid personal epoch quality record caught TypeError")
  | .error (.atomic _ _ (.other "ValueError")) => .error (.sourceValueError "invalid personal epoch quality record caught ValueError")
  | .error (.atomic _ _ (.other "KeyError")) => .error (.sourceValueError "invalid personal epoch quality record caught KeyError")
  | .error (.sourceValueError _) => .error (.sourceValueError "invalid personal epoch quality record caught ValueError")
  | .error fault => .error fault

def personalQualityFromRecord (p : Primitives) (value : Value) : R PersonalQuality := do
  let fields ← match value with
    | .object fields => .ok fields
    | _ => .error (.sourceValueError "unknown personal epoch quality schema or method")
  require (fieldDefault fields "schema"=.text "n2p3_personal_epoch_quality/1" ∧
    fieldDefault fields "method"=.text "personal_epoch_signal_evidence") "unknown personal epoch quality schema or method"
  wrapPersonal (do
    let model ← constructPersonalQuality p value
    let record ← personalQualityRecord p model
    let actual ← p.models.recovery.input.execution.semanticHash (.object record)
    let saved ← p.models.recovery.input.execution.semanticHash value
    require (actual=saved) "personal quality canonical semantic digest"
    pure model)

def numericValues : List Value → R (List ℝ)
  | [] => .ok []
  | value :: rest => do
    let number ← numeric value "numeric fold policy value"
    let tail ← numericValues rest
    pure (number :: tail)

def readFoldPolicy (p : Primitives) (value : Value) : R Fields := do
  let fields ← objectAt value "fold artifact policy mapping"
  let quantilesValue ← required fields "candidate_quantiles"
  let quantileTuple ← tupleJSON quantilesValue
  let fractionsValue ← required fields "candidate_bad_channel_fractions"
  let fractionTuple ← tupleJSON fractionsValue
  require (fieldKeys fields ⊆ fieldKeys foldPolicyDefaults) "fold policy known kwargs"
  let fields := Fields.ofList ((fieldKeys foldPolicyDefaults).map fun key =>
    (key,if key="candidate_quantiles" then quantileTuple else if key="candidate_bad_channel_fractions" then
      fractionTuple else (fields.get key).getD (fieldDefault foldPolicyDefaults key)))
  let quantiles ← tupleValue quantileTuple
  let quantiles ← numericValues quantiles
  require (quantiles ≠ [] ∧ ∀ q ∈ quantiles, 1/2 < q ∧ q < 1) "fold quantile interval"
  require (quantiles=quantiles.dedup.mergeSort (· ≤ ·)) "fold quantiles sorted unique"
  let flat ← numeric (fieldDefault fields "flat_quantile") "fold flat quantile numeric"
  require (0 ≤ flat ∧ flat < 1/2) "fold flat quantile interval"
  let kappaAtoms ← if fieldDefault fields "max_bad_channel_fraction"=.null then do
    let values ← tupleValue fractionTuple
    let floats ← values.mapM p.models.recovery.floatScalar
    pure floats
    else do
      let number ← p.models.recovery.floatScalar (fieldDefault fields "max_bad_channel_fraction")
      pure [number]
  -- _kappa_candidates completes every float conversion before validate()
  -- performs the first interval comparison on the resulting tuple.
  let kappas ← kappaAtoms.mapM PersonalERPModelRuntime.strictReal
  require (kappas ≠ [] ∧ ∀ k ∈ kappas, 0 ≤ k ∧ k < 1) "fold candidate bad channel fraction interval"
  require (kappas=kappas.dedup.mergeSort (· ≤ ·)) "fold bad channel fractions sorted unique"
  if fieldDefault fields "max_bad_channel_fraction" ≠ .null then
    let maximum ← numeric (fieldDefault fields "max_bad_channel_fraction") "fold maximum bad fraction numeric"
    require (0 ≤ maximum ∧ maximum < 1) "fold maximum bad fraction interval"
  let z ← numpyNumeric p.finite (fieldDefault fields "global_scale_mad_z") "finite fold scale MAD multiplier"
  require (0 < z) "positive fold scale MAD multiplier"
  let retained ← numeric (fieldDefault fields "min_training_epoch_retention") "fold retained fraction numeric"
  require (0 < retained ∧ retained ≤ 1) "fold retained fraction interval"
  let splits ← numeric (fieldDefault fields "cv_splits") "fold comparable CV splits"
  require (2 ≤ splits) "fold CV at least two splits"
  let minimum ← numeric (fieldDefault fields "min_clean_epochs") "fold comparable clean epoch minimum"
  require (1 ≤ minimum) "fold clean epoch minimum positive"
  pure fields

def pythonStrings (p : Primitives) : List Value → R (List String)
  | [] => .ok []
  | value :: rest => do
    let rendered ← p.models.recovery.input.pythonString value
    let tail ← pythonStrings p rest
    pure (rendered :: tail)

def foldQualityFromRecord (p : Primitives) (value : Value) : R FoldQuality := do
  let fields ← objectAt value "fold artifact record dictionary"
  require (fieldDefault fields "method"=.text "fold_local_ptp_cv" ∧
    fieldDefault fields "ptp_normalization"=.text "within_epoch_observed_channel_median" ∧ fields.get "schema"=none)
    "fold artifact method normalization and absent schema"
  let ptpValue ← required fields "ptp_thresholds"
  let ptp ← p.models.recovery.input.execution.floatArray ptpValue
  let flatValue ← required fields "flat_std_thresholds"
  let flat ← p.models.recovery.input.execution.floatArray flatValue
  let (ptpValues,flatValues) ← match fields.get "fit_observed_channel_mask" with
    | none => .ok (ptp.values,flat.values)
    | some value => do
      let observed ← p.models.recovery.genericArray value
      require (observed.dtype=.boolean ∧ observed.shape.length=1 ∧ observed.shape=ptp.shape ∧ observed.shape=flat.shape)
        "fold artifact observed mask geometry"
      let ptpNull ← originalNulls ptpValue
      let flatNull ← originalNulls flatValue
      maskedThresholds ptp.values flat.values observed.values ptpNull flatNull
  let policyValue ← required fields "policy"
  let policy ← readFoldPolicy p policyValue
  let calibrationValue ← required fields "global_scale_calibration"
  let calibration ← objectAt calibrationValue "fold scale calibration mapping"
  let selectedValue ← required fields "selected_quantiles"
  let selected ← p.models.recovery.input.execution.floatArray selectedValue
  let fractionValue ← required fields "selected_bad_channel_fraction"
  let fraction ← readFiniteFloat p.models.recovery fractionValue
  let centerValue ← required calibration "log_center"
  let center ← readFiniteFloat p.models.recovery centerValue
  let scaleValue ← required calibration "log_robust_std"
  let scale ← readFiniteFloat p.models.recovery scaleValue
  let fitEpochsValue ← required fields "fit_n_epochs"
  let fitEpochs ← p.pythonInt fitEpochsValue
  let groupsValue ← required fields "fit_groups"
  let groups ← tupleValue groupsValue
  let groups ← pythonStrings p groups
  let subjects ← tupleValue ((fields.get "fit_subject_ids").getD (.sequence .nil))
  let subjects ← pythonStrings p subjects
  let groupSource ← p.models.recovery.input.pythonString ((fields.get "fit_group_source").getD (.text "subject_ids"))
  let cv ← objectAt ((fields.get "cv_evidence").getD (.object (Fields.ofList
    [("status",.text "unavailable_legacy_record")]))) "fold artifact CV evidence mapping"
  let provenance ← objectAt ((fields.get "fit_provenance").getD (.object .nil)) "fold artifact provenance mapping"
  let digest := fieldDefault fields "fit_data_digest"
  require (ptp.shape.length=1 ∧ flat.shape=ptp.shape ∧ selected.shape=ptp.shape ∧ ptp.values ≠ [])
    "fold artifact threshold geometry"
  require (thresholdPairs ptpValues flatValues) "fold artifact finite or absent thresholds"
  require (finiteArray selected ∧ ∀ value ∈ selected.values, ∃ r, atomNumber value=some r ∧ 1/2 < r ∧ r < 1)
    "fold artifact selected quantiles interval"
  require (0 ≤ fraction ∧ fraction < 1 ∧ 0 < scale ∧ 1 ≤ fitEpochs ∧ groups ≠ [] ∧
    (groupSource="subject_ids" ∨ groupSource="explicit_fit_groups")) "fold artifact fitted scalar geometry"
  require (digest=.null ∨ ∃ s, digest=.text s ∧ s.length=64 ∧ ∀ char ∈ s.toList, char ∈ "0123456789abcdef".toList)
    "fold artifact optional lowercase SHA digest"
  pure ⟨policy,ptpValues,flatValues,selected,fraction,center,scale,fitEpochs,groups,subjects,groupSource,cv,digest,provenance⟩

def epochQualityFromRecord (p : Primitives) (value : Value) : R EpochQuality := do
  let fields ← objectAt value "epoch quality record dictionary"
  if fieldDefault fields "method"=.text "personal_epoch_signal_evidence" then
    (personalQualityFromRecord p value).map EpochQuality.personal
  else if fieldDefault fields "method"=.text "fold_local_ptp_cv" then
    (foldQualityFromRecord p value).map EpochQuality.fold
  else .error (.sourceValueError "unknown epoch quality method")

theorem refined_personal_typeerror_is_wrapped_as_ValueError (field : String) (operation : Operation) :
    wrapPersonal (Except.error (Fault.atomic field operation .typeError) : R PersonalQuality)=
      .error (.sourceValueError "invalid personal epoch quality record caught TypeError") := rfl

theorem personal_overflow_is_not_in_the_source_catch (field : String) (operation : Operation) :
    wrapPersonal (Except.error (Fault.atomic field operation .overflowError) : R PersonalQuality)=
      .error (.atomic field operation .overflowError) := rfl

end
end N2P3.PersonalERPQualityRuntime
