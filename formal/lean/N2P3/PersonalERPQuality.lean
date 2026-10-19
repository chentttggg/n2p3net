import N2P3.PersonalERPModels

/-!
# Explicit personal and legacy epoch-quality readers

Source: personal_quality.py:50-66,104-144,175-217;
artifact_sidecar.py:59-143; artifact.py:66-122,587-632.
These are field-level readers, including unsupported diagnostic NaN/null
coordinates and the legacy reader's intentionally weaker export policy.
Neither a whole epoch-quality validator nor a prevalidated model is an input.
Per-array conversions and elementary Python scalar conversions remain named
representation boundaries. This module is P06 work in progress.
-/

namespace N2P3.PersonalERPQuality

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPExecution N2P3.PersonalERPRaw N2P3.PersonalERPModels

noncomputable section

attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Primitives where
  models : ModalPrimitives
  pythonInt : Value → Check ℤ

def personalPolicyDefaults : Fields := Fields.ofList
  [("method",.text "personal_epoch_signal_evidence"),("diagnostic_mad_z",.number 6),
   ("minimum_group_epochs",.integer 4),("max_bad_channel_fraction",.number (1/2))]

def readPersonalPolicy (value : Value) : Check Fields := do
  let fields ← objectAt value "personal epoch quality policy mapping"
  ensure (fieldKeys fields ⊆ fieldKeys personalPolicyDefaults) "personal epoch quality known policy kwargs"
  let fields := Fields.ofList ((fieldKeys personalPolicyDefaults).map fun key =>
    (key,(fields.get key).getD (fieldDefault personalPolicyDefaults key)))
  ensure (fieldDefault fields "method"=.text "personal_epoch_signal_evidence") "personal policy method"
  let multiplier ← numeric (fieldDefault fields "diagnostic_mad_z") "finite personal diagnostic multiplier"
  ensure (0 < multiplier) "positive personal diagnostic multiplier"
  let minimum ← integer (fieldDefault fields "minimum_group_epochs") "personal exact integer minimum group epochs"
  ensure (2 ≤ minimum) "personal minimum group epochs at least two"
  let fraction ← numeric (fieldDefault fields "max_bad_channel_fraction") "finite personal bad channel fraction"
  ensure (0 ≤ fraction ∧ fraction < 1) "personal bad channel fraction interval"
  pure fields

structure PersonalQuality where
  policy : Fields
  lower : ArrayImage Atom
  upper : ArrayImage Atom
  counts : ArrayImage Atom
  observed : ArrayImage Atom
  channels : ℕ
  fitEpochs : Value
  fitGroups : Value
  fitSubjects : Value
  groupSource : Value
  fitDigest : Value
  provenance : Value

def supportedDiagnosticCell (count lower upper : Atom) : Prop :=
  match count with
  | .integer z => if 0 < z then
      ∃ lo hi, atomNumber lower=some lo ∧ atomNumber upper=some hi ∧ lo ≤ hi
    else lower=.nan ∧ upper=.nan
  | _ => False

def checkDiagnosticCells : List Atom → List Atom → List Atom → Check Unit
  | [],[],[] => .ok ()
  | count :: counts,lo :: lower,hi :: upper => do
    ensure (supportedDiagnosticCell count lo hi) "personal diagnostic envelope declared support"
    checkDiagnosticCells counts lower upper
  | _,_,_ => .error (.fieldFailure "personal diagnostic cell alignment")

def constructPersonalQuality (p : Primitives) (value : Value) : Check PersonalQuality := do
  let fields ← objectAt value "personal epoch quality record mapping"
  ensure (fieldDefault fields "schema"=.text "n2p3_personal_epoch_quality/1" ∧
    fieldDefault fields "method"=.text "personal_epoch_signal_evidence") "personal epoch quality schema method"
  -- Source constructs the policy here but invokes its validator in record().
  let policyValue ← required fields "policy"
  let policyFields ← objectAt policyValue "personal epoch quality policy mapping"
  ensure (fieldKeys policyFields ⊆ fieldKeys personalPolicyDefaults) "personal epoch quality constructor kwargs"
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
  let policy ← readPersonalPolicy policyValue
  let channels ← match observed.shape with
    | [] => .error (.fieldFailure "len personal observed mask")
    | channels :: _ => .ok channels
  let fitEpochCount ← numeric fitEpochs "personal comparable fit epoch count"
  ensure (0 < channels ∧ 2 ≤ fitEpochCount ∧ observed.dtype=.boolean ∧ observed.shape=[channels] ∧
    lower.shape=[channels,3] ∧ upper.shape=[channels,3] ∧ counts.shape=[channels,3] ∧ counts.dtype=.integer)
    "personal quality geometry and support dtype"
  ensure (∀ count ∈ counts.values, ∃ z, count=.integer z ∧ 0 ≤ z) "nonnegative personal group support"
  checkDiagnosticCells counts.values lower.values upper.values
  ensure (PyTruthy groups ∧ PyTruthy subjects ∧ PyTruthy fitDigest ∧
    (groupSource=.text "subject_ids" ∨ groupSource=.text "explicit_fit_groups")) "personal quality fit provenance"
  pure ⟨policy,lower,upper,counts,observed,channels,fitEpochs,groups,subjects,groupSource,fitDigest,provenance⟩

def finiteFloatJSON : Atom → Value
  | .real r => .number r
  | .integer z => .number z
  | .boolean b => .number (if b then 1 else 0)
  | _ => .null

def rowsOfThree : List Atom → List Value
  | a :: b :: c :: tail => .sequence (valuesOfList [finiteFloatJSON a,finiteFloatJSON b,finiteFloatJSON c]) :: rowsOfThree tail
  | _ => []

def personalQualityRecord (p : Primitives) (model : PersonalQuality) : Check Fields := do
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

def personalQualityFromRecord (p : Primitives) (value : Value) : Check PersonalQuality := do
  let model ← constructPersonalQuality p value
  let record ← personalQualityRecord p model
  let actual ← p.models.recovery.input.execution.semanticHash (.object record)
  let saved ← p.models.recovery.input.execution.semanticHash value
  ensure (actual=saved) "personal quality canonical semantic digest"
  pure model

def foldPolicyDefaults : Fields := Fields.ofList
  [("candidate_quantiles",.sequence (valuesOfList [.number (9/10),.number (95/100),.number (975/1000),.number (99/100)])),
   ("candidate_bad_channel_fractions",.sequence (valuesOfList [.number (1/8),.number (1/4),.number (3/8),.number (1/2)])),
   ("flat_quantile",.number (5/1000)),("max_bad_channel_fraction",.null),("global_scale_mad_z",.number 6),
   ("min_training_epoch_retention",.number (7/10)),("cv_splits",.integer 5),("min_clean_epochs",.integer 4),
   ("exclude_unusable_test_epochs",.boolean false)]

def numericValues : List Value → Check (List ℝ)
  | [] => .ok []
  | value :: rest => do
    let number ← numeric value "numeric fold policy value"
    let tail ← numericValues rest
    pure (number :: tail)

def readFoldPolicy (p : Primitives) (value : Value) : Check Fields := do
  let fields ← objectAt value "fold artifact policy mapping"
  let quantilesValue ← required fields "candidate_quantiles"
  let quantileTuple ← tupleJSON quantilesValue
  let fractionsValue ← required fields "candidate_bad_channel_fractions"
  let fractionTuple ← tupleJSON fractionsValue
  ensure (fieldKeys fields ⊆ fieldKeys foldPolicyDefaults) "fold policy known kwargs"
  let fields := Fields.ofList ((fieldKeys foldPolicyDefaults).map fun key =>
    (key,if key="candidate_quantiles" then quantileTuple else if key="candidate_bad_channel_fractions" then
      fractionTuple else (fields.get key).getD (fieldDefault foldPolicyDefaults key)))
  let quantiles ← tupleValue quantileTuple
  let quantiles ← numericValues quantiles
  ensure (quantiles ≠ [] ∧ ∀ q ∈ quantiles, 1/2 < q ∧ q < 1) "fold quantile interval"
  ensure (quantiles=quantiles.dedup.mergeSort (· ≤ ·)) "fold quantiles sorted unique"
  let flat ← numeric (fieldDefault fields "flat_quantile") "fold flat quantile numeric"
  ensure (0 ≤ flat ∧ flat < 1/2) "fold flat quantile interval"
  let kappas ← if fieldDefault fields "max_bad_channel_fraction"=.null then do
    let values ← tupleValue fractionTuple
    let floats ← values.mapM (readFiniteFloat p.models.recovery)
    pure floats
    else do
      let number ← readFiniteFloat p.models.recovery (fieldDefault fields "max_bad_channel_fraction")
      pure [number]
  ensure (kappas ≠ [] ∧ ∀ k ∈ kappas, 0 ≤ k ∧ k < 1) "fold candidate bad channel fraction interval"
  ensure (kappas=kappas.dedup.mergeSort (· ≤ ·)) "fold bad channel fractions sorted unique"
  if fieldDefault fields "max_bad_channel_fraction" ≠ .null then
    let maximum ← numeric (fieldDefault fields "max_bad_channel_fraction") "fold maximum bad fraction numeric"
    ensure (0 ≤ maximum ∧ maximum < 1) "fold maximum bad fraction interval"
  let z ← numeric (fieldDefault fields "global_scale_mad_z") "finite fold scale MAD multiplier"
  ensure (0 < z) "positive fold scale MAD multiplier"
  let retained ← numeric (fieldDefault fields "min_training_epoch_retention") "fold retained fraction numeric"
  ensure (0 < retained ∧ retained ≤ 1) "fold retained fraction interval"
  let splits ← numeric (fieldDefault fields "cv_splits") "fold comparable CV splits"
  ensure (2 ≤ splits) "fold CV at least two splits"
  let minimum ← numeric (fieldDefault fields "min_clean_epochs") "fold comparable clean epoch minimum"
  ensure (1 ≤ minimum) "fold clean epoch minimum positive"
  pure fields

structure FoldQuality where
  policy : Fields
  ptp : List Atom
  flat : List Atom
  selectedQuantiles : ArrayImage Atom
  badFraction : ℝ
  logCenter : ℝ
  logScale : ℝ
  fitEpochs : ℤ
  groups : List String
  subjects : List String
  groupSource : String
  cvEvidence : Fields
  fitDigest : Value
  provenance : Fields

def originalNulls (value : Value) : Check (List Bool) := do
  let values ← tupleValue value
  pure (values.map fun value => decide (value=.null))

def maskedThresholds (ptp flat mask : List Atom) (ptpNull flatNull : List Bool) : Check (List Atom × List Atom) :=
  match ptp,flat,mask,ptpNull,flatNull with
  | [],[],[],[],[] => .ok ([],[])
  | a :: aa,b :: bb,.boolean measured :: mm,isNullA :: nn,isNullB :: oo => do
    ensure (isNullA= !measured ∧ isNullB= !measured ∧
      (measured=true → finiteNumber a=true ∧ finiteNumber b=true)) "fold threshold nulls exactly absent channels"
    let (tailA,tailB) ← maskedThresholds aa bb mm nn oo
    pure ((if measured then a else .positiveInfinity)::tailA,(if measured then b else .negativeInfinity)::tailB)
  | _,_,_,_,_ => .error (.fieldFailure "fold threshold mask alignment")

def validThresholdPair (ptp flat : Atom) : Prop :=
  ((finiteNumber ptp=true ∧ finiteNumber flat=true) ∨ (ptp=.positiveInfinity ∧ flat=.negativeInfinity)) ∧
  (ptp=.positiveInfinity ∨ ∃ r, atomNumber ptp=some r ∧ 0 ≤ r)

def thresholdPairs : List Atom → List Atom → Prop
  | [],[] => True
  | a::aa,b::bb => validThresholdPair a b ∧ thresholdPairs aa bb
  | _,_ => False

def pythonStrings (p : Primitives) : List Value → Check (List String)
  | [] => .ok []
  | value :: rest => do
    let rendered ← p.models.recovery.input.pythonString value
    let tail ← pythonStrings p rest
    pure (rendered :: tail)

def foldQualityFromRecord (p : Primitives) (value : Value) : Check FoldQuality := do
  let fields ← objectAt value "fold artifact record dictionary"
  ensure (fieldDefault fields "method"=.text "fold_local_ptp_cv" ∧
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
      ensure (observed.dtype=.boolean ∧ observed.shape.length=1 ∧ observed.shape=ptp.shape ∧ observed.shape=flat.shape)
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
  ensure (ptp.shape.length=1 ∧ flat.shape=ptp.shape ∧ selected.shape=ptp.shape ∧ ptp.values ≠ [])
    "fold artifact threshold geometry"
  ensure (thresholdPairs ptpValues flatValues) "fold artifact finite or absent thresholds"
  ensure (finiteArray selected ∧ ∀ value ∈ selected.values, ∃ r, atomNumber value=some r ∧ 1/2 < r ∧ r < 1)
    "fold artifact selected quantiles interval"
  ensure (0 ≤ fraction ∧ fraction < 1 ∧ 0 < scale ∧ 1 ≤ fitEpochs ∧ groups ≠ [] ∧
    (groupSource="subject_ids" ∨ groupSource="explicit_fit_groups")) "fold artifact fitted scalar geometry"
  ensure (digest=.null ∨ ∃ s, digest=.text s ∧ s.length=64 ∧ ∀ char ∈ s.toList, char ∈ "0123456789abcdef".toList)
    "fold artifact optional lowercase SHA digest"
  pure ⟨policy,ptpValues,flatValues,selected,fraction,center,scale,fitEpochs,groups,subjects,groupSource,cv,digest,provenance⟩

inductive EpochQuality where
  | personal : PersonalQuality → EpochQuality
  | fold : FoldQuality → EpochQuality

def epochQualityFromRecord (p : Primitives) (value : Value) : Check EpochQuality := do
  let fields ← objectAt value "epoch quality record dictionary"
  if fieldDefault fields "method"=.text "personal_epoch_signal_evidence" then
    (personalQualityFromRecord p value).map EpochQuality.personal
  else if fieldDefault fields "method"=.text "fold_local_ptp_cv" then
    (foldQualityFromRecord p value).map EpochQuality.fold
  else .error (.sourceReject "unknown epoch quality method")

def epochQualityChannels : EpochQuality → ℕ
  | .personal model => model.channels
  | .fold model => model.ptp.length

theorem unsupported_diagnostic_coordinate_uses_nan : supportedDiagnosticCell (.integer 0) .nan .nan := by
  simp [supportedDiagnosticCell]

theorem unsupported_coordinate_does_not_accept_finite_placeholder :
    ¬supportedDiagnosticCell (.integer 0) (.real 0) (.real 0) := by
  simp [supportedDiagnosticCell]

theorem negative_group_support_is_refused (z : ℤ) (h : z < 0) :
    ¬(∃ n, Atom.integer z=.integer n ∧ 0 ≤ n) := by
  rintro ⟨n,hn,hpos⟩
  cases hn
  omega

theorem absent_legacy_threshold_is_a_paired_sentinel :
    validThresholdPair .positiveInfinity .negativeInfinity := by
  simp [validThresholdPair,finiteNumber]

theorem unpaired_legacy_infinity_is_refused : ¬validThresholdPair .positiveInfinity (.real 0) := by
  simp [validThresholdPair,finiteNumber]

end

end N2P3.PersonalERPQuality
