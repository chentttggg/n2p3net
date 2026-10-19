import N2P3.FrozenState
import Lean.Elab.Tactic.Omega

/-!
# The actual constructor ledger logic over decoded finite semantic values

Source: continuous_decision.py:162-188 and noise_model.py:37-61,117-134.
This consumes FrozenState's immutable Value/Fields ADT on its parsed unique-key
object boundary. It does not refine JSON/NPZ parsing, float/NumPy scalar tags,
np.isfinite/ufunc broadcasting or exception behavior on array-valued numerics,
oversized integer-to-NumPy conversions, IEEE NaN/infinity/rounding, covariance
PSD/tolerance/Cholesky checks, immutable-byte ownership, I/O or fit authority.
All numeric values here are decoded exact finite integers or real scalars.

`sourceReject` models an explicit source validation refusal. `fieldFailure`
distinguishes missing fields, noniterability, wrong structural tags and numeric
operations outside the declared scalar domain. It is not a claim that every
such case raises one identical Python exception class. In particular source
JSON rejects nonfinite numbers before these ledger branches are reached.

Python truthiness, exact int/bool tags, the 29-codepoint Python whitespace
table, raw-string uniqueness and exact key SETS are defined explicitly below.
No normalized textual-reference rule is added to analytic_reference, no bool
ridge refusal is invented, and constructor acceptance is never assumed to
imply that scoring can traverse the retained records value.
-/

namespace N2P3.StateLedger

open N2P3.FrozenState

noncomputable section

attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def PyTruthy : Value → Prop
  | .null => False
  | .boolean b => b = true
  | .integer z => z ≠ 0
  | .number r => r ≠ 0
  | .text s => s ≠ ""
  | .sequence .nil => False
  | .sequence (.cons _ _) => True
  | .object .nil => False
  | .object (.cons _ _ _) => True

def ExactInt : Value → Prop
  | .integer _ => True
  | _ => False

def ExactBool : Value → Prop
  | .boolean _ => True
  | _ => False

def numericScalar : Value → Option ℝ
  | .boolean true => some 1
  | .boolean false => some 0
  | .integer z => some z
  | .number r => some r
  | _ => none

/-- Python 3.14.3 str.isspace code points, independently read on the project
runtime. Lean's own whitespace predicate is not substituted for Python's. -/
def pythonWhitespaceCodes : List ℕ :=
  [9, 10, 11, 12, 13, 28, 29, 30, 31, 32, 133, 160, 5760,
    8192, 8193, 8194, 8195, 8196, 8197, 8198, 8199, 8200, 8201, 8202,
    8232, 8233, 8239, 8287, 12288]

def Nonblank (s : String) : Prop :=
  s.toList.any (fun c => !(pythonWhitespaceCodes.contains c.toNat)) = true

def LowerHexChar (c : Char) : Bool :=
  (Nat.ble 48 c.toNat && Nat.ble c.toNat 57) || (Nat.ble 97 c.toNat && Nat.ble c.toNat 102)

def LowerHex64 (s : String) : Prop := s.toList.length = 64 ∧ s.toList.all LowerHexChar = true

def fieldKeys : Fields → List String
  | .nil => []
  | .cons k _ tail => k :: fieldKeys tail

/-- Order and duplicate occurrences are ignored by set(provenance). Parsed
Python dictionaries already have unique keys; no new constructor rule is added. -/
def ExactKeys (fs : Fields) (expected : List String) : Prop :=
  (fieldKeys fs ⊆ expected) ∧ (expected ⊆ fieldKeys fs)

def valuesOfList : List Value → Values
  | [] => .nil
  | v :: tail => .cons v (valuesOfList tail)

def valuesToList : Values → List Value
  | .nil => []
  | .cons v tail => v :: valuesToList tail

def fieldDefault (fs : Fields) (k : String) : Value := (fs.get k).getD .null

inductive LedgerFault where
  | sourceReject (stage : String)
  | fieldFailure (operation : String)
  deriving DecidableEq

abbrev Check (α : Type*) := Except LedgerFault α

def ensure (p : Prop) (stage : String) : Check Unit := by
  classical
  exact if p then .ok () else .error (.sourceReject stage)

def required (fs : Fields) (k : String) : Check Value :=
  match fs.get k with
  | none => .error (.fieldFailure ("missing:" ++ k))
  | some v => .ok v

def requireDigest (v : Value) (stage : String) : Check Unit :=
  match v with
  | .text s => ensure (LowerHex64 s) stage
  | _ => .error (.sourceReject stage)

def Identifiers (ids : List String) : Prop :=
  ids ≠ [] ∧ (∀ s ∈ ids, Nonblank s) ∧ ids.Nodup

def requireIdentifiers (v : Value) (stage : String) : Check (List String) :=
  match v with
  | .sequence xs => match decodeTexts xs with
    | none => .error (.sourceReject stage)
    | some ids => do
      ensure (Identifiers ids) stage
      pure ids
  | _ => .error (.sourceReject stage)

def allowedFitScope (v : Value) (stage : String) : Check Unit :=
  match v with
  | .text s => ensure (s = "source_training" ∨ s = "target_calibration") stage
  | .sequence _ | .object _ => .error (.fieldFailure "unhashable_fit_scope")
  | _ => .error (.sourceReject stage)

def analyticKeys := ["fit_scope", "reference_id", "records"]
def recordKeys := ["record_id", "n_samples", "n_events", "n_targets", "signal_sha256",
  "input_sha256", "right_censored", "right_censored_events"]
def jointKeys := ["fit_scope", "scope_id", "draw_ids", "n_draws", "input_sha256",
  "shrinkage", "implementation_sha256", "estimator"]

/-- This is the source's first list comprehension, BEFORE per-record keysets. -/
def collectRecordIds : List Value → Check (List Value)
  | [] => .ok []
  | .object f :: tail => do
    let id ← required f "record_id"
    let ids ← collectRecordIds tail
    pure (id :: ids)
  | _ :: _ => .error (.fieldFailure "record_id_subscription_type")

/-- For strings/dicts, a nonempty iterator yields non-dictionary elements;
empty iterators safely produce an empty comprehension. None/bool/numbers do
not have an iterator, even when they are falsy. -/
def scoreRecordIds (records : Value) : Check (List Value) :=
  match records with
  | .sequence xs => collectRecordIds (valuesToList xs)
  | .text s => if s = "" then .ok [] else .error (.fieldFailure "record_id_subscription_type")
  | .object .nil => .ok []
  | .object (.cons _ _ _) => .error (.fieldFailure "record_id_subscription_type")
  | _ => .error (.fieldFailure "records_not_iterable")

/-- Only record iteration/subscription readiness. The source's subsequent set
construction also needs hashable IDs; full numerical score readiness is not
defined here. Accepted training ledgers separately enforce textual IDs. -/
def RecordIdTraversalReady (provenance : Value) : Prop :=
  match provenance with
  | .object f => ∃ ids, (do scoreRecordIds (← required f "records")) = .ok ids
  | _ => False

def checkRecord (basisWidth : ℕ) (record : Value) : Check Unit := do
  let f ← match record with
    | .object f => .ok f
    | _ => .error (.fieldFailure "record_mapping")
  ensure (ExactKeys f recordKeys) "continuous_record_keys"
  let ns ← required f "n_samples"
  let ne ← required f "n_events"
  let nt ← required f "n_targets"
  let events ← match ns, ne, nt with
    | .integer ns, .integer ne, .integer nt => do
      ensure ((basisWidth : ℤ) + 3 ≤ ns ∧ 1 ≤ ne ∧ ne ≤ ns ∧ 0 ≤ nt ∧ nt ≤ ne)
        "continuous_record_counts"
      pure ne
    | _, _, _ => .error (.sourceReject "continuous_record_counts")
  let rc ← required f "right_censored"
  let nc ← required f "right_censored_events"
  match rc, nc with
  | .boolean flag, .integer n =>
    ensure (0 ≤ n ∧ n ≤ events ∧ (flag = true ∨ n = 0)) "continuous_record_censor"
  | _, _ => .error (.sourceReject "continuous_record_censor")
  requireDigest (← required f "signal_sha256") "continuous_signal_sha256"
  requireDigest (← required f "input_sha256") "continuous_input_sha256"

def checkRecords (basisWidth : ℕ) : List Value → Check Unit
  | [] => .ok ()
  | record :: tail => do
    checkRecord basisWidth record
    checkRecords basisWidth tail

def trainingRecords (records : Value) : Check (List Value) :=
  match records with
  | .sequence xs => .ok (valuesToList xs)
  | .text s => if s = "" then .ok [] else .error (.fieldFailure "record_id_subscription_type")
  | .object .nil => .ok []
  | .object (.cons _ _ _) => .error (.fieldFailure "record_id_subscription_type")
  | _ => .error (.fieldFailure "records_not_iterable")

def checkRidge (f : Fields) : Check Unit :=
  match f.get "ridge" with
  | none => .error (.sourceReject "continuous_ridge")
  | some v => match numericScalar v with
    | none => .error (.fieldFailure "ridge_numeric_scalar")
    | some r => ensure (0 < r) "continuous_ridge"

def checkContinuousFields (basisWidth : ℕ) (f : Fields) : Check Unit := by
  classical
  exact do
    if fieldDefault f "fit_scope" = .text "analytic_reference" then
      ensure (ExactKeys f analyticKeys) "continuous_analytic_keys"
      let records ← required f "records"
      ensure (¬PyTruthy records) "continuous_analytic_records"
      let reference ← required f "reference_id"
      ensure (PyTruthy reference) "continuous_analytic_reference"
    else
      allowedFitScope (fieldDefault f "fit_scope") "continuous_scope_or_records"
      let records := fieldDefault f "records"
      ensure (PyTruthy records) "continuous_scope_or_records"
      let rows ← trainingRecords records
      let ids ← collectRecordIds rows
      let _ ← requireIdentifiers (.sequence (valuesOfList ids)) "continuous_record_ids"
      checkRecords basisWidth rows
      requireDigest (fieldDefault f "implementation_sha256") "continuous_implementation_sha256"
      checkRidge f

def checkContinuous (basisWidth : ℕ) (provenance : Value) : Check Unit :=
  match provenance with
  | .object f => checkContinuousFields basisWidth f
  | _ => .error (.fieldFailure "provenance_get")

def checkShrinkage (v : Value) : Check Unit :=
  match v with
  | .boolean _ => .error (.sourceReject "joint_shrinkage")
  | _ => match numericScalar v with
    | none => .error (.fieldFailure "shrinkage_numeric_scalar")
    | some r => ensure (0 ≤ r ∧ r ≤ 1) "joint_shrinkage"

def checkJoint (provenance : Value) : Check Unit := do
  let f ← match provenance with
    | .object f => .ok f
    | _ => .error (.sourceReject "joint_header")
  ensure (ExactKeys f jointKeys) "joint_header"
  allowedFitScope (fieldDefault f "fit_scope") "joint_header"
  match (← required f "scope_id") with
  | .text s => ensure (Nonblank s) "joint_header"
  | _ => .error (.sourceReject "joint_header")
  ensure (fieldDefault f "estimator" = .text "joint_standardized_covariance_with_cross_terms") "joint_header"
  let ids ← requireIdentifiers (← required f "draw_ids") "joint_draw_ids"
  match (← required f "n_draws") with
  | .integer n => ensure (n = (ids.length : ℤ) ∧ 3 ≤ n) "joint_draw_count"
  | _ => .error (.sourceReject "joint_draw_count")
  requireDigest (← required f "input_sha256") "joint_input_sha256"
  requireDigest (← required f "implementation_sha256") "joint_implementation_sha256"
  checkShrinkage (← required f "shrinkage")

theorem values_list_roundtrip (xs : List Value) : valuesToList (valuesOfList xs) = xs := by
  induction xs with
  | nil => rfl
  | cons v tail ih => simp [valuesOfList, valuesToList, ih]

theorem text_values_to_list (xs : List String) :
    valuesToList (encodeTexts xs) = xs.map Value.text := by
  induction xs with
  | nil => rfl
  | cons v tail ih => simp [encodeTexts, valuesToList, ih]

theorem text_list_encoding (ids : List String) :
    valuesOfList (ids.map Value.text) = encodeTexts ids := by
  induction ids with
  | nil => rfl
  | cons v tail ih => simp [valuesOfList, encodeTexts, ih]

theorem sequence_truthy_iff_nonempty (xs : List Value) :
    PyTruthy (.sequence (valuesOfList xs)) ↔ xs ≠ [] := by
  cases xs <;> simp [valuesOfList, PyTruthy]

theorem decoded_texts_preserve_exact_tags (xs : Values) (ids : List String)
    (h : decodeTexts xs = some ids) : xs = encodeTexts ids := by
  cases xs with
  | nil =>
    have he : [] = ids := Option.some.inj h
    subst ids
    rfl
  | cons v tail =>
    cases v <;> try { solve | simp [decodeTexts] at h }
    rename_i s
    cases ht : decodeTexts tail with
    | none => simp [decodeTexts, ht] at h
    | some rest =>
      have he : s :: rest = ids := by simpa [decodeTexts, ht] using h
      subst ids
      rw [encodeTexts, decoded_texts_preserve_exact_tags tail rest ht]
termination_by sizeOf xs

theorem encoded_texts_injective : Function.Injective encodeTexts := by
  intro a b h
  have hd := congrArg decodeTexts h
  simpa only [decode_encode_texts, Option.some.injEq] using hd

theorem ensure_ok_iff (p : Prop) (stage : String) : ensure p stage = .ok () ↔ p := by
  unfold ensure
  split_ifs <;> simp_all

theorem digest_ok_iff (v : Value) (stage : String) :
    requireDigest v stage = .ok () ↔ ∃ s, v = .text s ∧ LowerHex64 s := by
  cases v <;> simp [requireDigest, ensure_ok_iff]

theorem identifiers_ok_iff (v : Value) (stage : String) (ids : List String) :
    requireIdentifiers v stage = .ok ids ↔ v = .sequence (encodeTexts ids) ∧ Identifiers ids := by
  classical
  cases v <;> try { solve | simp [requireIdentifiers] }
  rename_i xs
  cases hd : decodeTexts xs with
  | none =>
    constructor
    · intro h
      simp [requireIdentifiers, hd] at h
    · rintro ⟨he, _⟩
      have hx := Value.sequence.inj he
      rw [hx, decode_encode_texts] at hd
      cases hd
  | some parsed =>
    have hx := decoded_texts_preserve_exact_tags xs parsed hd
    subst xs
    have he : encodeTexts parsed = encodeTexts ids ↔ parsed = ids :=
      ⟨fun h => encoded_texts_injective h, congrArg encodeTexts⟩
    by_cases hp : Identifiers parsed
    · simp [requireIdentifiers, decode_encode_texts, ensure, hp, he]
      exact fun h => h ▸ hp
    · simp [requireIdentifiers, decode_encode_texts, ensure, hp, he]
      exact fun h => h ▸ hp

theorem boolean_is_not_exact_integer (b : Bool) : ¬ExactInt (.boolean b) := by simp [ExactInt]

theorem integer_is_not_exact_boolean (n : ℤ) : ¬ExactBool (.integer n) := by simp [ExactBool]

def analyticLedger (reference records : Value) : Value := .object (Fields.ofList
  [("fit_scope", .text "analytic_reference"), ("reference_id", reference), ("records", records)])

/-- This exactly preserves the broad public constructor domain: arbitrary
falsy records and arbitrary truthy references, with no textual-ID assumption. -/
theorem analytic_accepted_iff (width : ℕ) (reference records : Value) :
    checkContinuous width (analyticLedger reference records) = .ok () ↔
      ¬PyTruthy records ∧ PyTruthy reference := by
  classical
  simp [checkContinuous, checkContinuousFields, analyticLedger, fieldDefault,
    Fields.ofList, Fields.get, ExactKeys, fieldKeys, analyticKeys, required, ensure]
  split_ifs <;> simp_all

theorem falsy_records_truthy_reference_accepted (width : ℕ) (reference records : Value)
    (hr : ¬PyTruthy records) (hi : PyTruthy reference) :
    checkContinuous width (analyticLedger reference records) = .ok () :=
  (analytic_accepted_iff _ _ _).mpr ⟨hr, hi⟩

theorem none_false_zero_are_accepted (width : ℕ) :
    checkContinuous width (analyticLedger (.boolean true) .null) = .ok () ∧
    checkContinuous width (analyticLedger (.boolean true) (.boolean false)) = .ok () ∧
    checkContinuous width (analyticLedger (.boolean true) (.integer 0)) = .ok () ∧
    checkContinuous width (analyticLedger (.boolean true) (.number 0)) = .ok () := by
  simp only [analytic_accepted_iff, PyTruthy]
  simp

theorem none_false_zero_fail_record_iteration :
    ¬RecordIdTraversalReady (analyticLedger (.boolean true) .null) ∧
    ¬RecordIdTraversalReady (analyticLedger (.boolean true) (.boolean false)) ∧
    ¬RecordIdTraversalReady (analyticLedger (.boolean true) (.integer 0)) ∧
    ¬RecordIdTraversalReady (analyticLedger (.boolean true) (.number 0)) := by
  simp [RecordIdTraversalReady, analyticLedger, Fields.ofList, Fields.get, required, scoreRecordIds]

theorem empty_iterables_are_ledger_ready :
    RecordIdTraversalReady (analyticLedger (.text " ") (.object .nil)) ∧
    RecordIdTraversalReady (analyticLedger (.text " ") (.text "")) ∧
    RecordIdTraversalReady (analyticLedger (.text " ") (.sequence .nil)) := by
  simp [RecordIdTraversalReady, analyticLedger, Fields.ofList, Fields.get, required,
    scoreRecordIds, valuesToList, collectRecordIds]

theorem nontextual_and_blank_analytic_references_allowed (width : ℕ) :
    checkContinuous width (analyticLedger (.boolean true) (.sequence .nil)) = .ok () ∧
    checkContinuous width (analyticLedger (.text " ") (.sequence .nil)) = .ok () := by
  simp only [analytic_accepted_iff, PyTruthy]
  decide

theorem zero_digest_has_lowercase_hex_format : LowerHex64 zeroDigest := by
  constructor <;> rfl

theorem raw_identifier_spacing_is_preserved :
    requireIdentifiers (.sequence (encodeTexts ["id", " id"])) "ids" = .ok ["id", " id"] := by
  rw [identifiers_ok_iff]
  constructor
  · rfl
  · have h0 : Nonblank "id" := rfl
    have h1 : Nonblank " id" := rfl
    refine ⟨by decide, ?_, by decide⟩
    intro s hs
    rcases List.mem_cons.mp hs with hs | hs
    · subst s; exact h0
    · have he := List.mem_singleton.mp hs
      subst s
      exact h1

structure RecordData where
  recordId : String
  samples : ℤ
  events : ℤ
  targets : ℤ
  rightCensored : Bool
  censoredEvents : ℤ
  signalSHA : String
  inputSHA : String

def encodeRecord (r : RecordData) : Value := .object (Fields.ofList
  [("record_id", .text r.recordId), ("n_samples", .integer r.samples),
    ("n_events", .integer r.events), ("n_targets", .integer r.targets),
    ("signal_sha256", .text r.signalSHA), ("input_sha256", .text r.inputSHA),
    ("right_censored", .boolean r.rightCensored), ("right_censored_events", .integer r.censoredEvents)])

def RecordBounds (width : ℕ) (r : RecordData) : Prop :=
  ((width : ℤ) + 3 ≤ r.samples ∧ 1 ≤ r.events ∧ r.events ≤ r.samples ∧
    0 ≤ r.targets ∧ r.targets ≤ r.events) ∧
  (0 ≤ r.censoredEvents ∧ r.censoredEvents ≤ r.events ∧
    (r.rightCensored = true ∨ r.censoredEvents = 0)) ∧
  LowerHex64 r.signalSHA ∧ LowerHex64 r.inputSHA

theorem record_check_exact_conditions (width : ℕ) (r : RecordData) :
    checkRecord width (encodeRecord r) = .ok () ↔ RecordBounds width r := by
  classical
  simp [checkRecord, encodeRecord, Fields.ofList, Fields.get, required,
    ExactKeys, fieldKeys, recordKeys, requireDigest, ensure, RecordBounds]
  split_ifs <;> simp_all

inductive FitScope where
  | sourceTraining
  | targetCalibration
  deriving DecidableEq

def FitScope.text : FitScope → String
  | .sourceTraining => "source_training"
  | .targetCalibration => "target_calibration"

/-- Only the actual producer's constructor-relevant projection is encoded.
The source emits extra diagnostics; their production and fitting authenticity
are not certified by the type/tag normalization theorem below. -/
def producedContinuous (scope : FitScope) (records : List RecordData)
    (implementation : String) (ridge : ℝ) : Value := .object (Fields.ofList
  [("fit_scope", .text scope.text), ("records", .sequence (valuesOfList (records.map encodeRecord))),
    ("implementation_sha256", .text implementation), ("ridge", .number ridge)])

def publicTraining (scope : FitScope) (records : List Value)
    (implementation : String) (ridge : Value) : Value := .object (Fields.ofList
  [("fit_scope", .text scope.text), ("records", .sequence (valuesOfList records)),
    ("implementation_sha256", .text implementation), ("ridge", ridge)])

def jointLedger (scope : FitScope) (scopeId : String) (ids : List String)
    (inputSHA implementation : String) (shrinkage : Value) : Value := .object (Fields.ofList
  [("fit_scope", .text scope.text), ("scope_id", .text scopeId),
    ("draw_ids", .sequence (encodeTexts ids)), ("n_draws", .integer ids.length),
    ("input_sha256", .text inputSHA), ("shrinkage", shrinkage),
    ("implementation_sha256", .text implementation),
    ("estimator", .text "joint_standardized_covariance_with_cross_terms")])

def producedJoint (scope : FitScope) (scopeId : String) (ids : List String)
    (inputSHA implementation : String) (shrinkage : ℝ) : Value :=
  jointLedger scope scopeId ids inputSHA implementation (.number shrinkage)

theorem training_ridge_numeric_equivalence (width : ℕ) (scope : FitScope)
    (records : List Value) (implementation : String) (a b : Value)
    (h : numericScalar a = numericScalar b) :
    checkContinuous width (publicTraining scope records implementation a) =
      checkContinuous width (publicTraining scope records implementation b) := by
  cases scope <;>
    simp [checkContinuous, publicTraining, checkContinuousFields, fieldDefault,
      Fields.ofList, Fields.get, FitScope.text, checkRidge, h]

theorem encoded_record_ids (records : List RecordData) :
    collectRecordIds (records.map encodeRecord) =
      .ok ((records.map RecordData.recordId).map Value.text) := by
  induction records with
  | nil => rfl
  | cons r tail ih =>
    simp [collectRecordIds, encodeRecord, required, Fields.ofList, Fields.get, ih]

theorem records_check_iff_each (width : ℕ) (records : List Value) :
    checkRecords width records = .ok () ↔ ∀ r ∈ records, checkRecord width r = .ok () := by
  induction records with
  | nil => simp [checkRecords]
  | cons r tail ih =>
    cases hr : checkRecord width r with
    | error e => simp [checkRecords, hr]
    | ok result => cases result; simp [checkRecords, hr, ih]

theorem typed_records_check_iff_bounds (width : ℕ) (records : List RecordData) :
    checkRecords width (records.map encodeRecord) = .ok () ↔ ∀ r ∈ records, RecordBounds width r := by
  rw [records_check_iff_each]
  constructor
  · intro h r hr
    exact (record_check_exact_conditions width r).mp
      (h (encodeRecord r) (List.mem_map.mpr ⟨r, hr, rfl⟩))
  · intro h v hv
    obtain ⟨r, hr, rfl⟩ := List.mem_map.mp hv
    exact (record_check_exact_conditions width r).mpr (h r hr)

/-- The producer's canonical scalar tags and explicit record/ID/digest/ridge
constraints suffice for every constructor ledger check. No acceptance or
score-readiness conclusion is supplied as a premise. -/
theorem produced_continuous_contract_accepted (width : ℕ) (scope : FitScope)
    (records : List RecordData) (implementation : String) (ridge : ℝ)
    (hids : Identifiers (records.map RecordData.recordId))
    (hrecords : ∀ r ∈ records, RecordBounds width r)
    (himpl : LowerHex64 implementation) (hridge : 0 < ridge) :
    checkContinuous width (producedContinuous scope records implementation ridge) = .ok () := by
  have hnonempty : records ≠ [] := by
    intro h
    exact hids.1 (by simp [h])
  have hi := (identifiers_ok_iff
    (.sequence (encodeTexts (records.map RecordData.recordId)))
    "continuous_record_ids" (records.map RecordData.recordId)).mpr ⟨rfl, hids⟩
  have hr := (typed_records_check_iff_bounds width records).mpr hrecords
  have hc := encoded_record_ids records
  have he : valuesOfList (records.map (Value.text ∘ RecordData.recordId)) =
      encodeTexts (records.map RecordData.recordId) := by
    rw [← List.map_map, text_list_encoding]
  cases scope <;>
    simp [checkContinuous, producedContinuous, checkContinuousFields, fieldDefault,
      Fields.ofList, Fields.get, FitScope.text, allowedFitScope, ensure,
      sequence_truthy_iff_nonempty, hnonempty, trainingRecords, values_list_roundtrip,
      hc, he, hi, hr, requireDigest, himpl, checkRidge, numericScalar, hridge]

theorem joint_checks_before_shrinkage (scope : FitScope) (scopeId : String) (ids : List String)
    (inputSHA implementation : String) (shrinkage : Value)
    (hs : Nonblank scopeId) (hi : Identifiers ids) (hn : 3 ≤ ids.length)
    (hin : LowerHex64 inputSHA) (himpl : LowerHex64 implementation) :
    checkJoint (jointLedger scope scopeId ids inputSHA implementation shrinkage) = checkShrinkage shrinkage := by
  have hn' : (3 : ℤ) ≤ ids.length := by omega
  have hid := (identifiers_ok_iff (.sequence (encodeTexts ids)) "joint_draw_ids" ids).mpr ⟨rfl, hi⟩
  cases scope <;>
    simp [checkJoint, jointLedger, Fields.ofList, Fields.get, required, fieldDefault,
      ExactKeys, fieldKeys, jointKeys, FitScope.text, allowedFitScope, ensure,
      hs, hid, hn', requireDigest, hin, himpl]

theorem produced_joint_contract_accepted (scope : FitScope) (scopeId : String) (ids : List String)
    (inputSHA implementation : String) (shrinkage : ℝ)
    (hs : Nonblank scopeId) (hi : Identifiers ids) (hn : 3 ≤ ids.length)
    (hin : LowerHex64 inputSHA) (himpl : LowerHex64 implementation)
    (hshrink : 0 ≤ shrinkage ∧ shrinkage ≤ 1) :
    checkJoint (producedJoint scope scopeId ids inputSHA implementation shrinkage) = .ok () := by
  rw [producedJoint, joint_checks_before_shrinkage _ _ _ _ _ _ hs hi hn hin himpl]
  simp [checkShrinkage, numericScalar, ensure, hshrink]

theorem producer_ridge_is_a_real_tag (scope : FitScope) (records : List RecordData)
    (implementation : String) (ridge : ℝ) :
    (match producedContinuous scope records implementation ridge with
      | .object f => f.get "ridge" | _ => none) = some (.number ridge) := by
  simp [producedContinuous, Fields.ofList, Fields.get]

theorem producer_record_count_tags (r : RecordData) :
    (match encodeRecord r with | .object f => f.get "n_samples" | _ => none) = some (.integer r.samples) ∧
    (match encodeRecord r with | .object f => f.get "n_events" | _ => none) = some (.integer r.events) ∧
    (match encodeRecord r with | .object f => f.get "n_targets" | _ => none) = some (.integer r.targets) ∧
    (match encodeRecord r with | .object f => f.get "right_censored_events" | _ => none) =
      some (.integer r.censoredEvents) := by
  simp [encodeRecord, Fields.ofList, Fields.get]

def legalRecord : RecordData := ⟨"calibration-0", 20, 3, 1, false, 0, zeroDigest, zeroDigest⟩

theorem legal_record_meets_all_record_guards : checkRecord 3 (encodeRecord legalRecord) = .ok () := by
  rw [record_check_exact_conditions]
  norm_num [RecordBounds, legalRecord, zero_digest_has_lowercase_hex_format]

def validTraining : Value := publicTraining .targetCalibration [encodeRecord legalRecord] zeroDigest (.number 1)
def booleanRidgeTraining : Value := publicTraining .targetCalibration [encodeRecord legalRecord] zeroDigest (.boolean true)
def validJoint : Value := producedJoint .targetCalibration " noise scope "
  ["draw-0", " draw-0", "draw-2"] zeroDigest zeroDigest 1

theorem calibration_identifiers_valid : Identifiers ["calibration-0"] := by
  have h : Nonblank "calibration-0" := rfl
  exact ⟨by decide, by simpa only [List.forall_mem_singleton] using h, by decide⟩

theorem draw_identifiers_valid : Identifiers ["draw-0", " draw-0", "draw-2"] := by
  have h0 : Nonblank "draw-0" := rfl
  have h1 : Nonblank " draw-0" := rfl
  have h2 : Nonblank "draw-2" := rfl
  refine ⟨by decide, ?_, by decide⟩
  intro s hs
  rcases List.mem_cons.mp hs with h | h
  · subst s; exact h0
  · rcases List.mem_cons.mp h with h | h
    · subst s; exact h1
    · have he := List.mem_singleton.mp h
      subst s
      exact h2

theorem valid_training_ledger_accepted : checkContinuous 3 validTraining = .ok () := by
  change checkContinuous 3 (producedContinuous .targetCalibration [legalRecord] zeroDigest 1) = .ok ()
  apply produced_continuous_contract_accepted
  · exact calibration_identifiers_valid
  · intro r hr
    have he := List.mem_singleton.mp hr
    subst r
    exact (record_check_exact_conditions _ _).mp legal_record_meets_all_record_guards
  · exact zero_digest_has_lowercase_hex_format
  · norm_num

theorem boolean_ridge_training_accepted : checkContinuous 3 booleanRidgeTraining = .ok () := by
  have h := training_ridge_numeric_equivalence 3 .targetCalibration [encodeRecord legalRecord]
    zeroDigest (.boolean true) (.number 1) rfl
  exact h.trans valid_training_ledger_accepted

theorem valid_joint_ledger_accepted : checkJoint validJoint = .ok () := by
  apply produced_joint_contract_accepted
  · rfl
  · exact draw_identifiers_valid
  · decide
  · exact zero_digest_has_lowercase_hex_format
  · exact zero_digest_has_lowercase_hex_format
  · norm_num

def booleanCountRecord : Value := .object (Fields.ofList
  [("record_id", .text "calibration-0"), ("n_samples", .integer 20),
    ("n_events", .boolean true), ("n_targets", .integer 0),
    ("signal_sha256", .text zeroDigest), ("input_sha256", .text zeroDigest),
    ("right_censored", .boolean false), ("right_censored_events", .integer 0)])

theorem boolean_event_count_rejected :
    checkRecord 3 booleanCountRecord = .error (.sourceReject "continuous_record_counts") := by
  simp [checkRecord, booleanCountRecord, Fields.ofList, Fields.get, required,
    ExactKeys, fieldKeys, recordKeys, ensure]

theorem joint_boolean_shrinkage_rejected (b : Bool) :
    checkShrinkage (.boolean b) = .error (.sourceReject "joint_shrinkage") := rfl

def booleanShrinkageJoint : Value := jointLedger .targetCalibration " noise scope "
  ["draw-0", " draw-0", "draw-2"] zeroDigest zeroDigest (.boolean false)

theorem boolean_shrinkage_full_ledger_rejected :
    checkJoint booleanShrinkageJoint = .error (.sourceReject "joint_shrinkage") := by
  unfold booleanShrinkageJoint
  rw [joint_checks_before_shrinkage _ _ _ _ _ _ (by rfl) draw_identifiers_valid
    (by decide) zero_digest_has_lowercase_hex_format zero_digest_has_lowercase_hex_format]
  exact joint_boolean_shrinkage_rejected false

def wrongKeysThenMissingId : Value := publicTraining .targetCalibration
  [.object (Fields.ofList [("record_id", .text "id-0")]), .object .nil] zeroDigest (.number 1)

theorem record_id_collection_precedes_record_key_rejection :
    checkContinuous 3 wrongKeysThenMissingId = .error (.fieldFailure "missing:record_id") := by
  simp [checkContinuous, wrongKeysThenMissingId, publicTraining, checkContinuousFields,
    fieldDefault, Fields.ofList, Fields.get, FitScope.text, allowedFitScope, PyTruthy,
    ensure, trainingRecords, valuesToList, valuesOfList, collectRecordIds, required]

end

end N2P3.StateLedger
