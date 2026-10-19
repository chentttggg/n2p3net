import N2P3.PersonalState
import N2P3.PersonalLDA
import Mathlib.Data.List.OfFn

/-!
# Concrete named-tensor to waveform-LDA restoration

The input representation is PersonalState.NamedState: every tensor has its
actual name, dtype, shape and ordered List Value contents. Vector parsing checks
rank/length and independently reads every numeric element. Intercept parsing
checks one element and shape product one; it does not invent a scalar-rank gate.
The current producers use float64 for all four LDA tensors. The source does not
require that dtype globally, so the parser does not reject other numeric tensor
dtypes merely because they differ. Dtype is preserved in the enclosing packet.

The canonical packet has four concrete float64 tensors, an optional int64
waveform_channels tensor and arbitrary extra named tensors. Their original
packet seals and recursive copy are handled by the already proved PersonalState
loader. We then actually parse the four fields and positive-scale subconstructor,
and parse the selected-channel tensor, before invoking PersonalLDA's explicit
standardization/affine evaluator. No state/score roundtrip is assumed.

This is the decoded finite same-representation subroute. Independent metadata,
input/QC/time/participant gates and other classifier families remain outside it.
Metadata and named-state objects retain PersonalState's parsed-unique-key and
finite-scalar domain. The subconstructor is not the full CalibratedPredictor.
The waveform width equality is an explicit type-domain restriction, not a gate
that calibrated.py's constructor actually checks. SciPy, dtype casts, sklearn
rounding, Torch/NumPy byte codecs and hash authentication remain unproved.
-/

namespace N2P3.PersonalRestore

open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalState

noncomputable section

attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

/-- Each list element is inspected, including the empty-vector length case. -/
def decodeVector : (d : ℕ) → List Value → Option (PersonalLDA.Vector d)
  | 0, [] => some (fun j => Fin.elim0 j)
  | d + 1, value :: tail => do
      let head ← numericScalar value
      let rest ← decodeVector d tail
      pure (Fin.cons head rest)
  | _, _ => none

def tensorVector (d : ℕ) (tensor : Tensor) : Option (PersonalLDA.Vector d) :=
  if tensor.shape = [d] then decodeVector d tensor.values else none

def tensorScalar (tensor : Tensor) : Option ℝ :=
  if tensor.numel = 1 then
    match tensor.values with
    | [value] => numericScalar value
    | _ => none
  else none

def namedToRecord (d : ℕ) (state : NamedState) : PersonalLDA.Record d := fun field => do
  let tensor ← lookupTensor field.key state
  match field with
  | .intercept => PersonalLDA.Payload.scalar <$> tensorScalar tensor
  | _ => PersonalLDA.Payload.vector <$> tensorVector d tensor

def canonicalVector {d : ℕ} (values : PersonalLDA.Vector d) : Tensor :=
  ⟨"torch.float64", [d], List.ofFn (fun i => .number (values i))⟩

def canonicalScalar (shape : List ℕ) (value : ℝ) : Tensor :=
  ⟨"torch.float64", shape, [.number value]⟩

def canonicalNamed {d : ℕ} (state : PersonalLDA.State d) (interceptShape : List ℕ)
    (extras : NamedState) : NamedState :=
  [("feature_mean", canonicalVector state.featureMean),
   ("feature_scale", canonicalVector state.featureScale),
   ("coefficient", canonicalVector state.coefficient),
   ("intercept", canonicalScalar interceptShape state.intercept)] ++ extras

theorem decodeVector_ofFn {d : ℕ} (values : PersonalLDA.Vector d) :
    decodeVector d (List.ofFn (fun i => .number (values i))) = some values := by
  induction d with
  | zero =>
    simp only [List.ofFn_zero, decodeVector, Option.some.injEq]
    funext i
    exact Fin.elim0 i
  | succ d ih =>
    simp only [List.ofFn_succ, decodeVector, numericScalar, ih]
    apply congrArg some
    funext i
    refine Fin.cases ?_ (fun j => ?_) i <;> rfl

theorem canonical_vector_parses {d : ℕ} (values : PersonalLDA.Vector d) :
    tensorVector d (canonicalVector values) = some values := by
  simp [tensorVector, canonicalVector, decodeVector_ofFn]

theorem canonical_scalar_parses (shape : List ℕ) (value : ℝ)
    (hshape : shape.foldr (· * ·) 1 = 1) :
    tensorScalar (canonicalScalar shape value) = some value := by
  simp [tensorScalar, canonicalScalar, Tensor.numel, hshape, numericScalar]

theorem named_mean_parses {d : ℕ} (state : PersonalLDA.State d) (shape : List ℕ)
    (extras : NamedState) :
    PersonalLDA.readVector (namedToRecord d (canonicalNamed state shape extras)) .featureMean =
      some state.featureMean := by
  simp [PersonalLDA.readVector, namedToRecord, canonicalNamed, PersonalLDA.Field.key,
    lookupTensor, canonical_vector_parses]

theorem named_scale_parses {d : ℕ} (state : PersonalLDA.State d) (shape : List ℕ)
    (extras : NamedState) :
    PersonalLDA.readVector (namedToRecord d (canonicalNamed state shape extras)) .featureScale =
      some state.featureScale := by
  simp [PersonalLDA.readVector, namedToRecord, canonicalNamed, PersonalLDA.Field.key,
    lookupTensor, canonical_vector_parses]

theorem named_coefficient_parses {d : ℕ} (state : PersonalLDA.State d) (shape : List ℕ)
    (extras : NamedState) :
    PersonalLDA.readVector (namedToRecord d (canonicalNamed state shape extras)) .coefficient =
      some state.coefficient := by
  simp [PersonalLDA.readVector, namedToRecord, canonicalNamed, PersonalLDA.Field.key,
    lookupTensor, canonical_vector_parses]

theorem named_intercept_parses {d : ℕ} (state : PersonalLDA.State d) (shape : List ℕ)
    (extras : NamedState) (hshape : shape.foldr (· * ·) 1 = 1) :
    PersonalLDA.readScalar (namedToRecord d (canonicalNamed state shape extras)) .intercept =
      some state.intercept := by
  simp [PersonalLDA.readScalar, namedToRecord, canonicalNamed, PersonalLDA.Field.key,
    lookupTensor, canonical_scalar_parses shape _ hshape]

theorem canonical_named_readState {d : ℕ} (state : PersonalLDA.State d) (shape : List ℕ)
    (extras : NamedState) (hshape : shape.foldr (· * ·) 1 = 1) :
    PersonalLDA.readState (namedToRecord d (canonicalNamed state shape extras)) = some state := by
  simp only [PersonalLDA.readState, named_mean_parses, named_scale_parses,
    named_coefficient_parses, named_intercept_parses state shape extras hshape]
  rfl

theorem canonical_named_constructor {d : ℕ} (state : PersonalLDA.State d) (shape : List ℕ)
    (extras : NamedState) (hshape : shape.foldr (· * ·) 1 = 1)
    (hscale : PersonalLDA.ValidState state) :
    PersonalLDA.loadState (namedToRecord d (canonicalNamed state shape extras)) = some state := by
  simp [PersonalLDA.loadState, canonical_named_readState state shape extras hshape, hscale]

/-- Actual selected-channel tensor elements must be exact integers. -/
def decodeChannel (channels : ℕ) : Value → Option (Fin channels)
  | .integer z =>
    if h : 0 ≤ z ∧ z < (channels : ℤ) then some ⟨z.toNat, by omega⟩ else none
  | _ => none

def decodeChannels (channels : ℕ) : List Value → Option (List (Fin channels))
  | [] => some []
  | value :: tail => do
    let channel ← decodeChannel channels value
    let rest ← decodeChannels channels tail
    pure (channel :: rest)

def readSelection (channels : ℕ) (state : NamedState) : Option (PersonalLDA.Selection channels) := do
  match lookupTensor "waveform_channels" state with
  | none => some none
  | some tensor =>
    if tensor.dtype = "torch.int64" ∧ tensor.shape = [tensor.values.length] then do
      let selected ← decodeChannels channels tensor.values
      if selected ≠ [] ∧ selected.Nodup then some (some selected) else none
    else none

def channelTensor {channels : ℕ} (selected : List (Fin channels)) : Tensor :=
  ⟨"torch.int64", [selected.length], selected.map (fun c => .integer c.val)⟩

def selectionFields {channels : ℕ} : PersonalLDA.Selection channels → NamedState
  | none => []
  | some selected => [("waveform_channels", channelTensor selected)]

/-- Source dictionaries have unique keys. Extras may contain arbitrary tensors,
with the four LDA names and waveform_channels reserved for this producer. -/
def FreshExtras (extras : NamedState) : Prop :=
  UniqueNames extras ∧
    (∀ field : PersonalLDA.Field, lookupTensor field.key extras = none) ∧
    lookupTensor "waveform_channels" extras = none

theorem decode_encoded_channel {channels : ℕ} (c : Fin channels) :
    decodeChannel channels (.integer c.val) = some c := by
  have hn : (0 : ℤ) ≤ c.val := by positivity
  have hc : (c.val : ℤ) < channels := by exact_mod_cast c.isLt
  simp [decodeChannel, hn, hc]

theorem decode_encoded_channels {channels : ℕ} (selected : List (Fin channels)) :
    decodeChannels channels (selected.map (fun c => .integer c.val)) = some selected := by
  induction selected with
  | nil => rfl
  | cons c tail ih => simp [decodeChannels, decode_encoded_channel, ih]

theorem canonical_selection_parses {d channels : ℕ} (state : PersonalLDA.State d)
    (shape : List ℕ) (selection : PersonalLDA.Selection channels) (extras : NamedState)
    (hselection : PersonalLDA.ValidSelection selection) (hfresh : FreshExtras extras) :
    readSelection channels (canonicalNamed state shape (selectionFields selection ++ extras)) =
      some selection := by
  cases selection with
  | none =>
    simp [readSelection, canonicalNamed, selectionFields, lookupTensor, hfresh.2.2]
  | some selected =>
    simp [PersonalLDA.ValidSelection] at hselection
    simp [readSelection, canonicalNamed, selectionFields, lookupTensor, channelTensor,
      decode_encoded_channels, hselection.1, hselection.2]

def ldaMetadata (extras : Fields) : Fields :=
  .cons "model_kind" (.text Kind.targetLDA.text) extras

def packetInput {d channels : ℕ} (state : PersonalLDA.State d) (shape : List ℕ)
    (selection : PersonalLDA.Selection channels) (extras : NamedState) (metadataExtras : Fields) :
    ConstructorInput :=
  ⟨.object (ldaMetadata metadataExtras), canonicalNamed state shape (selectionFields selection ++ extras)⟩

def constructorLDA (d : ℕ) (input : ConstructorInput) : Check (PersonalLDA.State d) := do
  ensure (inputKind input = some .targetLDA) "target_only_shrinkage_lda kind"
  match PersonalLDA.loadState (namedToRecord d input.state) with
  | some state => pure state
  | none => .error (.sourceReject "typed LDA tensor shapes or scale")

def restoredEvaluator (mh : PersonalState.MetadataHash) (sh : StateHash) (d : ℕ)
    (raw : Value) (features : PersonalLDA.Vector d) : Check ℝ := do
  let input ← loadInput mh sh raw
  let state ← constructorLDA d input
  pure (PersonalLDA.evaluate state features)

theorem packet_kind_is_lda {d channels : ℕ} (state : PersonalLDA.State d) (shape : List ℕ)
    (selection : PersonalLDA.Selection channels) (extras : NamedState) (metadataExtras : Fields) :
    inputKind (packetInput state shape selection extras metadataExtras) = some .targetLDA := by
  simp [packetInput, inputKind, ldaMetadata, Fields.get, kind_text_roundtrip]

theorem packet_constructor_loads {d channels : ℕ} (state : PersonalLDA.State d) (shape : List ℕ)
    (selection : PersonalLDA.Selection channels) (extras : NamedState) (metadataExtras : Fields)
    (hshape : shape.foldr (· * ·) 1 = 1) (hscale : PersonalLDA.ValidState state) :
    constructorLDA d (packetInput state shape selection extras metadataExtras) = .ok state := by
  unfold constructorLDA
  rw [packet_kind_is_lda]
  simp [ensure, packetInput,
    canonical_named_constructor state shape (selectionFields selection ++ extras) hshape hscale]

theorem sealed_packet_restores_evaluator {d channels : ℕ} (mh : PersonalState.MetadataHash) (sh : StateHash)
    (state : PersonalLDA.State d) (shape : List ℕ) (selection : PersonalLDA.Selection channels)
    (extras : NamedState) (metadataExtras : Fields)
    (hshape : shape.foldr (· * ·) 1 = 1) (hscale : PersonalLDA.ValidState state)
    (features : PersonalLDA.Vector d) :
    restoredEvaluator mh sh d (saveView mh sh (packetInput state shape selection extras metadataExtras)) features =
      .ok (PersonalLDA.evaluate state features) := by
  have hc : constructorLDA d
      ⟨.object (ldaMetadata metadataExtras), canonicalNamed state shape (selectionFields selection ++ extras)⟩ =
      .ok state := packet_constructor_loads state shape selection extras metadataExtras hshape hscale
  simp [restoredEvaluator, packetInput, current_source_copy_restore, hc]

def liftPrediction {α : Type*} : Except Unit α → Check α
  | .ok value => .ok value
  | .error _ => .error (.sourceReject "selected channel missing")

def restoredWaveform {epochs channels times bins : ℕ} (mh : PersonalState.MetadataHash) (sh : StateHash)
    (traceMap : PersonalLDA.TraceMap times bins) (raw : Value) (policy : PersonalLDA.MissingPolicy)
    (base : Fin epochs → Bool) (observed : Fin epochs → Fin channels → Bool)
    (epochsX : Fin epochs → Fin channels → Fin times → ℝ) :
    Check (Fin epochs → Option ℝ) := do
  let input ← loadInput mh sh raw
  let selection ← match readSelection channels input.state with
    | some selection => .ok selection
    | none => .error (.sourceReject "waveform channel selection")
  let state ← constructorLDA ((PersonalLDA.selectedChannels selection).length * bins) input
  liftPrediction (PersonalLDA.predictWaveform traceMap selection state policy base observed epochsX)

theorem sealed_packet_restores_waveform {epochs channels times bins : ℕ}
    (mh : PersonalState.MetadataHash) (sh : StateHash) (traceMap : PersonalLDA.TraceMap times bins)
    (selection : PersonalLDA.Selection channels)
    (state : PersonalLDA.State ((PersonalLDA.selectedChannels selection).length * bins))
    (shape : List ℕ) (extras : NamedState) (metadataExtras : Fields)
    (hshape : shape.foldr (· * ·) 1 = 1) (hscale : PersonalLDA.ValidState state)
    (hselection : PersonalLDA.ValidSelection selection) (hfresh : FreshExtras extras)
    (policy : PersonalLDA.MissingPolicy) (base : Fin epochs → Bool)
    (observed : Fin epochs → Fin channels → Bool)
    (epochsX : Fin epochs → Fin channels → Fin times → ℝ) :
    restoredWaveform mh sh traceMap
      (saveView mh sh (packetInput state shape selection extras metadataExtras)) policy base observed epochsX =
      liftPrediction (PersonalLDA.predictWaveform traceMap selection state policy base observed epochsX) := by
  have hc : constructorLDA ((PersonalLDA.selectedChannels selection).length * bins)
      ⟨.object (ldaMetadata metadataExtras), canonicalNamed state shape (selectionFields selection ++ extras)⟩ =
      .ok state := packet_constructor_loads state shape selection extras metadataExtras hshape hscale
  simp [restoredWaveform, packetInput, current_source_copy_restore,
    canonical_selection_parses state shape selection extras hselection hfresh, hc]

def witnessExtra : NamedState :=
  [("unused_extension", ⟨"torch.float32", [1], [.number 9]⟩)]

theorem nonempty_packet_restores_legal_score (mh : PersonalState.MetadataHash) (sh : StateHash) :
    restoredEvaluator mh sh 2
      (saveView mh sh (packetInput PersonalLDA.legalState [1, 1]
        (none : PersonalLDA.Selection 2) witnessExtra .nil)) PersonalLDA.legalFeatures = .ok 45 := by
  simpa only [PersonalLDA.legal_saved_score] using sealed_packet_restores_evaluator mh sh
    PersonalLDA.legalState [1, 1] (none : PersonalLDA.Selection 2) witnessExtra .nil
    (by decide) PersonalLDA.legal_state_valid PersonalLDA.legalFeatures

end

end N2P3.PersonalRestore
