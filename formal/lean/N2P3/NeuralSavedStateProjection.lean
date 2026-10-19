import N2P3.PersonalState
import Batteries.Data.String.Lemmas

/-! NN02 of the source-first neural_saved_state requirements. The Python
literal prefix operations use their exact code-point string interface here.
This projects the actual strict-load table and separately reads saved input
statistics/configuration. It is a preconstructor argument result, not whole
neural construction, native load_state_dict or frozen-forward equivalence. -/
namespace N2P3.NeuralSavedStateProjection
open N2P3.PersonalState N2P3.FrozenState
noncomputable section

def stripPrefix (tag key : String) : Option String :=
  if key.data.take tag.data.length = tag.data then
    some ⟨key.data.drop tag.data.length⟩ else none

theorem exact_added_prefix_is_removed_once (tag suffix : String) :
    stripPrefix tag (tag ++ suffix) = some suffix := by
  cases tag with
  | mk p => cases suffix with
    | mk s => simp [stripPrefix]

theorem successful_strip_is_exact_prefix_decomposition (tag key suffix : String) :
    stripPrefix tag key = some suffix ↔ key = tag ++ suffix := by
  constructor
  · intro reply
    unfold stripPrefix at reply
    split_ifs at reply with first
    · have tail := Option.some.inj reply
      apply String.ext
      change key.data = tag.data ++ suffix.data
      rw [← tail]
      have partition := List.take_append_drop tag.data.length key.data
      rw [first] at partition
      exact partition.symm
  · rintro rfl
    exact exact_added_prefix_is_removed_once tag suffix

theorem prefix_addition_is_injective (tag : String) : Function.Injective (fun key => tag ++ key) := by
  intro first second same
  have removed := congrArg (stripPrefix tag) same
  simpa only [exact_added_prefix_is_removed_once, Option.some.injEq] using removed

def project (tag : String) : NamedState → NamedState
  | [] => []
  | (key, value) :: tail => match stripPrefix tag key with
    | none => project tag tail
    | some suffix => (suffix, value) :: project tag tail

theorem actual_projection_preserves_order_across_appended_state (tag : String) (first second : NamedState) :
    project tag (first ++ second) = project tag first ++ project tag second := by
  induction first with
  | nil => rfl
  | cons head tail ih =>
    rcases head with ⟨key, value⟩
    cases reply : stripPrefix tag key <;> simp [project, reply, ih]

theorem actual_projection_of_prefixed_state_recovers_every_tensor (tag : String) (state : NamedState) :
    project tag (prefixState tag state) = state := by
  induction state with
  | nil => rfl
  | cons head tail ih =>
    rcases head with ⟨key, value⟩
    change project tag ((tag ++ key, value) :: prefixState tag tail) = (key, value) :: tail
    rw [project, exact_added_prefix_is_removed_once, ih]

theorem actual_projected_lookup_is_the_original_prefixed_lookup
    (tag key : String) (state : NamedState) :
    lookupTensor key (project tag state) = lookupTensor (tag ++ key) state := by
  induction state with
  | nil => rfl
  | cons head tail ih =>
    rcases head with ⟨original, value⟩
    cases reply : stripPrefix tag original with
    | none =>
      have different : tag ++ key ≠ original := by
        intro equal
        subst original
        rw [exact_added_prefix_is_removed_once] at reply
        cases reply
      simp [project, reply, lookupTensor, different, ih]
    | some suffix =>
      have originalEq := (successful_strip_is_exact_prefix_decomposition tag original suffix).mp reply
      subst original
      by_cases equal : key = suffix
      · subst suffix
        simp [project, exact_added_prefix_is_removed_once, lookupTensor]
      · have different : tag ++ key ≠ tag ++ suffix := fun same =>
          equal (prefix_addition_is_injective tag same)
        simp [project, exact_added_prefix_is_removed_once, lookupTensor, equal, different, ih]

theorem actual_selected_entry_has_its_original_prefixed_key
    (tag key : String) (tensor : Tensor) (state : NamedState) :
    (key, tensor) ∈ project tag state ↔ (tag ++ key, tensor) ∈ state := by
  induction state with
  | nil => simp [project]
  | cons head tail ih =>
    rcases head with ⟨original, value⟩
    cases reply : stripPrefix tag original with
    | none =>
      have different : tag ++ key ≠ original := by
        intro equal
        subst original
        rw [exact_added_prefix_is_removed_once] at reply
        cases reply
      simp [project, reply, different, ih]
    | some suffix =>
      have originalEq := (successful_strip_is_exact_prefix_decomposition tag original suffix).mp reply
      subst original
      have eqKeys : tag ++ key = tag ++ suffix ↔ key = suffix :=
        ⟨fun same => prefix_addition_is_injective tag same, congrArg (fun k => tag ++ k)⟩
      simp [project, exact_added_prefix_is_removed_once, eqKeys, ih]

theorem actual_projection_preserves_unique_names (tag : String) (state : NamedState)
    (unique : UniqueNames state) : UniqueNames (project tag state) := by
  induction state with
  | nil => simp [UniqueNames, project]
  | cons head tail ih =>
    rcases head with ⟨original, value⟩
    have conditions := List.nodup_cons.mp unique
    cases reply : stripPrefix tag original with
    | none => simpa [project, reply] using ih conditions.2
    | some suffix =>
      have originalEq := (successful_strip_is_exact_prefix_decomposition tag original suffix).mp reply
      change ((project tag ((original, value) :: tail)).map Prod.fst).Nodup
      rw [project, reply]
      apply List.nodup_cons.mpr
      constructor
      · intro member
        obtain ⟨entry, present, keyEq⟩ := List.mem_map.mp member
        rcases entry with ⟨key, tensor⟩
        change key = suffix at keyEq
        subst key
        have rawEntry := (actual_selected_entry_has_its_original_prefixed_key tag suffix tensor tail).mp present
        apply conditions.1
        rw [originalEq]
        exact List.mem_map.mpr ⟨(tag ++ suffix, tensor), rawEntry, rfl⟩
      · exact ih conditions.2

def objectField (key : String) : Value → Option Value
  | .object fields => fields.get key
  | _ => none

structure ConstructorArguments where
  architectureConfiguration : Option Value
  physicalInputContract : Option Value
  poolingMode : Option Value
  recipe : Option Value
  expectedArchitecture : Option Value
  trunkState : NamedState
  inputMean : Option Tensor
  inputStd : Option Tensor

def arguments (input : ConstructorInput) : ConstructorArguments :=
  ⟨objectField "architecture_configuration" input.metadata,
   objectField "input_contract" input.metadata,
   objectField "pooling_mode" input.metadata,
   objectField "recipe" input.metadata,
   objectField "architecture" input.metadata,
   project "trunk." input.state,
   lookupTensor "input_mean" input.state,
   lookupTensor "input_std" input.state⟩

theorem copied_packet_preserves_actual_constructor_arguments (input : ConstructorInput) :
    arguments (copyInput input) = arguments input := by rw [copyInput_preserves]

theorem saved_loaded_packet_preserves_actual_constructor_arguments
    (mh : PersonalState.MetadataHash) (sh : PersonalState.StateHash) (metadata : Fields) (state : NamedState) :
    (loadInput mh sh (saveView mh sh ⟨.object metadata, state⟩)).map arguments =
      .ok (arguments ⟨.object metadata, state⟩) := by
  rw [current_source_copy_restore]
  rfl

theorem selected_trunk_tensor_matches_the_original_saved_tensor
    (input : ConstructorInput) (name : String) :
    lookupTensor name (arguments input).trunkState = lookupTensor ("trunk." ++ name) input.state :=
  actual_projected_lookup_is_the_original_prefixed_lookup "trunk." name input.state

theorem fixed_nonprefix_is_not_a_selected_model_tensor : stripPrefix "trunk." "xtrunk.weight" = none := by decide

theorem fixed_repeated_prefix_is_removed_only_once :
    stripPrefix "trunk." "trunk.trunk.weight" = some "trunk.weight" := by decide

def witnessTensor : Tensor := ⟨"torch.float32", [1], [.number 2]⟩
def witnessState : NamedState := [("input_mean", witnessTensor), ("input_std", witnessTensor), ("trunk.weight", witnessTensor)]

theorem fixed_saved_statistics_are_separate_from_strict_model_state :
    project "trunk." witnessState = [("weight", witnessTensor)] ∧
      lookupTensor "input_mean" witnessState = some witnessTensor ∧
      lookupTensor "input_mean" (project "trunk." witnessState) = none ∧
      lookupTensor "input_std" (project "trunk." witnessState) = none := by
  simp [project, stripPrefix, witnessState, lookupTensor]

end
end N2P3.NeuralSavedStateProjection
