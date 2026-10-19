import Mathlib.Data.Finset.Card
import Mathlib.Data.List.Nodup

/-!+# Literal structural parameter ownership

Source: models/parameter_groups.py:12-34.  The contract and independent wrong
alternatives were fixed in v07_parameter_ownership_contract_01.json before
this module.  Parameter identities and ordered module enumeration are supplied
finite observations.  No ownership decision is delegated to an oracle.

Names are already strings from the declared mapping domain. Python object
identity, module iterator fidelity and dictionary uniqueness remain explicit
runtime boundaries. This does not yet cover every model's actual group list.
-/
namespace N2P3.ParameterOwnership

structure Group where
  name : String
  identities : List ℕ
  deriving DecidableEq

inductive Failure where
  | emptyName
  | overlap (name : String)
  | coverage
  deriving DecidableEq

def flattened (groups : List Group) : List ℕ :=
  groups.flatMap Group.identities

def resolve (model owned : Finset ℕ) : List Group → Except Failure (List Group)
  | [] => if owned = model then .ok [] else .error .coverage
  | group :: tail =>
    if group.name = "" then .error .emptyName
    else if ¬group.identities.Nodup ∨ ¬Disjoint owned group.identities.toFinset then
      .error (.overlap group.name)
    else (resolve model (owned ∪ group.identities.toFinset) tail).map (group :: ·)

def parameterGroups (model : Finset ℕ) (groups : List Group) : Except Failure (List Group) :=
  resolve model ∅ groups

theorem flattened_cons (group : Group) (tail : List Group) :
    flattened (group :: tail) = group.identities ++ flattened tail := rfl

theorem empty_name_refused_before_identity_or_coverage (model owned : Finset ℕ)
    (group : Group) (tail : List Group) (name : group.name = "") :
    resolve model owned (group :: tail) = .error .emptyName := by
  simp [resolve, name]

theorem duplicate_identity_refused_before_coverage (model owned : Finset ℕ)
    (group : Group) (tail : List Group) (name : group.name ≠ "")
    (duplicate : ¬group.identities.Nodup) :
    resolve model owned (group :: tail) = .error (.overlap group.name) := by
  simp [resolve, name, duplicate]

theorem prior_owned_identity_refused_before_coverage (model owned : Finset ℕ)
    (group : Group) (tail : List Group) (name : group.name ≠ "")
    (identity : ℕ) (earlier : identity ∈ owned) (current : identity ∈ group.identities) :
    resolve model owned (group :: tail) = .error (.overlap group.name) := by
  have overlap : ¬Disjoint owned group.identities.toFinset := by
    intro disjoint
    exact Finset.disjoint_left.mp disjoint earlier (by simpa using current)
  simp [resolve, name, overlap]

theorem resolve_from_independent_identity_fields (model owned : Finset ℕ)
    (groups : List Group)
    (names : ∀ group ∈ groups, group.name ≠ "")
    (unique : (flattened groups).Nodup)
    (fresh : Disjoint owned (flattened groups).toFinset)
    (coverage : owned ∪ (flattened groups).toFinset = model) :
    resolve model owned groups = .ok groups := by
  induction groups generalizing owned with
  | nil =>
    simpa [resolve, flattened] using coverage
  | cons group tail ih =>
    have nd := List.nodup_append.mp (show (group.identities ++ flattened tail).Nodup by
      simpa only [flattened_cons] using unique)
    have freshGroup : Disjoint owned group.identities.toFinset := by
      apply Finset.disjoint_left.mpr
      intro identity earlier current
      exact Finset.disjoint_left.mp fresh earlier (by
        simp only [flattened_cons, List.toFinset_append, Finset.mem_union]
        exact Or.inl current)
    have freshTail : Disjoint (owned ∪ group.identities.toFinset) (flattened tail).toFinset := by
      apply Finset.disjoint_left.mpr
      intro identity earlier current
      rcases Finset.mem_union.mp earlier with earlier | earlier
      · exact Finset.disjoint_left.mp fresh earlier (by
          simp only [flattened_cons, List.toFinset_append, Finset.mem_union]
          exact Or.inr current)
      · exact nd.2.2 identity (by simpa using earlier) identity (by simpa using current) rfl
    have final : (owned ∪ group.identities.toFinset) ∪ (flattened tail).toFinset = model := by
      simpa [flattened_cons, List.toFinset_append, Finset.union_assoc] using coverage
    have rest := ih (owned ∪ group.identities.toFinset)
      (fun next member => names next (by simp [member])) nd.2.1 freshTail final
    simp [resolve, names group (by simp), nd.1, freshGroup, rest, Except.map]

theorem parameter_groups_preserve_order_from_independent_fields (model : Finset ℕ)
    (groups : List Group)
    (names : ∀ group ∈ groups, group.name ≠ "")
    (unique : (flattened groups).Nodup)
    (coverage : (flattened groups).toFinset = model) :
    parameterGroups model groups = .ok groups := by
  apply resolve_from_independent_identity_fields model ∅ groups names unique
  · exact Finset.disjoint_empty_left _
  · simpa using coverage

structure IdentityFields (model owned : Finset ℕ) (groups : List Group) : Prop where
  names : ∀ group ∈ groups, group.name ≠ ""
  unique : (flattened groups).Nodup
  fresh : Disjoint owned (flattened groups).toFinset
  coverage : owned ∪ (flattened groups).toFinset = model

/-- This is a successful-execution postcondition. The preceding construction
theorem separately derives success from the independent input fields. -/
theorem every_success_recovers_original_order_and_identity_partition
    (model owned : Finset ℕ) (groups result : List Group)
    (success : resolve model owned groups = .ok result) :
    result = groups ∧ IdentityFields model owned groups := by
  induction groups generalizing owned result with
  | nil =>
    by_cases final : owned = model
    · simp [resolve, final] at success
      subst result
      exact ⟨rfl, ⟨by simp, by simp [flattened], by simp [flattened], by simpa [flattened] using final⟩⟩
    · simp [resolve, final] at success
  | cons group tail ih =>
    by_cases name : group.name = ""
    · simp [resolve, name] at success
    · by_cases conflict : ¬group.identities.Nodup ∨ ¬Disjoint owned group.identities.toFinset
      · simp [resolve, name, conflict] at success
      · have nodup : group.identities.Nodup := Classical.not_not.mp (not_or.mp conflict).1
        have freshGroup : Disjoint owned group.identities.toFinset :=
          Classical.not_not.mp (not_or.mp conflict).2
        cases tailResult : resolve model (owned ∪ group.identities.toFinset) tail with
        | error fault => simp [resolve, name, conflict, tailResult, Except.map] at success
        | ok output =>
          obtain ⟨same, fields⟩ := ih (owned ∪ group.identities.toFinset) output tailResult
          subst output
          simp [resolve, name, conflict, tailResult, Except.map] at success
          subst result
          have cross : ∀ identity ∈ group.identities, ∀ other ∈ flattened tail, identity ≠ other := by
            intro identity first other later equal
            subst other
            exact Finset.disjoint_left.mp fields.fresh
              (Finset.mem_union.mpr (Or.inr (List.mem_toFinset.mpr first)))
              (List.mem_toFinset.mpr later)
          have freshTail : Disjoint owned (flattened tail).toFinset := by
            apply Finset.disjoint_left.mpr
            intro identity earlier later
            exact Finset.disjoint_left.mp fields.fresh (by simp [earlier]) later
          refine ⟨rfl, ⟨?_, ?_, ?_, ?_⟩⟩
          · intro next member
            rcases List.mem_cons.mp member with same | later
            · simpa [same] using name
            · exact fields.names next later
          · rw [flattened_cons]
            exact List.nodup_append.mpr ⟨nodup, fields.unique, cross⟩
          · simpa only [flattened_cons, List.toFinset_append, Finset.disjoint_union_right] using
              And.intro freshGroup freshTail
          · simpa only [flattened_cons, List.toFinset_append, Finset.union_assoc] using fields.coverage

theorem accepted_parameter_identity_iff_model_identity (model : Finset ℕ) (groups result : List Group)
    (success : parameterGroups model groups = .ok result) (identity : ℕ) :
    identity ∈ flattened groups ↔ identity ∈ model := by
  have fields := (every_success_recovers_original_order_and_identity_partition model ∅ groups result success).2
  have coverage : (flattened groups).toFinset = model := by simpa using fields.coverage
  rw [← coverage]
  simp

def twoGroups : List Group := [⟨"spatial", [0]⟩, ⟨"readout", [1]⟩]
def overlappingGroups : List Group := [⟨"spatial", [0]⟩, ⟨"readout", [0, 1]⟩]
def duplicateGroup : List Group := [⟨"spatial", [0, 0]⟩]
def missingGroup : List Group := [⟨"spatial", [0]⟩]
def foreignGroup : List Group := [⟨"spatial", [0, 1]⟩]
def withEmptyRole : List Group := [⟨"normalization", []⟩, ⟨"readout", [0]⟩]
def whitespaceRole : List Group := [⟨" ", [0]⟩]

theorem nonempty_partition_is_accepted_in_original_order :
    parameterGroups {0, 1} twoGroups = .ok twoGroups := by
  simp +decide [parameterGroups, resolve, twoGroups, Except.map]

theorem cross_group_overlap_is_refused :
    parameterGroups {0, 1} overlappingGroups = .error (.overlap "readout") := by
  simp +decide [parameterGroups, resolve, overlappingGroups, Except.map]

theorem within_group_duplicate_is_refused :
    parameterGroups {0} duplicateGroup = .error (.overlap "spatial") := by
  simp +decide [parameterGroups, resolve, duplicateGroup]

theorem omitted_model_parameter_is_refused :
    parameterGroups {0, 1} missingGroup = .error .coverage := by
  simp +decide [parameterGroups, resolve, missingGroup, Except.map]

theorem foreign_parameter_is_refused :
    parameterGroups {0} foreignGroup = .error .coverage := by
  simp +decide [parameterGroups, resolve, foreignGroup, Except.map]

theorem empty_structural_role_is_allowed :
    parameterGroups {0} withEmptyRole = .ok withEmptyRole := by
  simp +decide [parameterGroups, resolve, withEmptyRole, Except.map]

theorem whitespace_name_is_allowed :
    parameterGroups {0} whitespaceRole = .ok whitespaceRole := by
  simp +decide [parameterGroups, resolve, whitespaceRole, Except.map]

end N2P3.ParameterOwnership
