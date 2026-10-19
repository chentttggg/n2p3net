import N2P3.ParameterOwnership

/- The six alternatives were fixed from source before this implementation. -/
namespace N2P3.NegativeParameterOwnership
open N2P3.ParameterOwnership

private theorem error_ne_ok {ε α : Type} (error : ε) (value : α) :
    (Except.error error : Except ε α) ≠ .ok value := by intro equal; cases equal

private theorem ok_ne_error {ε α : Type} (value : α) (error : ε) :
    (Except.ok value : Except ε α) ≠ .error error := by intro equal; cases equal

example : parameterGroups {0, 1} overlappingGroups = .ok overlappingGroups := by
  rw [cross_group_overlap_is_refused]
  simp only [error_ne_ok]

example : parameterGroups {0} duplicateGroup = .ok duplicateGroup := by
  rw [within_group_duplicate_is_refused]
  simp only [error_ne_ok]

example : parameterGroups {0, 1} missingGroup = .ok missingGroup := by
  rw [omitted_model_parameter_is_refused]
  simp only [error_ne_ok]

example : parameterGroups {0} foreignGroup = .ok foreignGroup := by
  rw [foreign_parameter_is_refused]
  simp only [error_ne_ok]

example : parameterGroups {0} withEmptyRole = .error .coverage := by
  rw [empty_structural_role_is_allowed]
  simp only [ok_ne_error]

example : parameterGroups {0} whitespaceRole = .error .emptyName := by
  rw [whitespace_name_is_allowed]
  simp only [ok_ne_error]

end N2P3.NegativeParameterOwnership
