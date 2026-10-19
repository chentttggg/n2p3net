import N2P3.RequestedBudgetPolicyWitnesses

namespace N2P3.NegativeRequestedBudgetPolicy

open N2P3.RequestedBudgetPolicyWeights N2P3.RequestedBudgetPolicyWitnesses

-- K01: omitting actual-K normalization incorrectly gives B0 weight1/8.
example : policyCoefficient program 1 0 200 (0,0)=1/8 := by
  rw [actual_variable_K_counts_derive_different_primary_row_coefficients.2.1]
  norm_num

-- K02: surviving K200 cannot rescue the person whose K800 grid failed.
example : policyCoefficient program 0 1 200 (0,0)=1/8 := by
  rw [actual_variable_K_counts_derive_different_primary_row_coefficients.2.2.2]
  norm_num

-- K03: a common K-count2 is incorrect for the complete B1 person with3 K.
example : policyCoefficient program 1 1 200 (0,0)=1/16 := by
  rw [actual_variable_K_counts_derive_different_primary_row_coefficients.2.2.1]
  norm_num

-- K04: the outer person with unknown K is not a third policy request.
example : (policyDomain program 0).card=3 := by
  rw [actual_policy_domains_exclude_unknown_K_but_keep_known_failed_people.2.2.2]
  norm_num

-- K05: missing an expected study does not become a full result0.
example : fullPolicyEstimate missingStudyProgram=some 0 := by
  rw [one_missing_study_closes_the_full_policy_point]
  simp

end N2P3.NegativeRequestedBudgetPolicy
