import N2P3.PersonalERPFit

/-! Seven source-fixed alternatives: unwanted adapter/LDA assignment, wrong
physical channel, unsliced label row, conflated outer/inner label guards,
class refusal before mean overflow, and invented post-mean finite refusal.
The fit fixture uses a declared numerical array operation; no genuine fitting
or physical source authentication is inferred from these semantic controls. -/
namespace N2P3.NegativePersonalERPFit
open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalState N2P3.PersonalERPFit

example : fitStages meanOverflowModel orderFixtureHeader orderFixtureValues threeLabels =
    .error (.sourceReject "StructuredERP at least two rows in each class") := by
  rw [declared_mean_overflow_precedes_three_row_class_refusal]
  have hn : (Except.error (.fieldFailure "numpy mean overflow") : Check NumericalPayload) ≠
      .error (.sourceReject "StructuredERP at least two rows in each class") := by intro h; cases h
  simp only [hn]

example : fitStages meanNonfiniteModel orderFixtureHeader orderFixtureValues threeLabels =
    .error (.sourceReject "StructuredERP finite input geometry") := by
  rw [declared_nonfinite_mean_preserves_three_row_class_refusal]
  have hn : (Except.error (.sourceReject "StructuredERP at least two rows in each class") : Check NumericalPayload) ≠
      .error (.sourceReject "StructuredERP finite input geometry") := by
    simp only [ne_eq, Except.error.injEq, LedgerFault.sourceReject.injEq]
    decide
  simp only [hn]

example : (buildCandidate (PersonalERPRecipe.pooledFields 1).recipe 64
    (fun _ => true) fitFixtureProduced).adapter = some [] := by
  rw [ERP_fixture_adapter_is_absent]
  have hn : (none : Option NamedState) ≠ some [] := by intro h; cases h
  simp only [hn]

example : (buildCandidate (PersonalERPRecipe.pooledFields 1).recipe 64
    (fun _ => true) fitFixtureProduced).lda = some [] := by
  rw [ERP_fixture_LDA_is_absent]
  have hn : (none : Option NamedState) ≠ some [] := by intro h; cases h
  simp only [hn]

example : fitFixtureInvocation.values firstFitRow firstFitChannel 0 = fitFixtureValues 2 0 0 := by
  rw [fit_fixture_uses_selected_channel_and_original_row.1,
    fit_fixture_uses_selected_channel_and_original_row.2]
  have hn : (21 : ℝ) ≠ 20 := by norm_num
  simp only [hn]

example : fitFixtureInvocation.labels firstFitRow = fitFixtureLabels 0 := by
  rw [fit_fixture_labels_are_sliced_with_values.1,
    fit_fixture_labels_are_sliced_with_values.2]
  rw [show ((false : Bool) = true) ↔ False by decide]

example : structuredLabelGate threeLabels = .ok () := by
  rw [three_rows_pass_candidate_but_fail_structured_fit.2]
  have hn : (Except.error (.sourceReject "StructuredERP at least two rows in each class") : Check Unit) ≠ .ok () := by
    intro h; cases h
  simp only [hn]

end N2P3.NegativePersonalERPFit
