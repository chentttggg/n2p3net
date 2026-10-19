import N2P3.PersonalERPRecipe

/-! Fourteen source-fixed alternatives N01-N12 and current-source NR01-NR02.
These are semantic controls;
source-role controls exercise the general decoded role gate, not physical fit
authentication. Every example must fail solely with an unsolved False goal. -/
namespace N2P3.NegativePersonalERPRecipe
open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalState N2P3.ERPState
open N2P3.PersonalERPRecipe

example : defaultWidth = .ok (.integer 2) := by
  rw [missing_width_reconstructs_one]
  have hn : (Except.ok (.integer 1) : Check Value) ≠ .ok (.integer 2) := by intro h; cases h
  simp only [hn]

example : validateERPRecipe { (pooledFields 2).raw with erpBinWidth := .boolean true } = .ok fixtureRecipe := by
  rw [boolean_width_rejected]
  have hn : (Except.error (.sourceReject "erp_bin_width") : Check Recipe) ≠ .ok fixtureRecipe := by intro h; cases h
  simp only [hn]

example : validateERPRecipe { (pooledFields 2).raw with dropout := .text "checkpoint" } = .ok fixtureRecipe := by
  rw [enabled_dropout_rejected]
  have hn : (Except.error (.sourceReject "ERP frozen treatment") : Check Recipe) ≠ .ok fixtureRecipe := by intro h; cases h
  simp only [hn]

example : readSelected 3 (selectedTensor [0, 0]) = .ok [0, 0] := by
  rw [duplicate_selection_rejected]
  have hn : (Except.error (.sourceReject "ERP selected unique/range") : Check (List ℕ)) ≠ .ok [0, 0] := by intro h; cases h
  simp only [hn]

example : readSelected 3 (selectedTensor [2, 0]) = .ok [0, 2] := by
  rw [selection_permutation_is_preserved]
  have hn : (Except.ok [2, 0] : Check (List ℕ)) ≠ .ok [0, 2] := by intro h; cases h
  simp only [hn]

example : fixtureBranch ⟨3, 6, .text "none"⟩ = .ok ⟨fixtureRecipe, [0, 2], fixtureModel⟩ := by
  rw [physical_time_mismatch_rejected]
  have hn : (Except.error (.sourceReject "ERP state recipe physical contract") : Check Bound) ≠
      .ok ⟨fixtureRecipe, [0, 2], fixtureModel⟩ := by intro h; cases h
  simp only [hn]

example : fixtureBranch fixturePhysical (recordFields (pooledFields 1).raw) = .ok ⟨fixtureRecipe, [0, 2], fixtureModel⟩ := by
  rw [recipe_width_mismatch_rejected]
  have hn : (Except.error (.sourceReject "ERP state recipe physical contract") : Check Bound) ≠
      .ok ⟨fixtureRecipe, [0, 2], fixtureModel⟩ := by intro h; cases h
  simp only [hn]

example : fromState fixturePrimitives fixtureReference (plainRecord .pooled 1 1 1 1 (.text "none"))
    (erpFields ((writePersonalERP (fixturePlain .pooled) (selectedTensor [0]) []) ++
      [("erp.unrecognized", realTensor [1] [9])])) = .ok (fixturePlain .pooled) := by
  rw [prefixed_unknown_key_reaches_inner_rejection]
  have hn : (Except.error (.sourceReject "ERP state keys") : Check Model) ≠ .ok (fixturePlain .pooled) := by intro h; cases h
  simp only [hn]

example : bindERP (temporalRecipe "source_supervised" 1) "source_temporal_target_erp" (.integer 1)
    ⟨1, 1, .text "none"⟩ [0] (roleModel .source) =
      .ok ⟨temporalRecipe "source_supervised" 1, [0], roleModel .source⟩ := by
  rw [source_one_is_not_true]
  have hn : (Except.error (.sourceReject "ERP strict source role") : Check Bound) ≠
      .ok ⟨temporalRecipe "source_supervised" 1, [0], roleModel .source⟩ := by intro h; cases h
  simp only [hn]

example : bindERP fixtureRecipe "target_only_structured_erp" (.integer 0) fixturePhysical [0, 2] fixtureModel =
    .ok ⟨fixtureRecipe, [0, 2], fixtureModel⟩ := by
  rw [source_free_zero_is_not_false]
  have hn : (Except.error (.sourceReject "ERP strict source role") : Check Bound) ≠
      .ok ⟨fixtureRecipe, [0, 2], fixtureModel⟩ := by intro h; cases h
  simp only [hn]

example : bindERP (temporalRecipe "source_supervised" 1) "target_only_structured_erp" (.boolean true)
    ⟨1, 1, .text "none"⟩ [0] (roleModel .source) =
      .ok ⟨temporalRecipe "source_supervised" 1, [0], roleModel .source⟩ := by
  rw [wrong_model_kind_rejected]
  have hn : (Except.error (.sourceReject "ERP strict source role") : Check Bound) ≠
      .ok ⟨temporalRecipe "source_supervised" 1, [0], roleModel .source⟩ := by intro h; cases h
  simp only [hn]

example : bindERP (temporalRecipe "fixed_cosine" 1) "source_temporal_target_erp" (.boolean true)
    ⟨1, 1, .text "none"⟩ [0] (roleModel .source) =
      .ok ⟨temporalRecipe "fixed_cosine" 1, [0], roleModel .source⟩ := by
  rw [wrong_temporal_recipe_kind_rejected]
  have hn : (Except.error (.sourceReject "ERP temporal kind/rank") : Check Bound) ≠
      .ok ⟨temporalRecipe "fixed_cosine" 1, [0], roleModel .source⟩ := by intro h; cases h
  simp only [hn]

example : CurrentERPState.fromState fixturePrimitives fixtureReference
    (xdawnRecord 1 1 1 1 1 1 (.text "none")) fixtureXDawnState =
      .ok ⟨⟨.xdawn, 1, 1, .text "none", none⟩, 1, 1, fixtureXDawnState, some 1, some 1⟩ := by
  rw [CurrentERPState.previous_one_time_xdawn_fixture_now_rejects]
  have hn : (Except.error (.sourceReject "current xDAWN universal zero energy") : Check Model) ≠
      .ok ⟨⟨.xdawn, 1, 1, .text "none", none⟩, 1, 1, fixtureXDawnState, some 1, some 1⟩ := by intro h; cases h
  simp only [hn]

example : (CurrentERPState.predictAfterAugmentation CurrentERPState.finiteRowControl [true, false]
    (realTensor [1] [1]) (realTensor [1] [1]) (realTensor [] [0])).result = .ok [7] := by
  rw [CurrentERPState.finite_row_countermodel_rejects_whole_batch]
  have hn : (Except.error CurrentERPState.PredictionFault.covarianceNonfinite :
      Except CurrentERPState.PredictionFault (List ℝ)) ≠ .ok [7] := by intro h; cases h
  simp only [hn]

end N2P3.NegativePersonalERPRecipe
