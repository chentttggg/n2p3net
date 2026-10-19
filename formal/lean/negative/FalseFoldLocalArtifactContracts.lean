import N2P3.FoldLocalArtifactContracts

namespace N2P3.FalseFoldLocalArtifactContracts
open N2P3.ArtifactFoldSupport N2P3.ArtifactCVSelection N2P3.ArtifactFoldLedger
noncomputable section

theorem false_rowwise_splitting_can_separate_equal_groups_in_the_group_assignment_model :
    innerValidation ({0,1} : Finset ℕ) (fun _ => (0 : ℕ)) id 0={0} := by
  norm_num [innerValidation,Finset.ext_iff]

theorem false_changing_inner_validation_target_cannot_change_its_candidate_error :
    cvMean [some ((0-2 : ℝ)^2)]=cvMean [some ((0-0 : ℝ)^2)] := by
  norm_num [cvMean,accumulate]

-- Candidate templates0,1 have losses0,1 for target0 and4,1 for target2.
theorem false_changing_the_fixed_validation_target_preserves_the_actual_winner :
    sourceChoice {0,1} (fun i => [some (if i=0 then 0 else 1)]) 0 (by norm_num) 1=
      sourceChoice {0,1} (fun i => [some (if i=0 then 4 else 1)]) 0 (by norm_num) 1 := by
  norm_num [sourceChoice,selectOrFallback,supportedCandidates,candidateScore,accumulate,
    selectSupported,ties,bestScore,Finset.filter_insert,Finset.filter_singleton]

theorem false_an_invalid_fold_still_increases_the_evidence_count :
    (accumulate [some (2 : ℝ),none]).2=2 := by
  norm_num [accumulate]

theorem false_an_invalid_fold_is_an_observed_zero_that_halves_the_mean :
    cvMean [some (2 : ℝ),none]=some 1 := by
  norm_num [cvMean,accumulate]

theorem false_kappa_pools_channels_instead_of_weighting_eligible_folds_equally :
    cvMean [channelMean [0],channelMean [3,3,3]]=some (9/4) := by
  norm_num [cvMean,accumulate,channelMean]

theorem false_tolerance_ties_choose_the_earliest_supported_index :
    selectSupported {0,1} (fun _ => 0) 0 (by simp) (by norm_num)=0 := by
  norm_num [selectSupported,ties,bestScore,Finset.filter_insert,Finset.filter_singleton]

theorem false_no_CV_fallback_is_positive_fold_evidence :
    0 < (accumulate ([] : List (Option ℝ))).2 := by
  norm_num [accumulate]

theorem false_local_masking_may_enable_a_missing_channel :
    effective false false=true := by
  norm_num [effective]

theorem false_deferred_zero_fill_is_equivalent_without_applying_the_mask :
    project false 7=7 := by
  norm_num [project]

theorem false_the_default_heldout_ledger_silently_applies_training_drop_flags :
    ∃ ledger, heldoutPhase .failClosed ({0,1} : Finset ℕ) (fun _ => false)=.ok ledger ∧
      ledger.scoringRows={0} := by
  simp [heldoutPhase,excludedTest,Finset.ext_iff]

theorem false_explicit_retained_count_can_replace_the_requested_denominator :
    (effectiveTest ({0,1} : Finset ℕ) (fun row => decide (row=0))).card=({0,1} : Finset ℕ).card := by
  norm_num [effectiveTest,Finset.filter_insert,Finset.filter_singleton]

theorem false_an_all_masked_heldout_row_allows_default_success :
    ∃ ledger, heldoutPhase .failClosed ({0} : Finset ℕ) (fun _ => true)=.ok ledger := by
  simp [heldoutPhase,excludedTest]

theorem false_edits_of_training_rows_are_covered_by_outer_heldout_isolation :
    trainingView [0] {0} (editRows {0} (fun (_ : ℕ) value => value+1) (fun _ => (0 : ℝ)))=
      trainingView [0] {0} (fun _ => (0 : ℝ)) := by
  norm_num [trainingView,editRows]

end
end N2P3.FalseFoldLocalArtifactContracts
