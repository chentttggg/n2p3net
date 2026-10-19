import N2P3.PersonalERPIdentity

/-! Source-fixed alternatives: authority-only/origin-only checks, category
tagging, study/display relabeling, resigned study metadata and T-only contracts.
These statements were chosen from the actual source guards before proof checks.
-/

namespace N2P3.NegativePersonalERPIdentity

open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalERPIdentity

example : validateTarget exampleTargetPrimitives exampleSource
    (targetOfRows examplePreprocessing "renamed" (exampleRows exampleSharedOriginIdentity)) = .ok () := by
  rw [changed_authority_shared_origin_is_refused]
  have hn : (Except.error (.sourceReject "temporal target participant") : Check Unit) ≠ .ok () := by
    intro h; cases h
  simp only [hn]

example : validateTarget exampleTargetPrimitives exampleSource
    (targetOfRows examplePreprocessing "different-origin" (exampleRows exampleSharedGlobalIdentity)) = .ok () := by
  rw [distinct_origins_shared_global_is_refused]
  have hn : (Except.error (.sourceReject "temporal target participant") : Check Unit) ≠ .ok () := by
    intro h; cases h
  simp only [hn]

example : validateTarget exampleTargetPrimitives exampleSource
    (targetOfRows examplePreprocessing "independent-display" (exampleRows exampleCrossCategoryIdentity)) = .ok () := by
  rw [origin_global_cross_category_is_refused]
  have hn : (Except.error (.sourceReject "temporal target participant") : Check Unit) ≠ .ok () := by
    intro h; cases h
  simp only [hn]

example : validateTarget exampleTargetPrimitives exampleSource
    (targetOfRows examplePreprocessing "renamed-safe-display" (exampleRows exampleSameStudyIdentity)) = .ok () := by
  rw [different_people_same_study_are_refused]
  have hn : (Except.error (.sourceReject "complete target study") : Check Unit) ≠ .ok () := by
    intro h; cases h
  simp only [hn]

example : validateTarget exampleTargetPrimitives exampleSource renamedSameStudyTarget = .ok () := by
  rw [renamed_same_study_is_still_refused]
  have hn : (Except.error (.sourceReject "complete target study") : Check Unit) ≠ .ok () := by
    intro h; cases h
  simp only [hn]

example (hash : Fields → String) : ERPState.readTemporal (exampleTemporalPrimitives hash)
    (.object (sourceRecord (exampleTemporalPrimitives hash) exampleBasis [exampleReceipt] ["safe-fiction"])) =
      .ok (sourceModel (exampleTemporalPrimitives hash) exampleBasis [exampleReceipt] ["safe-fiction"]) := by
  rw [resigning_fake_study_metadata_still_refuses]
  have hn : (Except.error (.sourceReject "temporal source studies") : Check ERPState.TemporalModel) ≠
      .ok (sourceModel (exampleTemporalPrimitives hash) exampleBasis [exampleReceipt] ["safe-fiction"]) := by
    intro h; cases h
  simp only [hn]

example : validateTarget exampleTargetPrimitives exampleSource
    (targetOfRows differentFrequencyPreprocessing "same-T" (exampleRows exampleTargetIdentity)) = .ok () := by
  rw [same_T_different_full_preprocessing_is_refused]
  have hn : (Except.error (.sourceReject "temporal target preprocessing") : Check Unit) ≠ .ok () := by
    intro h; cases h
  simp only [hn]

end N2P3.NegativePersonalERPIdentity
