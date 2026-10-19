import N2P3.OperatorState

/-! Six source-derived deliberate failures for the primitive reconstruction path. -/

namespace N2P3.NegativeOperatorState

open N2P3.FrozenState N2P3.StateLedger N2P3.OperatorState

-- Schema is checked before the missing/mismatching seal fields.
example (ah : ArrayHash) (mh : N2P3.FrozenState.MetadataHash) :
    loadEnvelope ah mh
      ⟨encodePrimitives fixtureArrays, Fields.ofList [("schema", .text "wrong-schema")]⟩ =
      .ok ⟨fixtureArrays, .nil⟩ := by
  have he := schema_rejected_before_seals ah mh
    ⟨encodePrimitives fixtureArrays, Fields.ofList [("schema", .text "wrong-schema")]⟩
    fixtureArrays (five_arrays_decode_independently fixtureArrays .nil)
    (by simp [fieldDefault, Fields.ofList, Fields.get])
  rw [he]
  have hn : (Except.error (.sourceReject "operator schema") : Check Envelope) ≠ .ok ⟨fixtureArrays, .nil⟩ := by
    intro h
    cases h
  simp only [hn]

-- Bool is excluded from the two exact integer class counts.
example : checkClassCounts (.sequence (.cons (.boolean true) (.cons (.integer 1) .nil))) 2 = .ok () := by
  rw [boolean_class_count_rejected]
  have hn : (Except.error (.sourceReject "operator class counts") : Check Unit) ≠ .ok () := by
    intro h
    cases h
  simp only [hn]

-- Source noise can have a distinct scope; operator scope_id only needs truthiness.
example : checkOperatorLedger (fixtureNoiseLedger .sourceTraining)
    (.object (fixtureOperatorFields (fixtureNoiseLedger .sourceTraining) (.boolean true))) ≠ .ok () := by
  simp only [source_noise_allows_distinct_truthy_scope, ne_eq, not_true_eq_false]

-- Calibration-fitted noise must instead share the classifier scope.
example : checkOperatorLedger (fixtureNoiseLedger .targetCalibration)
    (.object (fixtureOperatorFields (fixtureNoiseLedger .targetCalibration) (.boolean true))) = .ok () := by
  rw [target_noise_rejects_distinct_truthy_scope]
  have hn : (Except.error (.sourceReject "operator target noise scope") : Check Unit) ≠ .ok () := by
    intro h
    cases h
  simp only [hn]

def forgedDerived : Fields := Fields.ofList
  [("weights", encodeArray ⟨[1], [999]⟩), ("covariance", encodeArray ⟨[1, 1], [999]⟩),
   ("identifiable_projector", encodeArray ⟨[1, 1], [999]⟩), ("intercept", encodeArray ⟨[], [999]⟩)]

-- NPZ derived fields are ignored; both calls reconstruct the same legal fixture.
example (ah : ArrayHash) (mh : N2P3.FrozenState.MetadataHash) :
    load fixtureNumerics ah mh (attachArtifactSeal mh (encodeInput ah fixtureInput forgedDerived)) ≠
      load fixtureNumerics ah mh (attachArtifactSeal mh (encodeInput ah fixtureInput)) := by
  rw [extra_npz_arrays_are_ignored fixtureNumerics ah mh fixtureInput forgedDerived .nil]
  simp only [ne_eq, not_true_eq_false]

-- Sealed raw covariance and the constructor's normalized image differ here.
example : matrixOfArray 2 2 (normalizedJointCovariance 2 slightlyAsymmetricCovariance) 0 1 =
    covarianceEntry slightlyAsymmetricCovariance 2 0 1 := by
  simp only [raw_covariance_entry_changes_under_normalization]

end N2P3.NegativeOperatorState
