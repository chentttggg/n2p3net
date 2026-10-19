import N2P3.FrozenState

/-! Five deliberate failures at the actual semantic-codec/guard boundaries. -/

namespace N2P3.NegativeFrozenState

open N2P3.FrozenState

-- Arrays alone do not bind phi/operator metadata consumed by prediction.
example (ah : ArrayHash) :
    ah (continuousArrays referenceContinuous) = ah (continuousArrays changedMetadata) →
      arInnovation referenceContinuous 1 0 = arInnovation changedMetadata 1 0 := by
  intro _
  simp only [changed_metadata_changes_ar_step]

-- Hash equality is not a proof of semantic-state equality.
example (ah : ArrayHash) :
    (sealContinuous ah constantMetadataHash referenceContinuous).integrity =
      (sealContinuous ah constantMetadataHash changedMetadata).integrity →
        referenceContinuous = changedMetadata := by
  intro _
  simp only [changed_metadata_changes_state]

-- A self-consistent forged record does not authenticate an external original.
example (ah : ArrayHash) (mh : MetadataHash) :
    loadContinuous ah mh (attachArtifactSeal mh (encodeContinuous ah changedMetadata)) =
      .readyInput referenceContinuous := by
  rw [continuous_protocol_roundtrip ah mh _ changed_metadata_core]
  have hne : changedMetadata ≠ referenceContinuous := Ne.symm changed_metadata_changes_state
  simp only [LoadResult.readyInput.injEq, hne]

-- Correct metadata/state seals do not license bypassing the phi semantic guard.
example (ah : ArrayHash) (mh : MetadataHash) :
    loadContinuous ah mh (attachArtifactSeal mh (encodeContinuous ah invalidPhi)) =
      .readyInput invalidPhi := by
  rw [invalid_phi_rejected_despite_matching_seals]
  have hne : (LoadResult.rejected LoadFailure.semanticCore : LoadResult ContinuousState) ≠
      .readyInput invalidPhi := by intro h; cases h
  simp only [hne]

-- Exact constructor covariance averaging needs the stated symmetric normal form.
example : reconstructedCovarianceEntry slightlyAsymmetricCovariance 2 0 1 =
    covarianceEntry slightlyAsymmetricCovariance 2 0 1 := by
  simp only [arbitrary_covariance_normalization_can_change_entry]

end N2P3.NegativeFrozenState
