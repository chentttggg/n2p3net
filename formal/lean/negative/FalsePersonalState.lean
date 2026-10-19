import N2P3.PersonalState

/-! Six deliberate failures for decoded personal packets and preconstructor guards.
No negative example claims completion of a numerical model constructor. -/

namespace N2P3.NegativePersonalState

open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalState

-- Unsupported schema is refused before corrupt metadata or state is inspected.
example : loadInput zeroHashMetadata zeroHashState
    (payloadView (.text "n2p3_calibrated_predictor/2") .null .null .null .null) =
      .error (.sourceReject "metadata digest") := by
  rw [schema_rejected_first _ _ _ _ _ _ _ (by simp [SupportedSchema, currentSchema])]
  have hn : LedgerFault.sourceReject "schema" ≠ .sourceReject "metadata digest" := by decide
  simp only [Except.error.injEq, hn]

-- A malformed state cannot overtake the metadata seal failure.
example : loadInput zeroHashMetadata zeroHashState
    (payloadView (.text currentSchema) (.object .nil) .null (.text "bad") .null) =
      .error (.fieldFailure "decoded tensor state") := by
  rw [metadata_seal_before_state _ _ _ _ _ _ _
    (by simp [SupportedSchema]) (by simp [zeroHashMetadata])]
  have hn : LedgerFault.sourceReject "metadata digest" ≠
      .fieldFailure "decoded tensor state" := by intro h; cases h
  simp only [Except.error.injEq, hn]

-- Valid seals do not allow a new QC method in a legacy packet.
example : loadInput zeroHashMetadata zeroHashState
    (sealedView zeroHashMetadata zeroHashState (.text "n2p3_calibrated_predictor/1") legacyNewQC) =
      .ok legacyNewQC := by
  rw [legacy_new_quality_rejected]
  have hn : (Except.error (.sourceReject "legacy epoch quality") : Check ConstructorInput) ≠
      .ok legacyNewQC := by intro h; cases h
  simp only [hn]

-- Legacy QC uses is-not-None: False is not an absent quality value.
example : checkLegacyQC (.text "n2p3_calibrated_predictor/3")
    (Fields.ofList [("epoch_quality", .boolean false)]) = .ok () := by
  rw [false_quality_is_not_missing _ (by simp [currentSchema])]
  have hn : (Except.error (.fieldFailure "epoch_quality[model]") : Check Unit) ≠ .ok () := by
    intro h
    cases h
  simp only [hn]

-- Equality of independent seals supplies no tensor-state identity theorem.
example : zeroHashState (canonicalInput .targetLDA).state = zeroHashState changedDtype.state →
    (canonicalInput .targetLDA).state = changedDtype.state := by
  intro _
  simp only [same_seals_do_not_identify_tensor_state.2]

-- Equal shapes and values do not license dropping tensor dtype from the snapshot.
example : encodeTensor (vectorTensor "torch.int64" [.integer 1]) =
    encodeTensor (vectorTensor "torch.float64" [.integer 1]) := by
  have hn : encodeTensor (vectorTensor "torch.int64" [.integer 1]) ≠
      encodeTensor (vectorTensor "torch.float64" [.integer 1]) := by
    intro h
    have ht := tensor_encoding_injective h
    have hd := congrArg Tensor.dtype ht
    simp [vectorTensor] at hd
  simp only [hn]

end N2P3.NegativePersonalState
