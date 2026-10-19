import N2P3.ERPTemporalCache

namespace N2P3.NegativeERPTemporalCache

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPTemporalCache

example (digest : String) (fault : LedgerFault) :
    chooseDecodedDigest (some digest) (.error fault)=.error fault := by
  rw [stored_decoded_digest_does_not_rerun_content_hash]
  have h : (Except.ok digest : Check String) ≠ .error fault := by intro he; cases he
  simp only [h]

example (attestation : Fields) (before after : FileStat) (digest : String) (size : ℕ)
    (changed : fileIdentity after ≠ fileIdentity before) :
    checkAfterCopy attestation before after digest size=
      .error (.sourceReject "cache copied SHA256") := by
  rw [changed_file_identity_refused_before_size_and_hash attestation before after digest size changed]
  have h : (Except.error (.sourceReject "cache changed during snapshot copy") : Check Unit) ≠
      .error (.sourceReject "cache copied SHA256") := by
    intro equality
    simp at equality
  simp only [h]

end N2P3.NegativeERPTemporalCache
