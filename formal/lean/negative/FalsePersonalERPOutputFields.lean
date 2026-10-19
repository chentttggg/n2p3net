import N2P3.PersonalERPOutputFields

/-! Five source-derived wrong alternatives fixed in output_fields_contract_01.
Each must fail with one unsolved False and no auxiliary diagnostic. -/
namespace N2P3.NegativePersonalERPOutputFields
open N2P3.FrozenState N2P3.PersonalERPFitOutcome N2P3.PersonalERPOutputFields

example (coefficient intercept : Array64) :
    keys (payload (.linear coefficient intercept none)).arrays = ["coefficient","coefficient"] := by
  simp [keys,payload]

example (coefficient intercept basis : Array64) :
    keys (payload (.linear coefficient intercept (some basis))).arrays = ["coefficient","intercept"] := by
  simp [keys,payload]

example (filters prototypes reference coefficient intercept : Array64) (rank effective : ℕ) :
    keys (payload (.xdawn filters prototypes reference coefficient intercept rank effective)).arrays =
      ["coefficient","intercept"] := by
  simp [keys,payload]

example (coefficient intercept : Array64) :
    pack rejectIntercept (payload (.linear coefficient intercept none)) =
      .ok ⟨[("coefficient",coefficient)],none,none⟩ := by
  rw [later_cast_error_returns_no_partial_dictionary]
  have hn : (Except.error (.fieldFailure "intercept cast") : N2P3.StateLedger.Check Payload) ≠
      .ok ⟨[("coefficient",coefficient)],none,none⟩ := by intro h; cases h
  simp only [hn]

example (array : Array64) : (keys [("coefficient",array),("coefficient",array)]).Nodup := by
  simp [keys]

end N2P3.NegativePersonalERPOutputFields
