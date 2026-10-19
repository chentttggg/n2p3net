import N2P3.PersonalERPFitOutcome

/-!
# Literal ERP output dictionaries and final array packing

Source: structured_erp.py:97-123; calibration_candidates.py:311-321.
Numeric arrays are individual decoded operation outputs. Their key names,
optional basis insertion, rank attributes and final dictionary traversal are
constructed here. No whole fitted-artifact validity or arbitrary list key
uniqueness premise is supplied. Casts retain their individual error boundary;
this does not prove their IEEE or byte representation.
-/

namespace N2P3.PersonalERPOutputFields
open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalERPFitOutcome

noncomputable section
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

inductive BodyValues where
  | linear (coefficient intercept : Array64) (basis : Option Array64)
  | xdawn (filters prototypes reference coefficient intercept : Array64)
      (observedRank effectiveNfilter : ℕ)

def payload : BodyValues → Payload
  | .linear coefficient intercept basis =>
      ⟨[("coefficient",coefficient),("intercept",intercept)] ++
        (basis.map (fun matrix => [("temporal_basis",matrix)])).getD [],none,none⟩
  | .xdawn filters prototypes reference coefficient intercept rank effective =>
      ⟨[("filters",filters),("prototypes",prototypes),("reference",reference),
        ("coefficient",coefficient),("intercept",intercept)],some rank,some effective⟩

def keys (arrays : Arrays) : List String := arrays.map Prod.fst

def pack (castCopy : String → Array64 → Check Array64) (input : Payload) : Check Payload := do
  let arrays ← castAll castCopy input.arrays
  pure { input with arrays := arrays }

theorem literal_body_keys_are_unique (values : BodyValues) : (keys (payload values).arrays).Nodup := by
  cases values with
  | linear coefficient intercept basis => cases basis <;> simp [keys,payload]
  | xdawn filters prototypes reference coefficient intercept rank effective => simp [keys,payload]

theorem all_array_casts_preserve_key_order (castCopy : String → Array64 → Check Array64)
    (arrays copied : Arrays) (h : castAll castCopy arrays = .ok copied) : keys copied = keys arrays := by
  induction arrays generalizing copied with
  | nil => simpa [castAll,keys] using h.symm
  | cons entry tail ih =>
    rcases entry with ⟨name,array⟩
    cases hc : castCopy name array with
    | error fault => simp [castAll,hc] at h
    | ok converted =>
      cases ht : castAll castCopy tail with
      | error fault => simp [castAll,hc,ht] at h
      | ok convertedTail =>
        have he : (name,converted)::convertedTail=copied := by simpa [castAll,hc,ht] using h
        rw [←he]
        simpa [keys] using congrArg (List.cons name) (ih convertedTail ht)

theorem packing_preserves_key_order (castCopy : String → Array64 → Check Array64)
    (input output : Payload) (h : pack castCopy input=.ok output) :
    keys output.arrays=keys input.arrays := by
  cases hc : castAll castCopy input.arrays with
  | error fault => simp [pack,hc] at h
  | ok arrays =>
    have he : { input with arrays := arrays }=output := by simpa [pack,hc] using h
    rw [←he]
    exact all_array_casts_preserve_key_order castCopy input.arrays arrays hc

theorem packing_preserves_rank_attributes (castCopy : String → Array64 → Check Array64)
    (input output : Payload) (h : pack castCopy input=.ok output) :
    output.observedRank=input.observedRank ∧ output.effectiveNfilter=input.effectiveNfilter := by
  cases hc : castAll castCopy input.arrays with
  | error fault => simp [pack,hc] at h
  | ok arrays =>
    have he : { input with arrays := arrays }=output := by simpa [pack,hc] using h
    rw [←he]
    exact ⟨rfl,rfl⟩

theorem packed_literal_body_keys_are_unique (castCopy : String → Array64 → Check Array64)
    (values : BodyValues) (output : Payload) (h : pack castCopy (payload values)=.ok output) :
    (keys output.arrays).Nodup := by
  rw [packing_preserves_key_order castCopy (payload values) output h]
  exact literal_body_keys_are_unique values

theorem producer_key_order (arrays : Arrays) (selector : N2P3.PersonalState.Tensor) :
    (producer arrays selector).map Prod.fst =
      (keys arrays).map (fun name => "erp." ++ name) ++ ["waveform_channels"] := by
  simp [producer,keys,List.map_map,Function.comp_def]

theorem packed_producer_dictionary_keys_are_unique (castCopy : String → Array64 → Check Array64)
    (values : BodyValues) (output : Payload) (selector : N2P3.PersonalState.Tensor)
    (h : pack castCopy (payload values)=.ok output) :
    ((producer output.arrays selector).map Prod.fst).Nodup := by
  rw [producer_key_order,packing_preserves_key_order castCopy (payload values) output h]
  cases values with
  | linear coefficient intercept basis => cases basis <;> simp [keys,payload]
  | xdawn filters prototypes reference coefficient intercept rank effective => simp [keys,payload]

theorem clone_preserves_dictionary_key_order (state : State) :
    (cloneState state).map Prod.fst=state.map Prod.fst := by
  simp [cloneState,List.map_map,Function.comp_def]

theorem cloned_packed_producer_keys_are_unique (castCopy : String → Array64 → Check Array64)
    (values : BodyValues) (output : Payload) (selector : N2P3.PersonalState.Tensor)
    (h : pack castCopy (payload values)=.ok output) :
    ((cloneState (producer output.arrays selector)).map Prod.fst).Nodup := by
  rw [clone_preserves_dictionary_key_order]
  exact packed_producer_dictionary_keys_are_unique castCopy values output selector h

def rejectIntercept (name : String) (array : Array64) : Check Array64 :=
  if name="intercept" then .error (.fieldFailure "intercept cast") else .ok array

theorem later_cast_error_returns_no_partial_dictionary (coefficient intercept : Array64) :
    pack rejectIntercept (payload (.linear coefficient intercept none)) =
      .error (.fieldFailure "intercept cast") := by
  simp [pack,payload,castAll,rejectIntercept]

structure XdawnValues where
  filters : Array64
  prototypes : Array64
  reference : Array64
  coefficient : Array64
  intercept : Array64
  observedRank : ℕ
  effectiveNfilter : ℕ

/-- Numerical leaves return values/attributes, never dictionary keys or an
entire fitted model. The explicit header branch below chooses the fields. -/
structure EstimateCalls where
  linear : (r c t : ℕ) → N2P3.ERPState.Header →
    N2P3.PersonalERPFit.NumericBatch r c t → (Fin r → Bool) → Check (Array64 × Array64)
  xdawn : (r c t : ℕ) → N2P3.ERPState.Header →
    N2P3.PersonalERPFit.NumericBatch r c t → (Fin r → Bool) → Check XdawnValues
  basisView : N2P3.ERPState.TemporalModel → Check Array64

def computeBody (calls : EstimateCalls) (r c t : ℕ) (header : N2P3.ERPState.Header)
    (values : N2P3.PersonalERPFit.NumericBatch r c t) (labels : Fin r → Bool) : Check BodyValues := do
  if header.family = .xdawn then
    let output ← calls.xdawn r c t header values labels
    pure (.xdawn output.filters output.prototypes output.reference output.coefficient output.intercept
      output.observedRank output.effectiveNfilter)
  else
    let output ← calls.linear r c t header values labels
    let basis ← match header.temporal with
      | none => .ok none
      | some temporal => do
        let matrix ← calls.basisView temporal
        pure (some matrix)
    pure (.linear output.1 output.2 basis)

def estimateFields (calls : EstimateCalls) (r c t : ℕ) (header : N2P3.ERPState.Header)
    (values : N2P3.PersonalERPFit.NumericBatch r c t) (labels : Fin r → Bool) : Check Payload :=
  (computeBody calls r c t header values labels).map payload

def numerics (input : Numerics) (calls : EstimateCalls)
    (castCopy : String → Array64 → Check Array64) : Numerics :=
  { input with
    estimate := fun r c t header values labels => estimateFields calls r c t header values labels
    packFloat64 := pack castCopy }

theorem estimate_field_keys_follow_literal_constructors (calls : EstimateCalls) (r c t : ℕ)
    (header : N2P3.ERPState.Header) (values : N2P3.PersonalERPFit.NumericBatch r c t)
    (labels : Fin r → Bool) (output : Payload)
    (h : estimateFields calls r c t header values labels=.ok output) :
    (keys output.arrays).Nodup := by
  cases hc : computeBody calls r c t header values labels with
  | error fault => simp [estimateFields,hc] at h
  | ok body =>
    have he : payload body=output := by simpa [estimateFields,hc] using h
    rw [←he]
    exact literal_body_keys_are_unique body

theorem packed_estimate_follows_literal_field_order (calls : EstimateCalls)
    (castCopy : String → Array64 → Check Array64) (r c t : ℕ) (header : N2P3.ERPState.Header)
    (values : N2P3.PersonalERPFit.NumericBatch r c t) (labels : Fin r → Bool) (output : Payload)
    (h : (do let estimated ← estimateFields calls r c t header values labels
             pack castCopy estimated) = .ok output) :
    ∃ body, keys output.arrays=keys (payload body).arrays := by
  cases hc : computeBody calls r c t header values labels with
  | error fault => simp [estimateFields,hc] at h
  | ok body =>
    have hp : pack castCopy (payload body)=.ok output := by simpa [estimateFields,hc] using h
    exact ⟨body,packing_preserves_key_order castCopy (payload body) output hp⟩

/-- Postcondition of the fully exposed control graph, not a supplied fit
validity predicate: unfold fitStages, then derive fields in each source branch. -/
theorem actual_fit_program_follows_literal_field_order (input : Numerics) (calls : EstimateCalls)
    (castCopy : String → Array64 → Check Array64) {R C T : ℕ} (header : N2P3.ERPState.Header)
    (values : N2P3.ERPFeatures.Batch R C T) (labels : Fin R → Bool) (output : Payload)
    (h : fitStages (numerics input calls castCopy) header values labels=.ok output) :
    ∃ body, keys output.arrays=keys (payload body).arrays := by
  have success {α β : Type} (first : Check α) (next : α → Check β) (value : β)
      (hh : first.bind next=.ok value) : ∃ before, first=.ok before ∧ next before=.ok value := by
    cases first with
    | error fault => simp at hh
    | ok before => exact ⟨before,rfl,hh⟩
  unfold fitStages at h
  rcases success _ _ _ h with ⟨binned,_,h⟩
  rcases success _ _ _ h with ⟨_,_,h⟩
  cases ht : header.temporal with
  | none =>
    apply packed_estimate_follows_literal_field_order calls castCopy R C (T / header.binWidth) header binned labels output
    simpa [ht,numerics] using h
  | some temporal =>
    have branch :
        (do let projected ← temporalStage (numerics input calls castCopy) temporal binned
            let estimated ← estimateFields calls R C temporal.rank header projected labels
            pack castCopy estimated) = .ok output := by simpa [ht,numerics] using h
    rcases success _ _ _ branch with ⟨projected,_,hp⟩
    exact packed_estimate_follows_literal_field_order calls castCopy R C temporal.rank header projected labels output hp

theorem actual_fit_program_cannot_return_duplicate_keys (input : Numerics) (calls : EstimateCalls)
    (castCopy : String → Array64 → Check Array64) {R C T : ℕ} (header : N2P3.ERPState.Header)
    (values : N2P3.ERPFeatures.Batch R C T) (labels : Fin R → Bool) (output : Payload)
    (h : fitStages (numerics input calls castCopy) header values labels=.ok output) :
    (keys output.arrays).Nodup := by
  rcases actual_fit_program_follows_literal_field_order input calls castCopy header values labels output h with ⟨body,he⟩
  rw [he]
  exact literal_body_keys_are_unique body

theorem cloned_actual_fit_output_keys_are_unique (input : Numerics) (calls : EstimateCalls)
    (castCopy : String → Array64 → Check Array64) {R C T : ℕ} (header : N2P3.ERPState.Header)
    (values : N2P3.ERPFeatures.Batch R C T) (labels : Fin R → Bool) (output : Payload)
    (selector : N2P3.PersonalState.Tensor)
    (h : fitStages (numerics input calls castCopy) header values labels=.ok output) :
    ((cloneState (producer output.arrays selector)).map Prod.fst).Nodup := by
  rcases actual_fit_program_follows_literal_field_order input calls castCopy header values labels output h with ⟨body,he⟩
  rw [clone_preserves_dictionary_key_order,producer_key_order,he]
  cases body with
  | linear coefficient intercept basis => cases basis <;> simp [keys,payload]
  | xdawn filters prototypes reference coefficient intercept rank effective => simp [keys,payload]

end
end N2P3.PersonalERPOutputFields
