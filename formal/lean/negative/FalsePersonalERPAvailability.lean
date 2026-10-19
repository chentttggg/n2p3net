import N2P3.PersonalERPAvailability

/-! Fifteen source-derived wrong alternatives for observed support, strict
metadata, QC routing, minimum count, ordered views, fit scope and scatter.
Every example must reduce to exactly one unsolved False, with no other error. -/

namespace N2P3.NegativePersonalERPAvailability

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPAvailability

noncomputable section

-- A trial observation does not restore a globally absent channel.
example : observed (N:=1) (C:=1) (fun _ => false) (some (fun _ _ => true)) 0 0=true := by
  change false=true
  simp only [Bool.false_eq_true]

-- Numeric one is not Python's strict `is True` metadata tag.
example : channelLocal (localProcessing (.text "applied") (.integer 1) (.number (1/2)))=.ok true := by
  rw [numeric_one_is_not_strict_local]
  have hn : (.ok false : Check Bool) ≠ .ok true := by
    intro h
    have he := Except.ok.inj h
    cases he
  simp only [hn]

-- Inference local availability additionally requires applied status.
example : inferenceLocal (localProcessing (.text "not_applied") (.boolean true) (.number (1/2)))=.ok true := by
  have h := inactive_inference_is_not_local
    (localProcessing (.text "not_applied") (.boolean true) (.number (1/2)))
    (by simp [localProcessing,fieldDefault,Fields.ofList,Fields.get])
  rw [h]
  have hn : (.ok false : Check Bool) ≠ .ok true := by
    intro h
    have he := Except.ok.inj h
    cases he
  simp only [hn]

-- The explicit scalar ceiling witness for three half-observed channels is two.
example : minimumCount fixtureMinimum
    (localProcessing (.text "applied") (.boolean true) (.number (1/2))) 3=.ok 1 := by
  rw [applied_true_minimum_calls_only_scalar_ceil]
  have h : fixtureMinimum 3 (1/2)=.ok 2 := by simp [fixtureMinimum]
  rw [h]
  have hn : (.ok 2 : Check ℤ) ≠ .ok 1 := by
    intro h
    have he := Except.ok.inj h
    norm_num at he
  simp only [hn]

-- Original all-observed input does not override the fixed post-QC missing channel.
example : (fun i => (!fixtureQC.drop i && !fixtureQC.allBad i) &&
    countAvailable 2 fixtureQC.mask i && allSelected fixtureQC.mask fixtureSelected i) 1=true := by
  rw [fixture_quality_availability_is_true_false_false]
  change false=true
  simp only [Bool.false_eq_true]

-- Applied epoch QC is refused before another fixed QC result can be consumed.
example : prepare fixtureRaw (fun _ _ => true) (some fixtureQC) (.text "applied") ≠
    .error (.sourceReject "epoch QC applied twice") := by
  rw [prepare_refuses_double_quality]
  simp only [ne_eq,not_true_eq_false]

-- Without QC/local mode even a previously unavailable requested row can refuse the whole call.
example : selectionGate false false (fun _ : Fin 3 => false) fixturePostMask fixtureSelected=
    .ok (fun _ => false) := by
  rw [hard_refuses_even_previously_unavailable _ _ _ 1 2
    (by simp [fixtureSelected]) (by rfl)]
  have hn : (.error (.sourceReject "selected channels missing") : Check (Mask 3)) ≠
      .ok (fun _ => false) := by intro h; cases h
  simp only [hn]

-- An all-unavailable batch returns its N NaN slots before the failing backend.
example : dispatch fixtureSelected (fun _ _ => .exception "must not call") fixtureRaw
    (fun _ => false)=.exception "must not call" := by
  rw [no_available_rows_returns_all_nan_without_backend _ _ _ _ (by intro i; rfl)]
  have hn : (.finite (fun _ : Fin 3 => none) : NumericOutcome (Fin 3 → Option ℝ)) ≠
      .exception "must not call" := by intro h; cases h
  simp only [hn]

-- Scatter preserves the requested three-row denominator instead of the one retained row.
example : (List.ofFn (scatter ([0] : List (Fin 3)) (fun _ => some (53 : ℝ)))).length=1 := by
  rw [scatter_requested_denominator]
  have hn : (3 : ℕ) ≠ 1 := by decide
  simp only [hn]

-- Sorting a saved selected-channel list would change this source-shaped score.
example : ERPFeatures.evaluate ERPFeatures.fixtureState (ERPFeatures.binFeatures 2
    (orderedView fixtureRaw ([0] : List (Fin 3)) [2,0] ⟨0,by simp⟩))=53 := by
  rw [fixture_reversed_selected_score_51]
  have hn : (51 : ℝ) ≠ 53 := by norm_num
  simp only [hn]

-- Training's union fallback has no applied-status guard.
example : trainingSelection fixtureMinimum
    (localProcessing (.text "not_applied") (.boolean true) (.number (1/2)))
    (N:=6) (fun _ => true) trainingFixtureMask=
      .ok ⟨[],fun _ => true,false⟩ := by
  rw [fallback_without_applied_status_is_executed]
  have hn : (.ok (⟨[0,1],fun i => decide (i.val<4),true⟩ : TrainingSelection 6 2) :
      Check (TrainingSelection 6 2)) ≠ .ok ⟨[],fun _ => true,false⟩ := by
    intro h
    have hs := Except.ok.inj h
    have hc := congrArg TrainingSelection.channels hs
    cases hc
  simp only [hn]

-- Nonempty input training authority does not guarantee nonempty complete-union rows.
example : (match trainingSelection fixtureMinimum
    (localProcessing (.text "not_applied") (.boolean true) (.number (1/2)))
    (N:=2) (fun _ => true) emptyFallbackMask with
  | .ok selection => ∃ i, selection.rows i=true
  | .error _ => False) := by
  rw [training_union_fallback_may_return_no_rows]
  simp only [Bool.false_eq_true,exists_false]

-- The cleaned position is not the original input row used by the training mask.
example : remapTraining (![2,0] : Fin 2 → Fin 3) (![true,false,false] : Mask 3) 0=true := by
  change false=true
  simp only [Bool.false_eq_true]

-- Routing countermodel only: availability adds no finite gate to an arbitrary
-- backend. This does not exhibit an accepted current-version xDAWN state.
example : dispatch fixtureSelected nanBackend fixtureRaw fixtureAvailability ≠
    .finite (fun _ => none) := by
  rw [availability_does_not_imply_finite_output]
  simp only [ne_eq,not_true_eq_false]

-- Values alone cannot justify noninterference when the QC availability result changes.
example : run fixtureMinimum (localProcessing (.text "applied") (.boolean true) (.number (1/2)))
    (.text "not_applied") (some fixtureQC) fixtureRaw (fun _ _ => true)
    fixtureSelected (affineBackend 2 ERPFeatures.fixtureState)=
  run fixtureMinimum (localProcessing (.text "applied") (.boolean true) (.number (1/2)))
    (.text "not_applied") (some fixtureChangedQC) fixtureRaw (fun _ _ => true)
    fixtureSelected (affineBackend 2 ERPFeatures.fixtureState) := by
  rw [fixture_full_quality_run,fixture_changed_qc_full_run]
  have hn : (.ok (.finite (![some 53,none,none] : Fin 3 → Option ℝ)) :
      Check (NumericOutcome (Fin 3 → Option ℝ))) ≠ .ok (.finite (fun _ => none)) := by
    intro h
    have hf := NumericOutcome.finite.inj (Except.ok.inj h)
    have hi := congrFun hf 0
    simp at hi
  simp only [hn]

end

end N2P3.NegativePersonalERPAvailability
