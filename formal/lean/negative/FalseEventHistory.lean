import N2P3.EventHistory

/-! False claims obtained by deleting a required history ingredient. Every
antecedent is independently proved; the final diagnostic must be False. -/

namespace N2P3.EventHistory.Negative

private def planned : Finset (Fin 2) := {0}
private def substituted : Finset (Fin 2) := {1}

private theorem substituted_counts_match :
    ∀ f : Unit, histogram substituted (fun _ => ()) f = histogram planned (fun _ => ()) f := by
  intro f
  cases f
  norm_num [histogram, substituted, planned]

private theorem substituted_is_different : substituted ≠ planned := by
  intro h
  have hm : (0 : Fin 2) ∈ substituted := by rw [h]; simp [planned]
  norm_num [substituted] at hm

/-- False: matching per-flash counts recover identities without subset. -/
example : (∀ f : Unit, histogram substituted (fun _ => ()) f =
    histogram planned (fun _ => ()) f) → substituted = planned := by
  intro _
  simp only [substituted_is_different]

private def badOnset (e : Event 3) : ℤ :=
  if e.1 = 0 then if e.2.1 = 9 ∧ e.2.2 = 11 then 0 else 100
  else if e.1 = 1 then 100 else 200

private def B : EventDesign.Basis 1 1 := fun _ _ => 1

private theorem unordered_neighbors_pass :
    (geometry (1 : Fin 3) badOnset 1 0 0 400).reason
      (coverageFor (1 : Fin 3) Finset.univ canonicalCode) = none := by
  rw [CandidateCoverage.Geometry.success_iff]
  refine ⟨canonical_full_coverage _, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [geometry, badOnset, CandidateCoverage.Geometry.start,
      CandidateCoverage.Geometry.stop, CandidateCoverage.Geometry.previousOverlap,
      CandidateCoverage.Geometry.nextOverlap, firstEvent, lastEvent]

private theorem unordered_external_response :
    EventDesign.kernelAt B (badOnset (0, 0, 0)).toNat 100 0 = 1 := by
  have hbad : badOnset (0, 0, 0) = 100 := by decide
  rw [hbad]
  change EventDesign.kernelAt B 100 100 0 = 1
  norm_num [EventDesign.kernelAt, B]

/-- False: adjacent gates alone exclude all external events without global order.
The previous selection's first event is at 100 while its last event is at 0. -/
example : (geometry (1 : Fin 3) badOnset 1 0 0 400).reason
    (coverageFor (1 : Fin 3) Finset.univ canonicalCode) = none →
    EventDesign.kernelAt B (badOnset (0, 0, 0)).toNat 100 0 = 0 := by
  rw [unordered_neighbors_pass]
  simp only [true_implies, unordered_external_response]
  norm_num

/-- False: a width-one finite causal response contains its exclusive endpoint. -/
example : EventDesign.kernelAt B 100 101 0 = 1 := by
  rw [EventDesign.kernelAt_zero_after B 100 101 0 (by norm_num)]
  norm_num

end N2P3.EventHistory.Negative
