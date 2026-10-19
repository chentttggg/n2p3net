import N2P3.EventDesign

/-!
Counterexamples use the operational event fold, not an independently guessed
convolution. All onsets are natural, strictly ordered and have complete tails.
They refute overwriting overlap, linearity of the complete two-block design,
and invariance when an actual event is omitted. Exact real accumulation only.
-/

namespace N2P3.EventDesign.Negative

private def B : Basis 3 1 := fun lag _ => (lag.val + 1 : ℕ)

private def onset (e : Fin 2) : ℕ := e.val + 1

private def label (e : Fin 2) : ℝ := if e = 0 then 1 else 0

private theorem source_overlap_value :
    design B onset label [0, 1] (2 : Fin 8) (.inl 0) = 3 := by
  norm_num [design, loopFrom, sliceUpdate, retained, B, onset, label]

private theorem zero_labels_keep_common :
    design B onset (fun _ => 0) [0, 1] (2 : Fin 8) (.inl 0) = 3 := by
  norm_num [design, loopFrom, sliceUpdate, retained, B, onset]

private theorem omitted_event_value :
    design B onset label [0] (2 : Fin 8) (.inl 0) = 2 := by
  norm_num [design, loopFrom, sliceUpdate, retained, B, onset, label]

/-- False: the most recent overlapping event overwrites the existing basis row. -/
example : design B onset label [0, 1] (2 : Fin 8) (.inl 0) = 1 := by
  rw [source_overlap_value]
  norm_num

/-- False: the entire label-to-design map is linear, so zero labels give zero design. -/
example : (design B onset (fun _ => 0) [0, 1] : Design 8 1) = 0 := by
  have hne : (design B onset (fun _ => 0) [0, 1] : Design 8 1) ≠ 0 := by
    intro h
    have hv := congrArg (fun D : Design 8 1 => D 2 (.inl 0)) h
    change design B onset (fun _ => 0) [0, 1] (2 : Fin 8) (.inl 0) = 0 at hv
    rw [zero_labels_keep_common] at hv
    norm_num at hv
  simp only [hne]

/-- False: dropping a response-generating event leaves the source design unchanged. -/
example : (design B onset label [0, 1] : Design 8 1) = design B onset label [0] := by
  have hne : (design B onset label [0, 1] : Design 8 1) ≠ design B onset label [0] := by
    intro h
    have hv := congrArg (fun D : Design 8 1 => D 2 (.inl 0)) h
    change design B onset label [0, 1] (2 : Fin 8) (.inl 0) =
      design B onset label [0] (2 : Fin 8) (.inl 0) at hv
    rw [source_overlap_value, omitted_event_value] at hv
    norm_num at hv
  simp only [hne]

end N2P3.EventDesign.Negative
