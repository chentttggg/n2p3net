import N2P3.PreprocessingFitScope
import Lean.Elab.Tactic.Omega

/-!
# Source QC threshold, interval and dependency-support mathematics

Source-fixed science_contract_03.json; continuous_qc.py:214-224,355-483,
489-510. Epoch and anomaly supports are half open; nominal epoch tmax is
converted by the caller to an exclusive stop with +1. Clustering timestamps
describe observation/confirmation, not an inferred true artifact onset.
Native statistics/rounding and complete Python execution remain separate.
-/
namespace N2P3.QCIntervalContracts
noncomputable section

def differenceFlag (difference center z scale : ℝ) : Prop := |difference-center| > z*scale

theorem equality_at_the_difference_limit_is_not_rejected
    (difference center z scale : ℝ) (atLimit : |difference-center|=z*scale) :
    ¬differenceFlag difference center z scale := by
  unfold differenceFlag
  rw [atLimit]
  exact lt_irrefl _

theorem an_unflagged_difference_is_within_the_declared_limit
    (difference center z scale : ℝ) (notFlagged : ¬differenceFlag difference center z scale) :
    |difference-center| ≤ z*scale := le_of_not_gt notFlagged

theorem positive_fitted_scale_and_multiplier_make_a_positive_limit
    (z scale : ℝ) (hz : 0 < z) (hs : 0 < scale) : 0 < z*scale := mul_pos hz hs

def overlaps (start stop anomalyStart anomalyStop : ℤ) : Prop := start < anomalyStop ∧ anomalyStart < stop

theorem half_open_epoch_and_anomaly_touching_at_either_endpoint_do_not_overlap
    (start stop anomalyStart anomalyStop : ℤ) (touch : stop=anomalyStart ∨ start=anomalyStop) :
    ¬overlaps start stop anomalyStart anomalyStop := by
  unfold overlaps
  omega

theorem extending_an_epoch_cannot_remove_an_existing_overlap
    (start stop earlier later anomalyStart anomalyStop : ℤ)
    (left : earlier ≤ start) (right : stop ≤ later) (bad : overlaps start stop anomalyStart anomalyStop) :
    overlaps earlier later anomalyStart anomalyStop := by
  unfold overlaps at *
  omega

structure Interval where
  start : ℤ
  stop : ℤ
  affects : String → Prop

def invalid (start stop : ℤ) (channel : String) (rows : List Interval) : Prop :=
  ∃ row ∈ rows, row.affects channel ∧ overlaps start stop row.start row.stop

theorem adding_applicable_anomaly_intervals_cannot_restore_an_invalid_channel
    (start stop : ℤ) (channel : String) (rows more : List Interval) (included : rows ⊆ more)
    (alreadyBad : invalid start stop channel rows) : invalid start stop channel more := by
  obtain ⟨row,member,affected,overlap⟩ := alreadyBad
  exact ⟨row,included member,affected,overlap⟩

def dependencyObserved {Native : Type*} (weights : Native → ℝ) (observed : Native → Prop) : Prop :=
  (∃ parent, weights parent ≠ 0) ∧ ∀ parent, weights parent ≠ 0 → observed parent

theorem an_unfitted_zero_mixing_row_is_not_an_observed_output
    {Native : Type*} (observed : Native → Prop) : ¬dependencyObserved (fun _ => (0 : ℝ)) observed := by
  simp [dependencyObserved]

theorem a_missing_nonzero_native_dependency_excludes_the_output
    {Native : Type*} (weights : Native → ℝ) (observed : Native → Prop) (parent : Native)
    (used : weights parent ≠ 0) (missing : ¬observed parent) : ¬dependencyObserved weights observed := by
  rintro ⟨_,allParents⟩
  exact missing (allParents parent used)

def clusterStop (segmentEnd last settle : ℕ) : ℕ := min segmentEnd (last+settle+1)

theorem every_clustered_flag_precedes_the_clipped_exclusive_stop
    (segmentEnd first last settle flagged : ℕ) (hlast : last < segmentEnd)
    (ordered : first ≤ flagged ∧ flagged ≤ last) : first ≤ flagged ∧ flagged < clusterStop segmentEnd last settle := by
  unfold clusterStop
  omega

theorem clipped_cluster_stop_never_crosses_the_acquisition_segment_end
    (segmentEnd last settle : ℕ) : clusterStop segmentEnd last settle ≤ segmentEnd := Nat.min_le_left _ _

def detected (segmentEnd first lookahead : ℕ) : ℕ := min (segmentEnd-1) (first+lookahead)
def confirmed (segmentEnd stop lookahead : ℕ) : ℕ := min (segmentEnd-1) (stop-1+lookahead)

theorem cluster_detection_is_not_later_than_confirmation
    (segmentEnd first last settle lookahead : ℕ) (hlast : last < segmentEnd) (ordered : first ≤ last) :
    detected segmentEnd first lookahead ≤ confirmed segmentEnd (clusterStop segmentEnd last settle) lookahead := by
  unfold detected confirmed clusterStop
  omega

def guardedStart (segmentStart first guard : ℕ) : ℕ := max segmentStart (first-guard)

theorem precautionary_pre_guard_stays_inside_the_source_segment
    (segmentStart first guard : ℕ) (inside : segmentStart ≤ first) :
    segmentStart ≤ guardedStart segmentStart first guard ∧ guardedStart segmentStart first guard ≤ first := by
  unfold guardedStart
  omega

theorem a_finite_pre_guard_does_not_bound_an_unknown_true_onset :
    guardedStart 0 40 5=35 ∧ 0 < guardedStart 0 40 5 := by decide

theorem exact_constant_detection_requires_observing_the_declared_run
    (start stop minimum : ℕ) (positive : 0 < minimum) (longEnough : minimum ≤ stop-start) :
    start ≤ start+minimum-1 ∧ start+minimum-1 < stop := by omega

end
end N2P3.QCIntervalContracts
