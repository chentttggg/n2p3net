import N2P3.PreprocessingFitScope
import Mathlib.Data.Finset.Max
import Mathlib.Algebra.Order.BigOperators.Group.List

/-!
# Actual eligible-fold accumulators and the least aggressive tolerance tie

Source artifact.py:279-319,378-411. Each eligible channel/candidate fold adds
one error and one count. Kappa first averages the currently valid channels
inside a fold, then gives that fold one count. None represents an ineligible
fold/cell, never an observed zero error. This is exact-real finite arithmetic;
NumPy infinity sentinels/rounding stay outside this mathematical model.
-/
namespace N2P3.ArtifactCVSelection
noncomputable section
attribute [local instance] Classical.propDecidable

def accumulate : List (Option ℝ) → ℝ × ℕ
  | [] => (0,0)
  | none :: tail => accumulate tail
  | some error :: tail => (error+(accumulate tail).1,(accumulate tail).2+1)

theorem accumulation_is_exactly_eligible_error_sum_and_eligible_fold_count (rows : List (Option ℝ)) :
    (accumulate rows).1=(rows.filterMap id).sum ∧ (accumulate rows).2=(rows.filterMap id).length := by
  induction rows with
  | nil => simp [accumulate]
  | cons head tail ih => cases head <;> simp [accumulate,ih]

theorem accumulated_independent_fold_lists_add_their_errors_and_counts (first second : List (Option ℝ)) :
    accumulate (first++second)=((accumulate first).1+(accumulate second).1,
      (accumulate first).2+(accumulate second).2) := by
  induction first with
  | nil => simp [accumulate]
  | cons head tail ih => cases head <;> simp [accumulate,ih,add_assoc,Nat.add_comm,Nat.add_left_comm]

def cvMean (rows : List (Option ℝ)) : Option ℝ :=
  if (accumulate rows).2=0 then none else some ((accumulate rows).1/((accumulate rows).2 : ℝ))

theorem adding_an_ineligible_fold_changes_neither_score_nor_evidence (rows : List (Option ℝ)) :
    accumulate (rows++[none])=accumulate rows ∧ cvMean (rows++[none])=cvMean rows := by
  have same : accumulate (rows++[none])=accumulate rows := by
    rw [accumulated_independent_fold_lists_add_their_errors_and_counts]
    simp [accumulate]
  exact ⟨same,by simp only [cvMean,same]⟩

theorem nonnegative_eligible_errors_have_a_nonnegative_returned_mean
    (rows : List (Option ℝ)) (value : ℝ)
    (nonnegative : ∀ error ∈ rows.filterMap id, 0 ≤ error) (returned : cvMean rows=some value) : 0 ≤ value := by
  have total : 0 ≤ (accumulate rows).1 := by
    rw [(accumulation_is_exactly_eligible_error_sum_and_eligible_fold_count rows).1]
    exact List.sum_nonneg nonnegative
  by_cases zero : (accumulate rows).2=0
  · simp [cvMean,zero] at returned
  · have equal : (accumulate rows).1/((accumulate rows).2 : ℝ)=value := by
      simpa only [cvMean,if_neg zero,Option.some.injEq] using returned
    rw [← equal]
    exact div_nonneg total (Nat.cast_nonneg _)

def channelMean : List ℝ → Option ℝ
  | [] => none
  | head :: tail => some ((head+tail.sum)/((tail.length+1 : ℕ) : ℝ))

theorem kappa_counts_each_eligible_fold_once_regardless_of_its_channel_count (foldChannels : List (List ℝ)) :
    (accumulate (foldChannels.map channelMean)).2=
      (foldChannels.filter (fun channels => !channels.isEmpty)).length := by
  induction foldChannels with
  | nil => rfl
  | cons channels tail ih => cases channels <;> simp [channelMean,accumulate,ih]

theorem equal_fold_weighting_is_not_pooled_channel_weighting :
    cvMean [channelMean [0],channelMean [3,3,3]]=some (3/2) ∧
      channelMean [0,3,3,3]=some (9/4) := by
  norm_num [cvMean,accumulate,channelMean]

def bestScore (supported : Finset ℕ) (score : ℕ → ℝ) (nonempty : supported.Nonempty) : ℝ :=
  (supported.image score).min' (nonempty.image score)

theorem the_best_supported_score_is_attained_by_an_actual_supported_candidate
    (supported : Finset ℕ) (score : ℕ → ℝ) (nonempty : supported.Nonempty) :
    ∃ candidate ∈ supported, score candidate=bestScore supported score nonempty := by
  have member := Finset.min'_mem (supported.image score) (nonempty.image score)
  exact Finset.mem_image.mp member

def ties (supported : Finset ℕ) (score : ℕ → ℝ) (tolerance : ℝ) (nonempty : supported.Nonempty) : Finset ℕ :=
  supported.filter (fun candidate => score candidate ≤ bestScore supported score nonempty+tolerance)

theorem nonnegative_tolerance_leaves_at_least_the_best_supported_candidate
    (supported : Finset ℕ) (score : ℕ → ℝ) (tolerance : ℝ) (nonempty : supported.Nonempty)
    (nonnegative : 0 ≤ tolerance) : (ties supported score tolerance nonempty).Nonempty := by
  obtain ⟨candidate,member,best⟩ := the_best_supported_score_is_attained_by_an_actual_supported_candidate supported score nonempty
  refine ⟨candidate,Finset.mem_filter.mpr ⟨member,?_⟩⟩
  rw [best]
  exact le_add_of_nonneg_right nonnegative

def selectSupported (supported : Finset ℕ) (score : ℕ → ℝ) (tolerance : ℝ)
    (nonempty : supported.Nonempty) (nonnegative : 0 ≤ tolerance) : ℕ :=
  (ties supported score tolerance nonempty).max'
    (nonnegative_tolerance_leaves_at_least_the_best_supported_candidate supported score tolerance nonempty nonnegative)

theorem tolerance_selection_returns_the_largest_index_with_supported_near_minimum_error
    (supported : Finset ℕ) (score : ℕ → ℝ) (tolerance : ℝ) (nonempty : supported.Nonempty)
    (nonnegative : 0 ≤ tolerance) :
    selectSupported supported score tolerance nonempty nonnegative ∈ supported ∧
    score (selectSupported supported score tolerance nonempty nonnegative) ≤ bestScore supported score nonempty+tolerance ∧
    ∀ other ∈ supported, score other ≤ bestScore supported score nonempty+tolerance →
      other ≤ selectSupported supported score tolerance nonempty nonnegative := by
  have member := Finset.max'_mem (ties supported score tolerance nonempty)
    (nonnegative_tolerance_leaves_at_least_the_best_supported_candidate supported score tolerance nonempty nonnegative)
  refine ⟨(Finset.mem_filter.mp member).1,(Finset.mem_filter.mp member).2,?_⟩
  intro other present near
  exact Finset.le_max' _ other (Finset.mem_filter.mpr ⟨present,near⟩)

def supportedCandidates (grid : Finset ℕ) (observations : ℕ → List (Option ℝ)) : Finset ℕ :=
  grid.filter (fun candidate => 0 < (accumulate (observations candidate)).2)

theorem supported_selection_requires_actual_eligible_fold_evidence
    (grid : Finset ℕ) (observations : ℕ → List (Option ℝ)) (score : ℕ → ℝ) (tolerance : ℝ)
    (nonempty : (supportedCandidates grid observations).Nonempty) (nonnegative : 0 ≤ tolerance) :
    let selected := selectSupported (supportedCandidates grid observations) score tolerance nonempty nonnegative
    selected ∈ grid ∧ (observations selected).filterMap id ≠ [] := by
  dsimp only
  have member := (tolerance_selection_returns_the_largest_index_with_supported_near_minimum_error
    (supportedCandidates grid observations) score tolerance nonempty nonnegative).1
  have both := Finset.mem_filter.mp member
  refine ⟨both.1,?_⟩
  have count := both.2
  rw [(accumulation_is_exactly_eligible_error_sum_and_eligible_fold_count _).2] at count
  exact List.ne_nil_of_length_pos count

def selectOrFallback (supported : Finset ℕ) (score : ℕ → ℝ) (tolerance : ℝ)
    (nonnegative : 0 ≤ tolerance) (lastGridIndex : ℕ) : ℕ :=
  if nonempty : supported.Nonempty then selectSupported supported score tolerance nonempty nonnegative else lastGridIndex

def candidateScore (observations : ℕ → List (Option ℝ)) (candidate : ℕ) : ℝ :=
  (accumulate (observations candidate)).1/((accumulate (observations candidate)).2 : ℝ)

def sourceChoice (grid : Finset ℕ) (observations : ℕ → List (Option ℝ)) (tolerance : ℝ)
    (nonnegative : 0 ≤ tolerance) (lastGridIndex : ℕ) : ℕ :=
  selectOrFallback (supportedCandidates grid observations) (candidateScore observations) tolerance nonnegative lastGridIndex

theorem actual_supported_choice_uses_the_actual_eligible_fold_mean_and_largest_admissible_index
    (grid : Finset ℕ) (observations : ℕ → List (Option ℝ)) (tolerance : ℝ)
    (nonnegative : 0 ≤ tolerance) (lastGridIndex : ℕ)
    (nonempty : (supportedCandidates grid observations).Nonempty) :
    let selected := sourceChoice grid observations tolerance nonnegative lastGridIndex
    selected ∈ grid ∧ cvMean (observations selected)=some (candidateScore observations selected) ∧
    ∀ other ∈ supportedCandidates grid observations,
      candidateScore observations other ≤ bestScore (supportedCandidates grid observations) (candidateScore observations) nonempty+tolerance →
      other ≤ selected := by
  dsimp only
  have selected : sourceChoice grid observations tolerance nonnegative lastGridIndex=
      selectSupported (supportedCandidates grid observations) (candidateScore observations) tolerance nonempty nonnegative := by
    simp only [sourceChoice,selectOrFallback,dif_pos nonempty]
  rw [selected]
  have property := tolerance_selection_returns_the_largest_index_with_supported_near_minimum_error
    (supportedCandidates grid observations) (candidateScore observations) tolerance nonempty nonnegative
  have actual := Finset.mem_filter.mp property.1
  have nonzero : (accumulate (observations
      (selectSupported (supportedCandidates grid observations) (candidateScore observations) tolerance nonempty nonnegative))).2 ≠ 0 :=
    Nat.ne_of_gt actual.2
  exact ⟨actual.1,by simp only [cvMean,if_neg nonzero,candidateScore],property.2.2⟩

theorem no_supported_CV_candidate_uses_the_declared_last_grid_fallback
    (score : ℕ → ℝ) (tolerance : ℝ) (nonnegative : 0 ≤ tolerance) (lastGridIndex : ℕ) :
    selectOrFallback ∅ score tolerance nonnegative lastGridIndex=lastGridIndex := by
  simp [selectOrFallback]

theorem the_source_relative_tolerance_is_positive_for_nonnegative_loss
    (best epsilon : ℝ) (nonnegative : 0 ≤ best) (positive : 0 < epsilon) :
    0 < best*(1/1000000)+epsilon :=
  add_pos_of_nonneg_of_pos (mul_nonneg nonnegative (by norm_num)) positive

end
end N2P3.ArtifactCVSelection
