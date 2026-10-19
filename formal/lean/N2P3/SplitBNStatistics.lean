import N2P3.ObservedStatistics
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

/-!
# Source-weighted SplitBN statistics with the actual near-unit mass

Source: splitbn.py:75-100,432-445. Config permits an absolute 1e-9
mass tolerance. Evaluation first casts through running_mean.new_tensor and
then uses the effective tensor weights directly, without renormalization.
Exact identities concern those effective weight values. Config tolerance
does not establish an effective-mass bound after native cast/reduction.
Per-feature finite exact-real observations and individual batch counts are
used. Native IEEE/isclose and the full lifecycle remain separate.
Tracked counts are signed observations. The literal zero-test does not prove
positive/reachable counts or genuine training provenance.
Contract: v07_splitbn_statistics_contract_03.json, fixed before revised checks.
-/
namespace N2P3.SplitBNStatistics
open scoped BigOperators
noncomputable section
attribute [local instance] Classical.propDecidable
variable {D : ℕ}

def mass (weights : Fin D → ℝ) : ℝ := ∑ domain, weights domain
def mean (weights means : Fin D → ℝ) : ℝ := ∑ domain, weights domain * means domain
def second (weights means variances : Fin D → ℝ) : ℝ :=
  ∑ domain, weights domain * (variances domain + means domain ^ 2)
def rawVariance (weights means variances : Fin D → ℝ) : ℝ :=
  second weights means variances - mean weights means ^ 2
def variance (weights means variances : Fin D → ℝ) : ℝ := max (rawVariance weights means variances) 0
def centered (weights means variances : Fin D → ℝ) : ℝ :=
  ∑ domain, weights domain * (variances domain + (means domain - mean weights means) ^ 2)

theorem centered_expansion_keeps_the_actual_total_weight (weights means variances : Fin D → ℝ) :
    centered weights means variances = second weights means variances -
      2 * mean weights means * mean weights means + mean weights means ^ 2 * mass weights := by
  unfold centered
  calc
    _ = ∑ domain, (weights domain * (variances domain + means domain ^ 2) -
        (2 * mean weights means) * (weights domain * means domain) +
        mean weights means ^ 2 * weights domain) := by
      apply Finset.sum_congr rfl
      intro domain _
      ring
    _ = _ := by
      simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, second, mean, mass, Finset.mul_sum]

theorem actual_variance_has_the_mass_error_term (weights means variances : Fin D → ℝ) :
    rawVariance weights means variances =
      centered weights means variances + (1 - mass weights) * mean weights means ^ 2 := by
  rw [centered_expansion_keeps_the_actual_total_weight]
  unfold rawVariance
  ring

theorem centered_second_moment_is_nonnegative (weights means variances : Fin D → ℝ)
    (nonnegativeWeights : ∀ domain, 0 ≤ weights domain)
    (nonnegativeVariances : ∀ domain, 0 ≤ variances domain) :
    0 ≤ centered weights means variances := by
  apply Finset.sum_nonneg
  intro domain _
  exact mul_nonneg (nonnegativeWeights domain) (add_nonneg (nonnegativeVariances domain) (sq_nonneg _))

theorem exact_unit_mass_is_a_conditional_special_case (weights means variances : Fin D → ℝ)
    (unit : mass weights = 1) : rawVariance weights means variances = centered weights means variances := by
  rw [actual_variance_has_the_mass_error_term, unit]
  ring

theorem near_unit_mass_gives_the_declared_negative_error_bound
    (weights means variances : Fin D → ℝ) (tolerance : ℝ)
    (nonnegativeWeights : ∀ domain, 0 ≤ weights domain)
    (nonnegativeVariances : ∀ domain, 0 ≤ variances domain)
    (near : |mass weights - 1| ≤ tolerance) :
    -tolerance * mean weights means ^ 2 ≤ rawVariance weights means variances := by
  have center := centered_second_moment_is_nonnegative weights means variances nonnegativeWeights nonnegativeVariances
  have upper := (abs_le.mp near).2
  have factor : -tolerance ≤ 1 - mass weights := by linarith
  have error := mul_le_mul_of_nonneg_right factor (sq_nonneg (mean weights means))
  rw [actual_variance_has_the_mass_error_term]
  linarith

theorem source_clamp_is_always_nonnegative (weights means variances : Fin D → ℝ) :
    0 ≤ variance weights means variances := le_max_right _ _

theorem exact_unit_mass_with_nonnegative_inputs_makes_clamp_an_identity
    (weights means variances : Fin D → ℝ) (unit : mass weights = 1)
    (nonnegativeWeights : ∀ domain, 0 ≤ weights domain)
    (nonnegativeVariances : ∀ domain, 0 ≤ variances domain) :
    variance weights means variances = rawVariance weights means variances := by
  apply max_eq_left
  rw [exact_unit_mass_is_a_conditional_special_case weights means variances unit]
  exact centered_second_moment_is_nonnegative weights means variances nonnegativeWeights nonnegativeVariances

structure SourceBuffer where
  mean : ℝ
  variance : ℝ
  batches : ℤ

inductive Failure where
  | missingWeights
  | weightTensorCast (message : String)
  | uninitializedActiveDomain
  deriving DecidableEq

def activeReady (weights : Fin D → ℝ) (buffers : Fin D → SourceBuffer) : Prop :=
  ∀ domain, 0 < weights domain → (buffers domain).batches ≠ 0

def weightedSource (weights : Option (Fin D → ℝ)) (buffers : Fin D → SourceBuffer) : Except Failure (ℝ × ℝ) :=
  match weights with
  | none => .error .missingWeights
  | some weights => if activeReady weights buffers then
      .ok (mean weights (fun domain => (buffers domain).mean),
        variance weights (fun domain => (buffers domain).mean) (fun domain => (buffers domain).variance))
    else .error .uninitializedActiveDomain

/-- One Tensor.new_tensor operation precedes the effective-weight graph. -/
def weightedSourceWithCast (cast : (Fin D → ℝ) → Except String (Fin D → ℝ))
    (configured : Option (Fin D → ℝ)) (buffers : Fin D → SourceBuffer) : Except Failure (ℝ × ℝ) := do
  match configured with
  | none => .error .missingWeights
  | some configured =>
    let effective ← (cast configured).mapError Failure.weightTensorCast
    weightedSource (some effective) buffers

theorem missing_weights_are_refused_before_domain_initialization (buffers : Fin D → SourceBuffer) :
    weightedSource none buffers = .error .missingWeights := rfl

theorem missing_configured_weights_do_not_invoke_the_tensor_cast
    (cast : (Fin D → ℝ) → Except String (Fin D → ℝ)) (buffers : Fin D → SourceBuffer) :
    weightedSourceWithCast cast none buffers = .error .missingWeights := rfl

theorem source_output_from_individual_active_domain_counts (weights : Fin D → ℝ) (buffers : Fin D → SourceBuffer)
    (counts : ∀ domain, 0 < weights domain → (buffers domain).batches ≠ 0) :
    weightedSource (some weights) buffers =
      .ok (mean weights (fun domain => (buffers domain).mean),
        variance weights (fun domain => (buffers domain).mean) (fun domain => (buffers domain).variance)) := by
  have ready : activeReady weights buffers := counts
  simp only [weightedSource, if_pos ready]

theorem output_from_one_tensor_cast_and_each_effective_active_count
    (cast : (Fin D → ℝ) → Except String (Fin D → ℝ))
    (configured effective : Fin D → ℝ) (buffers : Fin D → SourceBuffer)
    (castReply : cast configured = .ok effective)
    (counts : ∀ domain, 0 < effective domain → (buffers domain).batches ≠ 0) :
    weightedSourceWithCast cast (some configured) buffers =
      .ok (mean effective (fun domain => (buffers domain).mean),
        variance effective (fun domain => (buffers domain).mean) (fun domain => (buffers domain).variance)) := by
  simp only [weightedSourceWithCast, castReply, Except.mapError, Except.bind, Bind.bind]
  exact source_output_from_individual_active_domain_counts effective buffers counts

theorem tensor_cast_failure_precedes_every_active_domain_guard
    (cast : (Fin D → ℝ) → Except String (Fin D → ℝ)) (configured : Fin D → ℝ)
    (buffers : Fin D → SourceBuffer) (message : String) (failure : cast configured = .error message) :
    weightedSourceWithCast cast (some configured) buffers = .error (.weightTensorCast message) := by
  simp only [weightedSourceWithCast, failure, Except.mapError, Except.bind, Bind.bind]

theorem a_positive_weight_with_zero_batches_is_refused (weights : Fin D → ℝ) (buffers : Fin D → SourceBuffer)
    (domain : Fin D) (positive : 0 < weights domain) (empty : (buffers domain).batches = 0) :
    weightedSource (some weights) buffers = .error .uninitializedActiveDomain := by
  have unready : ¬activeReady weights buffers := by
    intro ready
    have count := ready domain positive
    rw [empty] at count
    exact count rfl
  simp [weightedSource, unready]

def equalWeights (_domain : Fin 2) : ℝ := 1 / 2
def separatedMeans (domain : Fin 2) : ℝ := if domain = 0 then -1 else 1
def zeroVariances (_domain : Fin 2) : ℝ := 0
def nearWeights (domain : Fin 2) : ℝ := if domain = 0 then 1 + 1 / 2147483648 else 0
def constantMeans (_domain : Fin 2) : ℝ := 1

theorem between_domain_means_contribute_the_missing_variance :
    mean equalWeights separatedMeans = 0 ∧ variance equalWeights separatedMeans zeroVariances = 1 ∧
      (∑ domain, equalWeights domain * zeroVariances domain) = 0 := by
  norm_num [mean, variance, rawVariance, second, equalWeights, separatedMeans, zeroVariances, Fin.sum_univ_succ]

theorem near_unit_effective_real_weights_need_not_have_exact_unit_mass :
    |mass nearWeights - 1| ≤ 1 / 1000000000 ∧ mass nearWeights ≠ 1 := by
  norm_num [mass, nearWeights, Fin.sum_univ_succ]

theorem near_unit_fixture_weights_are_nonnegative (domain : Fin 2) : 0 ≤ nearWeights domain := by
  by_cases first : domain = 0 <;> norm_num [nearWeights, first]

theorem near_unit_effective_mass_can_give_negative_unclipped_variance :
    rawVariance nearWeights constantMeans zeroVariances < 0 ∧ variance nearWeights constantMeans zeroVariances = 0 := by
  norm_num [rawVariance, second, mean, variance, nearWeights, constantMeans, zeroVariances, Fin.sum_univ_succ]

theorem source_does_not_silently_renormalize_effective_tensor_weights :
    mean nearWeights constantMeans = 1 + 1 / 2147483648 ∧ mean nearWeights constantMeans ≠ 1 := by
  norm_num [mean, nearWeights, constantMeans, Fin.sum_univ_succ]

def singleActiveWeight (domain : Fin 2) : ℝ := if domain = 0 then 1 else 0
def oneInitializedBuffer (domain : Fin 2) : SourceBuffer :=
  if domain = 0 then ⟨1, 0, 1⟩ else ⟨7, 4, 0⟩

theorem zero_weight_domain_does_not_require_source_initialization :
    weightedSource (some singleActiveWeight) oneInitializedBuffer = .ok (1, 0) := by
  have counts : ∀ domain, 0 < singleActiveWeight domain → (oneInitializedBuffer domain).batches ≠ 0 := by
    intro domain positive
    by_cases zero : domain = 0
    · simp [oneInitializedBuffer, zero]
    · simp [singleActiveWeight, zero] at positive
  rw [source_output_from_individual_active_domain_counts _ _ counts]
  norm_num [mean, variance, rawVariance, second, singleActiveWeight, oneInitializedBuffer, Fin.sum_univ_succ]

def negativeTrackedBuffer (domain : Fin 2) : SourceBuffer :=
  if domain = 0 then ⟨1, 0, -1⟩ else ⟨7, 4, 0⟩

theorem a_negative_nonzero_count_passes_this_literal_source_guard :
    weightedSource (some singleActiveWeight) negativeTrackedBuffer = .ok (1, 0) := by
  have counts : ∀ domain, 0 < singleActiveWeight domain → (negativeTrackedBuffer domain).batches ≠ 0 := by
    intro domain positive
    by_cases zero : domain = 0
    · simp [negativeTrackedBuffer, zero]
    · simp [singleActiveWeight, zero] at positive
  rw [source_output_from_individual_active_domain_counts _ _ counts]
  norm_num [mean, variance, rawVariance, second, singleActiveWeight, negativeTrackedBuffer, Fin.sum_univ_succ]

end
end N2P3.SplitBNStatistics
