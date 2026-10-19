import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# Finite candidate normalization from the actual scoring interface

`ContinuousDecisionModel.score` computes `exp(evidence - logsumexp(evidence))`.
The BNCI caller reports `logsumexp(evidence) - evidence[truth]`. These definitions
use finite exact real scores, not NumPy/Scipy floating-point execution. In
particular they do not discharge overflow, NaN, membership, or timing gates.

The common-density-constant result is an algebraic consumer of a separately
proved log-density identity. It does not assume that a normalized likelihood is
a posterior for an arbitrary candidate prior. A candidate prior is explicit
below; the source probability is recovered for the uniform prior only.
-/

namespace N2P3.CandidateProbability

open scoped BigOperators

noncomputable section

variable {κ : Type*} [Fintype κ]

def partition (e : κ → ℝ) : ℝ := ∑ j, Real.exp (e j)

def logPartition (e : κ → ℝ) : ℝ := Real.log (partition e)

def probability (e : κ → ℝ) (i : κ) : ℝ :=
  Real.exp (e i - logPartition e)

def nll (e : κ → ℝ) (truth : κ) : ℝ := logPartition e - e truth

def normalizedDensity (density : κ → ℝ) (i : κ) : ℝ :=
  density i / ∑ j, density j

/-- Candidate-level priors are distinct from the fitted coefficient prior. -/
def IsPrior (prior : κ → ℝ) : Prop := (∀ i, 0 ≤ prior i) ∧ ∑ i, prior i = 1

def weightedPartition (prior e : κ → ℝ) : ℝ := ∑ j, prior j * Real.exp (e j)

def posterior (prior e : κ → ℝ) (i : κ) : ℝ :=
  prior i * Real.exp (e i) / weightedPartition prior e

def uniformPrior (_ : κ) : ℝ := 1 / (Fintype.card κ : ℝ)

theorem partition_positive [Nonempty κ] (e : κ → ℝ) : 0 < partition e := by
  exact Finset.sum_pos (fun i _ => Real.exp_pos (e i)) Finset.univ_nonempty

/-- The source exp/logsumexp expression equals the normalized positive weights. -/
theorem probability_eq_exp_div_partition [Nonempty κ] (e : κ → ℝ) (i : κ) :
    probability e i = Real.exp (e i) / partition e := by
  rw [probability, logPartition, Real.exp_sub, Real.exp_log (partition_positive e)]

theorem probability_positive (e : κ → ℝ) (i : κ) : 0 < probability e i :=
  Real.exp_pos _

theorem probability_sum_one [Nonempty κ] (e : κ → ℝ) :
    ∑ i, probability e i = 1 := by
  simp_rw [probability_eq_exp_div_partition]
  simp only [div_eq_mul_inv]
  rw [← Finset.sum_mul, ← partition, mul_inv_cancel₀ (partition_positive e).ne']

theorem probability_le_one [Nonempty κ] (e : κ → ℝ) (i : κ) :
    probability e i ≤ 1 := by
  classical
  rw [← probability_sum_one e]
  exact Finset.single_le_sum (fun j _ => (probability_positive e j).le) (Finset.mem_univ i)

theorem nll_eq_neg_log_probability (e : κ → ℝ) (truth : κ) :
    nll e truth = -Real.log (probability e truth) := by
  rw [probability, Real.log_exp, nll]
  ring

theorem nll_nonnegative [Nonempty κ] (e : κ → ℝ) (truth : κ) :
    0 ≤ nll e truth := by
  rw [nll_eq_neg_log_probability]
  exact neg_nonneg.mpr (Real.log_nonpos (probability_positive e truth).le
    (probability_le_one e truth))

theorem partition_common_constant (e : κ → ℝ) (a : ℝ) :
    partition (fun i => e i + a) = Real.exp a * partition e := by
  simp only [partition, Real.exp_add, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem logPartition_common_constant [Nonempty κ] (e : κ → ℝ) (a : ℝ) :
    logPartition (fun i => e i + a) = logPartition e + a := by
  rw [logPartition, partition_common_constant,
    Real.log_mul (Real.exp_ne_zero a) (partition_positive e).ne', Real.log_exp,
    logPartition]
  ring

theorem probability_common_constant [Nonempty κ] (e : κ → ℝ) (a : ℝ) (i : κ) :
    probability (fun j => e j + a) i = probability e i := by
  simp only [probability, logPartition_common_constant]
  congr 1
  ring

theorem nll_common_constant [Nonempty κ] (e : κ → ℝ) (a : ℝ) (truth : κ) :
    nll (fun i => e i + a) truth = nll e truth := by
  simp only [nll, logPartition_common_constant]
  ring

omit [Fintype κ] in
/-- A true positive density recovers its value from its independently proved log. -/
theorem density_eq_exp_times_score (e density : κ → ℝ) (a : ℝ)
    (hpositive : ∀ i, 0 < density i)
    (hlog : ∀ i, Real.log (density i) = e i + a) (i : κ) :
    density i = Real.exp a * Real.exp (e i) := by
  rw [← Real.exp_log (hpositive i), hlog i, Real.exp_add, mul_comm]

theorem density_sum_eq_exp_times_partition (e density : κ → ℝ) (a : ℝ)
    (hpositive : ∀ i, 0 < density i)
    (hlog : ∀ i, Real.log (density i) = e i + a) :
    ∑ i, density i = Real.exp a * partition e := by
  simp_rw [density_eq_exp_times_score e density a hpositive hlog]
  exact (Finset.mul_sum _ _ _).symm

/-- Normalization is over candidates at one shared observation and base measure.
The needed positive-density/log-score identity is supplied by the Gaussian law
proof, not by an assumption about these normalized output probabilities. -/
theorem probability_eq_normalized_density [Nonempty κ] (e density : κ → ℝ) (a : ℝ)
    (hpositive : ∀ i, 0 < density i)
    (hlog : ∀ i, Real.log (density i) = e i + a) (i : κ) :
    probability e i = normalizedDensity density i := by
  rw [normalizedDensity, density_sum_eq_exp_times_partition e density a hpositive hlog,
    density_eq_exp_times_score e density a hpositive hlog i,
    mul_div_mul_left _ _ (Real.exp_ne_zero a)]
  exact probability_eq_exp_div_partition e i

theorem weightedPartition_positive (prior e : κ → ℝ) (hprior : IsPrior prior) :
    0 < weightedPartition prior e := by
  obtain ⟨i, _, hi⟩ := (Finset.sum_pos_iff_of_nonneg (fun j _ => hprior.1 j)).mp
    (show (0 : ℝ) < ∑ j, prior j by rw [hprior.2]; norm_num)
  exact Finset.sum_pos' (fun j _ => mul_nonneg (hprior.1 j) (Real.exp_pos (e j)).le)
    ⟨i, Finset.mem_univ i, mul_pos hi (Real.exp_pos (e i))⟩

theorem posterior_nonnegative (prior e : κ → ℝ) (hprior : IsPrior prior) (i : κ) :
    0 ≤ posterior prior e i := by
  exact div_nonneg (mul_nonneg (hprior.1 i) (Real.exp_pos (e i)).le)
    (weightedPartition_positive prior e hprior).le

theorem posterior_sum_one (prior e : κ → ℝ) (hprior : IsPrior prior) :
    ∑ i, posterior prior e i = 1 := by
  simp only [posterior]
  simp only [div_eq_mul_inv]
  rw [← Finset.sum_mul, ← weightedPartition,
    mul_inv_cancel₀ (weightedPartition_positive prior e hprior).ne']

/-- A zero prior is legal and gives zero posterior mass, unlike source softmax. -/
theorem posterior_zero_of_prior_zero (prior e : κ → ℝ) (i : κ) (hi : prior i = 0) :
    posterior prior e i = 0 := by
  simp [posterior, hi]

theorem uniformPrior_isPrior [Nonempty κ] : IsPrior (uniformPrior : κ → ℝ) := by
  have hcard : (Fintype.card κ : ℝ) ≠ 0 :=
    Nat.cast_ne_zero.mpr (Fintype.card_ne_zero (α := κ))
  constructor
  · intro i
    exact one_div_nonneg.mpr (Nat.cast_nonneg _)
  · simp [uniformPrior, hcard]

/-- The actual source omits candidate prior factors; its posterior interpretation
therefore uses the uniform candidate prior, not an arbitrary one. -/
theorem uniform_posterior_eq_probability [Nonempty κ] (e : κ → ℝ) (i : κ) :
    posterior uniformPrior e i = probability e i := by
  have hcard : (Fintype.card κ : ℝ) ≠ 0 :=
    Nat.cast_ne_zero.mpr (Fintype.card_ne_zero (α := κ))
  have hweight : (1 / (Fintype.card κ : ℝ)) ≠ 0 := one_div_ne_zero hcard
  have hp : weightedPartition uniformPrior e =
      (1 / (Fintype.card κ : ℝ)) * partition e := by
    simp only [weightedPartition, uniformPrior, partition, Finset.mul_sum]
  rw [posterior, uniformPrior, hp, mul_div_mul_left _ _ hweight,
    probability_eq_exp_div_partition]

theorem posterior_common_constant (prior e : κ → ℝ) (a : ℝ) (i : κ) :
    posterior prior (fun j => e j + a) i = posterior prior e i := by
  have hp : weightedPartition prior (fun j => e j + a) =
      Real.exp a * weightedPartition prior e := by
    simp only [weightedPartition, Real.exp_add, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    ring
  simp only [posterior, hp, Real.exp_add]
  rw [show prior i * (Real.exp (e i) * Real.exp a) =
      Real.exp a * (prior i * Real.exp (e i)) by ring,
    mul_div_mul_left _ _ (Real.exp_ne_zero a)]

/-- Discrete Bayes weighting of independently derived observation densities. -/
theorem posterior_eq_weighted_density (prior e density : κ → ℝ) (a : ℝ)
    (hpositive : ∀ i, 0 < density i)
    (hlog : ∀ i, Real.log (density i) = e i + a) (i : κ) :
    posterior prior e i = prior i * density i / ∑ j, prior j * density j := by
  have hsum : (∑ j, prior j * density j) = Real.exp a * weightedPartition prior e := by
    simp_rw [density_eq_exp_times_score e density a hpositive hlog]
    simp only [weightedPartition, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    ring
  rw [hsum, density_eq_exp_times_score e density a hpositive hlog i,
    show prior i * (Real.exp a * Real.exp (e i)) =
      Real.exp a * (prior i * Real.exp (e i)) by ring,
    mul_div_mul_left _ _ (Real.exp_ne_zero a)]
  rfl

/-- Equal observation scores preserve an explicit proper candidate prior. -/
theorem constant_scores_posterior_eq_prior (prior : κ → ℝ) (hprior : IsPrior prior)
    (a : ℝ) (i : κ) : posterior prior (fun _ => a) i = prior i := by
  have hp : weightedPartition prior (fun _ => a) = Real.exp a := by
    rw [weightedPartition, ← Finset.sum_mul, hprior.2, one_mul]
  rw [posterior, hp, mul_div_cancel_right₀ _ (Real.exp_ne_zero a)]

end

end N2P3.CandidateProbability
