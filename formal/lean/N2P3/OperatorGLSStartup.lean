import N2P3.TemporalWhitening
import N2P3.EventDesign
import N2P3.ScaleGauge

/-!
# Actual OperatorGLS startup rows, original nuisance grid and prior-scale rule

Contract: operator_gls_caller_source_contract_01.json, fixed before this file.
The active implementation is doc/assets/operator_research_20261004/continuous_model.py.
All three branches call the already literal TemporalWhitening loops. The
conditional branch drops likelihood rows, retains their observed AR history,
and whitens the nuisance grid generated at the original record length.
Marginal prior scale changes lambda, not the unit-innovation whitening loop.
These are fixed decoded mathematical fields; no native estimator/IEEE or
physical completeness/independence conclusion is supplied.
-/
namespace N2P3.OperatorGLSStartup
open Matrix
open N2P3.TemporalWhitening
open scoped BigOperators Matrix Kronecker
noncomputable section

structure Noise where
  mode : Mode
  coefficients : List ℝ
  reflections : List ℝ
  root : ℕ → ℕ → ℝ
  startupCov00 : ℝ
  diagnosticVariance : ℝ

def rows (noise : Noise) (n : ℕ) : ℕ := outputRows noise.mode noise.coefficients.length n

def white {C : Type*} (noise : Noise) (n : ℕ) (x : Signal C) : Matrix (Fin (rows noise n)) C ℝ :=
  fun i c => rowAt (whiten noise.mode noise.coefficients noise.reflections noise.root
    noise.diagnosticVariance n x) i.val c

def finiteSignal {n : ℕ} {C : Type*} (x : Matrix (Fin n) C ℝ) : Signal C :=
  fun time c => if h : time < n then x ⟨time,h⟩ c else 0

def processedSamples {n : ℕ} {C : Type*} (hasFilter : Bool) (F : Matrix (Fin n) (Fin n) ℝ)
    (x : Matrix (Fin n) C ℝ) : Matrix (Fin n) C ℝ := if hasFilter then F * x else x

def processedDesign {n : ℕ} {C : Type*} (hasFilter filterDesign : Bool) (F : Matrix (Fin n) (Fin n) ℝ)
    (D : Matrix (Fin n) C ℝ) : Matrix (Fin n) C ℝ := if hasFilter && filterDesign then F * D else D

theorem unfiltered_input_preserves_both_raw_arrays {n : ℕ} {p c : Type*}
    (F : Matrix (Fin n) (Fin n) ℝ) (X : Matrix (Fin n) c ℝ) (D : Matrix (Fin n) p ℝ)
    (filterDesign : Bool) :
    processedSamples false F X = X ∧ processedDesign false filterDesign F D = D := by
  simp [processedSamples, processedDesign]

theorem literal_whitening_has_exact_output_rows (noise : Noise) (n columns : ℕ)
    (x : Signal (Fin columns)) (legal : InputAdmissible noise.mode noise.coefficients.length n columns)
    (lengths : noise.reflections.length = noise.coefficients.length) :
    (whiten noise.mode noise.coefficients noise.reflections noise.root noise.diagnosticVariance n x).length = rows noise n :=
  whiten_output_length noise.mode noise.coefficients noise.reflections noise.root noise.diagnosticVariance n columns x legal lengths

theorem stationary_keeps_original_rows (noise : Noise) (n : ℕ) (mode : noise.mode = .stationary) : rows noise n = n := by
  simp [rows, outputRows, mode]

theorem stationary_innovations_keeps_original_rows (noise : Noise) (n : ℕ)
    (mode : noise.mode = .stationaryInnovations) : rows noise n = n := by
  simp [rows, outputRows, mode]

theorem conditional_prefix_keeps_only_innovation_rows (noise : Noise) (n : ℕ)
    (mode : noise.mode = .conditionalPrefix) : rows noise n = n - noise.coefficients.length := by
  simp [rows, outputRows, mode]

theorem retained_observation_time_is_inside_original_record (noise : Noise) (n : ℕ)
    (i : Fin (rows noise n)) : observationTime noise.mode noise.coefficients.length i.val < n := by
  have bound := i.isLt
  cases mode : noise.mode <;> simp [rows, outputRows, observationTime, mode] at bound ⊢ <;> omega

theorem actual_whitening_reads_only_available_within_record_history {C : Type*}
    (noise : Noise) (n : ℕ) (i : Fin (rows noise n)) (x y : Signal C) (c : C)
    (same : ∀ t ≤ observationTime noise.mode noise.coefficients.length i.val, x t c = y t c) :
    white noise n x i c = white noise n y i c :=
  whiten_within_record_causal noise.mode noise.coefficients noise.reflections noise.root noise.diagnosticVariance n i.val x y c same

theorem finite_record_values_determine_every_actual_whitened_row {C : Type*}
    (noise : Noise) (n : ℕ) (x y : Signal C)
    (same : ∀ time < n, ∀ c, x time c = y time c) (i : Fin (rows noise n)) (c : C) :
    white noise n x i c = white noise n y i c := by
  apply actual_whitening_reads_only_available_within_record_history
  intro time available
  exact same time (lt_of_le_of_lt available (retained_observation_time_is_inside_original_record noise n i)) c

theorem conditional_coordinate_uses_observed_prefix_history {C : Type*}
    (noise : Noise) (n : ℕ) (x : Signal C) (c : C) (i : Fin (rows noise n))
    (mode : noise.mode = .conditionalPrefix) :
    white noise n x i c = x (noise.coefficients.length + i.val) c -
      (lagTerms noise.coefficients 1 noise.coefficients.length x i.val c).sum := by
  have bound : i.val < n - noise.coefficients.length := by simpa [rows, outputRows, mode] using i.isLt
  simpa only [white, whiten, mode] using conditional_coordinate noise.coefficients n i.val x c bound

theorem stationary_first_row_uses_actual_triangular_startup {C : Type*}
    (noise : Noise) (n : ℕ) (x : Signal C) (c : C) (mode : noise.mode = .stationary)
    (order : 0 < noise.coefficients.length) (nonempty : 0 < n) :
    white noise n x ⟨0, by simpa [rows, outputRows, mode] using nonempty⟩ c = x 0 c / noise.root 0 0 := by
  simp only [white, whiten, mode]
  rw [stationary_prefix_coordinate _ _ _ _ _ _ order, prefixSolve_zero]

theorem lattice_first_row_uses_actual_suffix_variance {C : Type*}
    (noise : Noise) (n : ℕ) (x : Signal C) (c : C) (mode : noise.mode = .stationaryInnovations)
    (reflections : noise.reflections ≠ []) (nonempty : 0 < n) :
    white noise n x ⟨0, by simpa [rows, outputRows, mode] using nonempty⟩ c =
      x 0 c * prefixScale noise.reflections 0 := by
  cases ks : noise.reflections with
  | nil => exact False.elim (reflections ks)
  | cons k tail => simp [white, whiten, mode, ks, lattice, rowAt]

/-- np.linspace(-1,1,len(values)) is generated before noise.whiten. -/
def originalNuisance (n : ℕ) : Signal (Fin 2) :=
  fun t column => if column = 0 then 1 else -1 + 2 * (t : ℝ) / (n - 1 : ℕ)

theorem conditional_nuisance_uses_original_grid (noise : Noise) (n : ℕ)
    (i : Fin (rows noise n)) (column : Fin 2) (mode : noise.mode = .conditionalPrefix) :
    white noise n (originalNuisance n) i column =
      originalNuisance n (noise.coefficients.length + i.val) column -
        (lagTerms noise.coefficients 1 noise.coefficients.length (originalNuisance n) i.val column).sum :=
  conditional_coordinate_uses_observed_prefix_history noise n (originalNuisance n) column i mode

theorem modeled_event_design_is_zero_before_conditioning_cutoff
    {width rank n : ℕ} {E : Type*} (basis : EventDesign.Basis width rank)
    (onsets : E → ℕ) (labels : E → ℝ) (events : List E) (cutoff : ℕ)
    (available : ∀ event ∈ events, cutoff ≤ onsets event)
    (time : Fin n) (earlier : time.val < cutoff) (column : EventDesign.Columns rank) :
    EventDesign.design basis onsets labels events time column = 0 := by
  induction events with
  | nil => simp [EventDesign.design_nil]
  | cons event tail ih =>
      have before : ¬onsets event ≤ time.val := by have := available event (by simp); omega
      have later : ∀ e ∈ tail, cutoff ≤ onsets e := fun e member => available e (by simp [member])
      rw [EventDesign.design_cons]
      cases column <;> simp [EventDesign.eventTerm, EventDesign.responseAt, before, ih later]

/-- _validate_record establishes strict onset order; the OperatorGLS caller
checks only the first onset against p. Every modeled prefix row is then zero
without assuming that whole design property as an input. -/
theorem actual_first_onset_guard_derives_candidate_free_modeled_prefix
    {width rank n : ℕ} {E : Type*} (basis : EventDesign.Basis width rank)
    (onsets : E → ℕ) (labels : E → ℝ) (first : E) (tail : List E) (cutoff : ℕ)
    (ordered : (first :: tail).Pairwise (fun left right => onsets left < onsets right))
    (firstAllowed : cutoff ≤ onsets first)
    (time : Fin n) (earlier : time.val < cutoff) (column : EventDesign.Columns rank) :
    EventDesign.design basis onsets labels (first :: tail) time column = 0 := by
  apply modeled_event_design_is_zero_before_conditioning_cutoff basis onsets labels (first :: tail) cutoff
    _ time earlier column
  intro event member
  rcases List.mem_cons.mp member with same | later
  · simpa [same] using firstAllowed
  · exact firstAllowed.trans (le_of_lt ((List.pairwise_cons.mp ordered).1 event later))

def conditionalHistoryAllowed (noise : Noise) (hasProjectFilter : Bool) (firstOnset : ℕ) : Prop :=
  noise.mode ≠ .conditionalPrefix ∨ (hasProjectFilter = false ∧ noise.coefficients.length ≤ firstOnset)

theorem conditional_history_guard_retains_both_actual_restrictions (noise : Noise)
    (hasProjectFilter : Bool) (firstOnset : ℕ) (mode : noise.mode = .conditionalPrefix) :
    conditionalHistoryAllowed noise hasProjectFilter firstOnset ↔
      hasProjectFilter = false ∧ noise.coefficients.length ≤ firstOnset := by
  simp [conditionalHistoryAllowed, mode]

def marginalReference (noise : Noise) : ℝ :=
  if noise.coefficients = [] then 1 else match noise.mode with
    | .stationary => noise.startupCov00
    | .conditionalPrefix | .stationaryInnovations => Real.exp (logVariance noise.reflections)

theorem stationary_startup_variance_positive_from_actual_first_factor (noise : Noise)
    (equation : noise.startupCov00 = (noise.root 0 0)^2) (diagonal : noise.root 0 0 ≠ 0) :
    0 < noise.startupCov00 := by rw [equation]; exact sq_pos_of_ne_zero diagonal

theorem actual_marginal_reference_is_positive (noise : Noise)
    (stationary : noise.mode = .stationary → noise.coefficients ≠ [] → 0 < noise.startupCov00) :
    0 < marginalReference noise := by
  by_cases zeroOrder : noise.coefficients = []
  · simp [marginalReference, zeroOrder]
  · cases mode : noise.mode with
    | stationary => simpa [marginalReference, zeroOrder, mode] using stationary mode zeroOrder
    | conditionalPrefix => simp [marginalReference, zeroOrder, mode, Real.exp_pos]
    | stationaryInnovations => simp [marginalReference, zeroOrder, mode, Real.exp_pos]

theorem nonstationary_reference_is_actual_reflection_product (noise : Noise)
    (nonzero : noise.coefficients ≠ []) (mode : noise.mode ≠ .stationary)
    (stable : ∀ k ∈ noise.reflections, |k| < 1) :
    marginalReference noise = varianceProduct noise.reflections := by
  cases hm : noise.mode <;> simp_all [marginalReference,
    innovation_variance_is_actual_parcor_product noise.reflections stable]

inductive PriorScale where | innovation | marginal deriving DecidableEq

def effectiveLambda (scale : PriorScale) (noise : Noise) (ridge : ℝ) : ℝ :=
  match scale with | .innovation => ridge | .marginal => ridge / marginalReference noise

theorem all_startup_and_scale_branches_have_positive_actual_lambda
    (scale : PriorScale) (noise : Noise) (ridge : ℝ) (positive : 0 < ridge)
    (stationary : noise.mode = .stationary → noise.coefficients ≠ [] → 0 < noise.startupCov00) :
    0 < effectiveLambda scale noise ridge := by
  cases scale
  · exact positive
  · exact div_pos positive (actual_marginal_reference_is_positive noise stationary)

theorem marginal_lambda_recovers_the_declared_ridge (noise : Noise) (ridge : ℝ)
    (positive : 0 < marginalReference noise) :
    effectiveLambda .marginal noise ridge * marginalReference noise = ridge := by
  exact div_mul_cancel₀ ridge (ne_of_gt positive)

theorem actual_marginal_prior_inverse_scale (noise : Noise) (ridge : ℝ) :
    (effectiveLambda .marginal noise ridge)⁻¹ = marginalReference noise * ridge⁻¹ := by
  simp [effectiveLambda, div_eq_mul_inv]

theorem only_coordinated_gauge_preserves_physical_prior
    {p c : Type*} [Fintype p] [Fintype c] [DecidableEq p]
    (Sigma : Matrix c c ℝ) (noise : Noise) (ridge : ℝ) (positive : 0 < marginalReference noise) :
    ScaleGauge.predictiveCov ((effectiveLambda .marginal noise ridge)⁻¹ • (1 : Matrix p p ℝ))
      ((marginalReference noise)⁻¹ • Sigma) =
      ScaleGauge.predictiveCov (ridge⁻¹ • (1 : Matrix p p ℝ)) Sigma := by
  exact ScaleGauge.physicalPrior_gauge Sigma ridge (marginalReference noise) (ne_of_gt positive)

theorem diagnostic_variance_does_not_divide_whitened_values_again {C : Type*}
    (noise : Noise) (first second : ℝ) (n : ℕ) (x : Signal C) :
    white {noise with diagnosticVariance := first} n x = white {noise with diagnosticVariance := second} n x := rfl

def conditionalFixture : Noise := ⟨.conditionalPrefix, [1/2], [-1/2], fun _ _ => 1, 4/3, 1⟩
def historySignal : Signal (Fin 1) := fun t _ => if t = 0 then 2 else 3
def fixtureRow : Fin (rows conditionalFixture 4) := ⟨0, by decide⟩

theorem conditional_fixture_retains_real_history_and_shortens_rows :
    rows conditionalFixture 4 = 3 ∧ white conditionalFixture 4 historySignal fixtureRow 0 = 2 := by
  constructor
  · rfl
  · rw [conditional_coordinate_uses_observed_prefix_history conditionalFixture 4 historySignal 0 fixtureRow rfl]
    norm_num [conditionalFixture, historySignal, lagTerms, fixtureRow]

theorem original_nuisance_grid_differs_from_rebuilt_conditional_grid :
    originalNuisance 4 1 1 = (-1/3 : ℝ) ∧ originalNuisance 3 0 1 = -1 := by
  norm_num [originalNuisance]

def priorFixture : Noise := ⟨.stationary, [3/5], [-3/5], fun _ _ => 5/4, 25/16, 1⟩

theorem changing_prior_convention_changes_lambda_with_same_noise :
    effectiveLambda .innovation priorFixture (25/8) = 25/8 ∧
    effectiveLambda .marginal priorFixture (25/8) = 2 := by
  norm_num [effectiveLambda, marginalReference, priorFixture]

end
end N2P3.OperatorGLSStartup
