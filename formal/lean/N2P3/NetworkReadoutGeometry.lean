import N2P3.ObservedTemporal
import N2P3.ERPFeatures
import N2P3.MatrixNormalCoordinates
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

/-!
# Source ST/MST geometry and the six actual readout branches

The signed output-size expressions follow n2p3net.py:300-390,689-819,
1182-1228. MST branches contain no temporal pool. The only Pm pool belongs to
MSFlattenPool. Core width planning is a constructor stage; latency reference/
candidate-window admission is a separate stage, not assumed to succeed here.

Complete-window reductions reuse ERPFeatures.completeBins, while network
flattening reuses MatrixNormalCoordinates with (row,column)=(feature,time).
ERPFeatures' later time-major flatten is deliberately not reused.
Source-first contract: v07_network_readout_contract_01.json.
-/
namespace N2P3.NetworkReadoutGeometry
open scoped BigOperators
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map

inductive Mode where
  | globalAverage | msFlatten | fullUnfold | mlpUnfold | quadraticUnfold | latencyContrast
  deriving DecidableEq

inductive Failure where
  | divisionByZero | shortWindow | missingDeclaredTime
  deriving DecidableEq

def convolutionTime (input : ℤ) (dilation kernel : ℕ) : ℤ :=
  (input + 2 * (ObservedTemporal.sourcePadding dilation kernel : ℕ) -
    (dilation * (kernel - 1) : ℕ) - 1) / 1 + 1

def poolTimeInteger (input : ℤ) (pool : ℕ) : ℤ :=
  (input - (pool : ℤ)) / pool + 1

def pooledTime (input pool : ℕ) : Except Failure ℕ :=
  if pool = 0 then .error .divisionByZero
  else
    let output := poolTimeInteger input pool
    if output < 1 then .error .shortWindow else .ok output.toNat

def twoPoolTimes (input stPool msPool : ℕ) : Except Failure ℕ := do
  let stTimes ← pooledTime input stPool
  pooledTime stTimes msPool

def requiresDeclaredTime : Mode → Bool
  | .msFlatten | .fullUnfold | .mlpUnfold | .quadraticUnfold => true
  | _ => false

def headWidth (mode : Mode) (channels times msPool : ℕ) : ℕ :=
  match mode with
  | .globalAverage | .latencyContrast => channels
  | .msFlatten => channels * (times / msPool)
  | _ => channels * times

/-- The literal constructor width stage, excluding separate clock/window and
native parameter-object admission. No complete constructor/forward success is
a premise of the statements below. -/
def geometryPlan (mode : Mode) (declared : Option ℕ) (channels stPool msPool : ℕ) :
    Except Failure (Option ℕ × ℕ) :=
  match declared with
  | none => if requiresDeclaredTime mode then .error .missingDeclaredTime else .ok (none, channels)
  | some input => do
    let stTimes ← pooledTime input stPool
    match mode with
    | .msFlatten => do
      let msTimes ← pooledTime stTimes msPool
      .ok (some stTimes, channels * msTimes)
    | _ => .ok (some stTimes, headWidth mode channels stTimes msPool)

theorem source_same_padding_conv_time_reuses_the_actual_odd_kernel_identity
    (input : ℤ) (dilation kernel : ℕ) (odd : kernel % 2 = 1) :
    convolutionTime input dilation kernel = input := by
  have padding : (2 : ℤ) * (ObservedTemporal.sourcePadding dilation kernel : ℕ) =
      (dilation * (kernel - 1) : ℕ) := by
    exact_mod_cast ObservedTemporal.odd_source_kernel_has_exact_symmetric_padding dilation kernel odd
  simp only [convolutionTime, Int.ediv_one]
  omega

theorem signed_pool_expression_is_exact_integer_floor_division
    (input : ℤ) (pool : ℕ) (positive : 0 < pool) :
    poolTimeInteger input pool = input / pool := by
  have nonzero : (pool : ℤ) ≠ 0 := by exact_mod_cast (ne_of_gt positive)
  unfold poolTimeInteger
  rw [Int.sub_ediv_of_dvd input (dvd_refl (pool : ℤ)), Int.ediv_self nonzero]
  omega

theorem signed_pool_expression_on_natural_time_is_the_natural_quotient
    (input pool : ℕ) (positive : 0 < pool) :
    poolTimeInteger input pool = (input / pool : ℕ) := by
  rw [signed_pool_expression_is_exact_integer_floor_division input pool positive]
  exact (Int.natCast_ediv input pool).symm

theorem actual_pool_guard_requires_at_least_one_complete_window
    (input pool : ℕ) (positive : 0 < pool) :
    pooledTime input pool =
      if pool ≤ input then .ok (input / pool) else .error .shortWindow := by
  rw [pooledTime, if_neg (ne_of_gt positive),
    signed_pool_expression_on_natural_time_is_the_natural_quotient input pool positive]
  by_cases complete : pool ≤ input
  · have quotientPositive : 0 < input / pool := Nat.div_pos complete positive
    have integerBound : (1 : ℤ) ≤ (input / pool : ℕ) := by exact_mod_cast (Nat.succ_le_of_lt quotientPositive)
    have integerPositive : ¬((input / pool : ℕ) : ℤ) < 1 := not_lt.mpr integerBound
    simp only [if_pos complete, if_neg integerPositive, Int.toNat_natCast]
  · have empty : input / pool = 0 := Nat.div_eq_of_lt (Nat.lt_of_not_ge complete)
    simp [complete, empty]

theorem two_actual_pool_guards_require_the_product_window
    (input stPool msPool : ℕ) (stPositive : 0 < stPool) (msPositive : 0 < msPool) :
    twoPoolTimes input stPool msPool =
      if stPool * msPool ≤ input then .ok (input / (stPool * msPool)) else .error .shortWindow := by
  have secondDomain : msPool ≤ input / stPool ↔ stPool * msPool ≤ input := by
    simpa only [Nat.mul_comm] using (Nat.le_div_iff_mul_le stPositive : msPool ≤ input / stPool ↔ msPool * stPool ≤ input)
  by_cases stComplete : stPool ≤ input
  · simp only [twoPoolTimes, actual_pool_guard_requires_at_least_one_complete_window input stPool stPositive,
      if_pos stComplete, Bind.bind, Except.bind]
    rw [actual_pool_guard_requires_at_least_one_complete_window _ msPool msPositive]
    simp only [secondDomain, Nat.div_div_eq_div_mul]
  · have productNotComplete : ¬stPool * msPool ≤ input := by
      have atLeastST : stPool ≤ stPool * msPool := Nat.le_mul_of_pos_right stPool msPositive
      omega
    simp [twoPoolTimes, actual_pool_guard_requires_at_least_one_complete_window,
      stPositive, stComplete, productNotComplete]

structure CoreShape (branches : ℕ) where
  batch : ℕ
  temporalTimes : ℤ
  spatialHeight : ℤ
  spatialChannels : ℕ
  stTimes : ℤ
  branchTimes : Fin branches → ℤ
  concatenatedChannels : ℕ
  residualTimes : ℤ

def literalCoreShape {M : ℕ} (batch input electrodes temporalFeatures depth stKernel stDilation stPool perScale : ℕ)
    (kernels dilations : Fin M → ℕ) (residualKernel : Option ℕ) : CoreShape M :=
  let temporal := convolutionTime input stDilation stKernel
  let spatialTime := convolutionTime temporal 1 1
  let pooled := poolTimeInteger spatialTime stPool
  { batch := batch
    temporalTimes := temporal
    spatialHeight := ((electrodes : ℤ) - electrodes) / 1 + 1
    spatialChannels := temporalFeatures * depth
    stTimes := pooled
    branchTimes := fun branch => convolutionTime pooled (dilations branch) (kernels branch)
    concatenatedChannels := M * perScale
    residualTimes := match residualKernel with
      | none => pooled
      | some kernel => convolutionTime pooled 1 kernel }

theorem complete_ST_MST_concat_residual_geometry_from_the_local_source_guards
    {M : ℕ} (batch input electrodes temporalFeatures depth stKernel stDilation stPool perScale : ℕ)
    (kernels dilations : Fin M → ℕ) (residualKernel : Option ℕ)
    (stOdd : stKernel % 2 = 1) (branchOdd : ∀ branch, kernels branch % 2 = 1)
    (residualOdd : ∀ kernel, residualKernel = some kernel → kernel % 2 = 1)
    (stPositive : 0 < stPool) (completeST : stPool ≤ input)
    (temporalPositive : 0 < temporalFeatures) (depthPositive : 0 < depth)
    (branchesPositive : 0 < M) (perScalePositive : 0 < perScale) :
    literalCoreShape batch input electrodes temporalFeatures depth stKernel stDilation stPool perScale
        kernels dilations residualKernel =
      ⟨batch, input, 1, temporalFeatures * depth, input / stPool,
        (fun _ => input / stPool), M * perScale, input / stPool⟩ ∧
      0 < temporalFeatures * depth ∧ 0 < M * perScale ∧ 0 < input / stPool := by
  have temporalTime := source_same_padding_conv_time_reuses_the_actual_odd_kernel_identity
    input stDilation stKernel stOdd
  have shapeIdentity : literalCoreShape batch input electrodes temporalFeatures depth stKernel stDilation stPool
      perScale kernels dilations residualKernel =
      ⟨batch, input, 1, temporalFeatures * depth, input / stPool,
        (fun _ => input / stPool), M * perScale, input / stPool⟩ := by
    cases residualKernel with
    | none =>
      simp [literalCoreShape, temporalTime,
        source_same_padding_conv_time_reuses_the_actual_odd_kernel_identity,
        signed_pool_expression_on_natural_time_is_the_natural_quotient, stPositive, branchOdd]
    | some kernel =>
      have odd := residualOdd kernel rfl
      simp [literalCoreShape, temporalTime,
        source_same_padding_conv_time_reuses_the_actual_odd_kernel_identity,
        signed_pool_expression_on_natural_time_is_the_natural_quotient, stPositive, branchOdd, odd]
  exact ⟨shapeIdentity, Nat.mul_pos temporalPositive depthPositive,
    Nat.mul_pos branchesPositive perScalePositive, Nat.div_pos completeST stPositive⟩

theorem all_six_readout_widths_use_the_actual_pooling_branch (channels times msPool : ℕ) :
    headWidth .globalAverage channels times msPool = channels ∧
      headWidth .latencyContrast channels times msPool = channels ∧
      headWidth .msFlatten channels times msPool = channels * (times / msPool) ∧
      headWidth .fullUnfold channels times msPool = channels * times ∧
      headWidth .mlpUnfold channels times msPool = channels * times ∧
      headWidth .quadraticUnfold channels times msPool = channels * times := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem the_four_position_preserving_modes_require_declared_time
    (mode : Mode) (positionPreserving : mode ∈ [.msFlatten, .fullUnfold, .mlpUnfold, .quadraticUnfold])
    (channels stPool msPool : ℕ) :
    geometryPlan mode none channels stPool msPool = .error .missingDeclaredTime := by
  cases mode <;> simp_all [geometryPlan, requiresDeclaredTime]

theorem dynamic_global_and_latency_widths_do_not_invent_a_time_dimension
    (channels stPool msPool : ℕ) :
    geometryPlan .globalAverage none channels stPool msPool = .ok (none, channels) ∧
      geometryPlan .latencyContrast none channels stPool msPool = .ok (none, channels) := by
  exact ⟨rfl, rfl⟩

theorem all_known_time_readout_plans_and_positive_head_widths_follow_from_local_windows
    (mode : Mode) (input channels stPool msPool : ℕ) (channelsPositive : 0 < channels)
    (stPositive : 0 < stPool) (msPositive : 0 < msPool) (completeST : stPool ≤ input)
    (completeMS : mode = .msFlatten → stPool * msPool ≤ input) :
    geometryPlan mode (some input) channels stPool msPool =
        .ok (some (input / stPool), headWidth mode channels (input / stPool) msPool) ∧
      0 < headWidth mode channels (input / stPool) msPool := by
  have stReply : pooledTime input stPool = .ok (input / stPool) := by
    rw [actual_pool_guard_requires_at_least_one_complete_window input stPool stPositive, if_pos completeST]
  have msWindow : mode = .msFlatten → msPool ≤ input / stPool := by
    intro isMS
    exact (Nat.le_div_iff_mul_le stPositive).mpr (by simpa only [Nat.mul_comm] using completeMS isMS)
  have msReply : mode = .msFlatten → pooledTime (input / stPool) msPool = .ok (input / stPool / msPool) := by
    intro isMS
    rw [actual_pool_guard_requires_at_least_one_complete_window _ msPool msPositive, if_pos (msWindow isMS)]
  have planReply : geometryPlan mode (some input) channels stPool msPool =
      .ok (some (input / stPool), headWidth mode channels (input / stPool) msPool) := by
    cases mode with
    | msFlatten =>
      simp only [geometryPlan, stReply, Bind.bind, Except.bind, msReply rfl, headWidth]
    | globalAverage | fullUnfold | mlpUnfold | quadraticUnfold | latencyContrast =>
      simp only [geometryPlan, stReply, Bind.bind, Except.bind]
  have widthPositive : 0 < headWidth mode channels (input / stPool) msPool := by
    cases mode with
    | globalAverage | latencyContrast => exact channelsPositive
    | msFlatten => exact Nat.mul_pos channelsPositive (Nat.div_pos (msWindow rfl) msPositive)
    | fullUnfold | mlpUnfold | quadraticUnfold =>
      exact Nat.mul_pos channelsPositive (Nat.div_pos completeST stPositive)
  exact ⟨planReply, widthPositive⟩

def fullUnfold {C T : ℕ} (features : Matrix (Fin C) (Fin T) ℝ) : Fin (C * T) → ℝ :=
  MatrixNormalCoordinates.rowMajorVector features

def concat {M F T : ℕ} (branches : Fin M → Matrix (Fin F) (Fin T) ℝ) : Matrix (Fin (M * F)) (Fin T) ℝ :=
  fun channel time => branches (finProdFinEquiv.symm channel).1 (finProdFinEquiv.symm channel).2 time

def msFlatten {C T : ℕ} (pool : ℕ) (features : Matrix (Fin C) (Fin T) ℝ) : Fin (C * (T / pool)) → ℝ :=
  fullUnfold (ERPFeatures.completeBins pool features)

theorem actual_concat_and_unfold_keep_branch_feature_time_order
    {M F T : ℕ} (branches : Fin M → Matrix (Fin F) (Fin T) ℝ)
    (branch : Fin M) (feature : Fin F) (time : Fin T) :
    (finProdFinEquiv (finProdFinEquiv (branch, feature), time)).val =
        T * (F * branch.val + feature.val) + time.val ∧
      fullUnfold (concat branches) (finProdFinEquiv (finProdFinEquiv (branch, feature), time)) =
        branches branch feature time := by
  constructor
  · rw [MatrixNormalCoordinates.rowMajorIndex_value,
      MatrixNormalCoordinates.rowMajorIndex_value]
  · rw [fullUnfold, MatrixNormalCoordinates.rowMajorVector_at]
    simp [concat]

theorem direct_unfold_is_injective_on_encoded_coordinates {C T : ℕ} :
    Function.Injective (fullUnfold : Matrix (Fin C) (Fin T) ℝ → Fin (C * T) → ℝ) := by
  intro first second same
  funext channel time
  have cell := congrFun same (finProdFinEquiv (channel, time))
  simpa only [fullUnfold, MatrixNormalCoordinates.rowMajorVector_at] using cell

theorem actual_MS_readout_uses_complete_window_means_then_channel_major_order
    {C T : ℕ} (pool : ℕ) (features : Matrix (Fin C) (Fin T) ℝ)
    (channel : Fin C) (bin : Fin (T / pool)) :
    msFlatten pool features (finProdFinEquiv (channel, bin)) =
      (∑ offset, features channel (ERPFeatures.binCoordinate pool bin offset)) / pool := by
  rw [msFlatten, fullUnfold, MatrixNormalCoordinates.rowMajorVector_at,
    ERPFeatures.complete_bin_formula]

def orderedFeatures : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 3, 4]

theorem nonempty_feature_order_is_channel_first_and_time_fast :
    fullUnfold orderedFeatures 1 = 2 ∧ fullUnfold orderedFeatures 2 = 3 := by
  constructor
  · change fullUnfold orderedFeatures (finProdFinEquiv ((0 : Fin 2), (1 : Fin 2))) = 2
    rw [fullUnfold, MatrixNormalCoordinates.rowMajorVector_at]
    rfl
  · change fullUnfold orderedFeatures (finProdFinEquiv ((1 : Fin 2), (0 : Fin 2))) = 3
    rw [fullUnfold, MatrixNormalCoordinates.rowMajorVector_at]
    rfl

def leftImpulse : Matrix (Fin 1) (Fin 4) ℝ := fun _ time => if time = 0 then 1 else 0
def rightImpulse : Matrix (Fin 1) (Fin 4) ℝ := fun _ time => if time = 1 then 1 else 0

theorem complete_average_pool_collides_while_direct_unfold_retains_the_difference :
    msFlatten 2 leftImpulse = msFlatten 2 rightImpulse ∧ fullUnfold leftImpulse ≠ fullUnfold rightImpulse := by
  have bins : ERPFeatures.completeBins 2 leftImpulse = ERPFeatures.completeBins 2 rightImpulse := by
    funext channel bin
    fin_cases channel
    fin_cases bin <;>
      norm_num [ERPFeatures.completeBins, ERPFeatures.loopMean, ERPFeatures.sumLoop,
        ERPFeatures.reshapeBins, ERPFeatures.truncated, ERPFeatures.binCoordinate,
        leftImpulse, rightImpulse, finProdFinEquiv]
  have different : leftImpulse ≠ rightImpulse := by
    intro same
    have cell := congrFun (congrFun same 0) 0
    norm_num [leftImpulse, rightImpulse] at cell
  exact ⟨congrArg fullUnfold bins, fun same => different (direct_unfold_is_injective_on_encoded_coordinates same)⟩

theorem fixed_source_default_readout_widths_separate_full_MS_and_global :
    headWidth .fullUnfold (2 * 2) (128 / 4) 8 = 128 ∧
      headWidth .msFlatten (2 * 2) (128 / 4) 8 = 16 ∧
      headWidth .globalAverage (2 * 2) (128 / 4) 8 = 4 ∧
      headWidth .latencyContrast (2 * 2) (128 / 4) 8 = 4 := by norm_num [headWidth]

theorem incomplete_source_windows_round_down_and_MS_rejects_the_short_case :
    pooledTime 113 4 = .ok 28 ∧ pooledTime 31 4 = .ok 7 ∧
      twoPoolTimes 31 4 8 = .error .shortWindow := by
  rw [actual_pool_guard_requires_at_least_one_complete_window 113 4 (by decide),
    actual_pool_guard_requires_at_least_one_complete_window 31 4 (by decide),
    two_actual_pool_guards_require_the_product_window 31 4 8 (by decide) (by decide)]
  norm_num

end
end N2P3.NetworkReadoutGeometry
