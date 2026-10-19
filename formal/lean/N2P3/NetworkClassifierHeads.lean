import N2P3.NetworkReadoutGeometry
import Mathlib.Tactic.Ring

/-!
# Actual affine, factorized-quadratic and residual-MLP readouts

Source: n2p3net.py:392-437,789-800,1230-1256,1277-1278. Linear maps keep
the exact source biases. GELU is one per-coordinate real operation interface;
zero residual-output identity holds for every such operation, without a whole
head/forward-success premise or any assumption on its desired output.

Parameter cells are disjointly tagged by the actual parameter-array fields.
Their cardinality is allocated storage, while the source parameter_count()
filters requires_grad. Zero initialization does not remove parameter cells.
Source-first contract: v07_network_readout_contract_01.json, H01-H04.
-/
namespace N2P3.NetworkClassifierHeads
open scoped BigOperators
noncomputable section
attribute [local instance] Classical.propDecidable

def affine {P O : ℕ} (weight : Matrix (Fin O) (Fin P) ℝ) (bias : Fin O → ℝ)
    (input : Fin P → ℝ) (output : Fin O) : ℝ :=
  (∑ feature, weight output feature * input feature) + bias output

def projection {P R : ℕ} (weight : Matrix (Fin R) (Fin P) ℝ)
    (input : Fin P → ℝ) (rank : Fin R) : ℝ := ∑ feature, weight rank feature * input feature

def quadraticResidual {P O R : ℕ} (left right : Matrix (Fin R) (Fin P) ℝ)
    (outputWeight : Matrix (Fin O) (Fin R) ℝ) (input : Fin P → ℝ) (output : Fin O) : ℝ :=
  ∑ rank, outputWeight output rank * (projection left input rank * projection right input rank)

def quadraticHead {P O R : ℕ} (linear : Matrix (Fin O) (Fin P) ℝ) (bias : Fin O → ℝ)
    (left right : Matrix (Fin R) (Fin P) ℝ) (outputWeight : Matrix (Fin O) (Fin R) ℝ)
    (input : Fin P → ℝ) (output : Fin O) : ℝ :=
  affine linear bias input output + quadraticResidual left right outputWeight input output

def mlpHead {P O H : ℕ} (linear : Matrix (Fin O) (Fin P) ℝ) (bias : Fin O → ℝ)
    (hidden : Matrix (Fin H) (Fin P) ℝ) (hiddenBias : Fin H → ℝ)
    (nonlinearOutput : Matrix (Fin O) (Fin H) ℝ) (gelu : ℝ → ℝ)
    (input : Fin P → ℝ) (output : Fin O) : ℝ :=
  affine linear bias input output +
    ∑ unit, nonlinearOutput output unit * gelu (affine hidden hiddenBias input unit)

def quadraticKernel {P O R : ℕ} (left right : Matrix (Fin R) (Fin P) ℝ)
    (outputWeight : Matrix (Fin O) (Fin R) ℝ) (output : Fin O) (first second : Fin P) : ℝ :=
  ∑ rank, outputWeight output rank * left rank first * right rank second

theorem zero_quadratic_output_preserves_the_entire_affine_base
    {P O R : ℕ} (linear : Matrix (Fin O) (Fin P) ℝ) (bias : Fin O → ℝ)
    (left right : Matrix (Fin R) (Fin P) ℝ) (input : Fin P → ℝ) :
    quadraticHead linear bias left right (fun _ _ => 0) input = affine linear bias input := by
  funext output
  simp [quadraticHead, quadraticResidual]

theorem zero_MLP_output_preserves_the_entire_affine_base_for_the_actual_GELU_interface
    {P O H : ℕ} (linear : Matrix (Fin O) (Fin P) ℝ) (bias : Fin O → ℝ)
    (hidden : Matrix (Fin H) (Fin P) ℝ) (hiddenBias : Fin H → ℝ)
    (gelu : ℝ → ℝ) (input : Fin P → ℝ) :
    mlpHead linear bias hidden hiddenBias (fun _ _ => 0) gelu input = affine linear bias input := by
  funext output
  simp [mlpHead]

theorem actual_factorized_quadratic_residual_has_the_derived_bilinear_kernel
    {P O R : ℕ} (left right : Matrix (Fin R) (Fin P) ℝ)
    (outputWeight : Matrix (Fin O) (Fin R) ℝ) (input : Fin P → ℝ) (output : Fin O) :
    quadraticResidual left right outputWeight input output =
      ∑ first, ∑ second, quadraticKernel left right outputWeight output first second * input first * input second := by
  calc
    _ = ∑ rank, ∑ first, ∑ second,
        outputWeight output rank * left rank first * input first * right rank second * input second := by
      unfold quadraticResidual projection
      simp only [Finset.mul_sum, Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro rank _
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro first _
      apply Finset.sum_congr rfl
      intro second _
      ring
    _ = ∑ first, ∑ second, ∑ rank,
        outputWeight output rank * left rank first * input first * right rank second * input second := by
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro first _
      rw [Finset.sum_comm]
    _ = _ := by
      simp only [quadraticKernel, Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro first _
      apply Finset.sum_congr rfl
      intro second _
      apply Finset.sum_congr rfl
      intro rank _
      ring

theorem full_resolution_affine_readout_uses_each_encoded_channel_time_coordinate
    {C T O : ℕ} (weight : Matrix (Fin O) (Fin (C * T)) ℝ) (bias : Fin O → ℝ)
    (features : Matrix (Fin C) (Fin T) ℝ) (output : Fin O) :
    affine weight bias (NetworkReadoutGeometry.fullUnfold features) output =
      (∑ channel, ∑ time, weight output (finProdFinEquiv (channel, time)) * features channel time) +
        bias output := by
  unfold affine
  congr 1
  rw [← Equiv.sum_comp (finProdFinEquiv : Fin C × Fin T ≃ Fin (C * T))]
  simp [Fintype.sum_prod_type, NetworkReadoutGeometry.fullUnfold, MatrixNormalCoordinates.rowMajorVector_at]

abbrev LinearCells (P O : ℕ) := (Fin O × Fin P) ⊕ Fin O
abbrev QuadraticCells (P O R : ℕ) :=
  LinearCells P O ⊕ ((Fin R × Fin P) ⊕ ((Fin R × Fin P) ⊕ (Fin O × Fin R)))
abbrev MLPCells (P O H : ℕ) :=
  LinearCells P O ⊕ ((Fin H × Fin P) ⊕ (Fin H ⊕ (Fin O × Fin H)))
abbrev ResidualAdapterCells (C K B : ℕ) :=
  (Fin C × Fin K) ⊕ ((Fin B × Fin C) ⊕ (Fin B ⊕ ((Fin C × Fin B) ⊕ (Fin C ⊕ Fin 1))))

def allocatedLinear (P O : ℕ) : ℕ := Fintype.card (LinearCells P O)
def allocatedQuadratic (P O R : ℕ) : ℕ := Fintype.card (QuadraticCells P O R)
def allocatedMLP (P O H : ℕ) : ℕ := Fintype.card (MLPCells P O H)
def allocatedResidualAdapter (C K B : ℕ) : ℕ := Fintype.card (ResidualAdapterCells C K B)

theorem actual_parameter_arrays_give_the_three_source_head_counts (P O R H : ℕ) :
    allocatedLinear P O = O * P + O ∧
      allocatedQuadratic P O R = O * P + O + 2 * R * P + O * R ∧
      allocatedMLP P O H = O * P + O + H * P + H + O * H := by
  refine ⟨?_, ?_, ?_⟩
  · simp [allocatedLinear, LinearCells]
  · simp only [allocatedQuadratic, QuadraticCells, LinearCells, Fintype.card_sum,
      Fintype.card_prod, Fintype.card_fin]
    ring
  · simp only [allocatedMLP, MLPCells, LinearCells, Fintype.card_sum,
      Fintype.card_prod, Fintype.card_fin]
    ring

theorem actual_residual_adapter_arrays_include_both_biases_and_scalar_gate (C K B : ℕ) :
    allocatedResidualAdapter C K B = C * K + 2 * C * B + B + C + 1 := by
  simp only [allocatedResidualAdapter, ResidualAdapterCells, Fintype.card_sum,
    Fintype.card_prod, Fintype.card_fin]
  ring

theorem matched_feature_dependent_slopes_leave_a_real_parameter_count_gap (P R : ℕ) :
    allocatedMLP P 2 (2 * R) = allocatedQuadratic P 2 R + 4 * R := by
  rw [(actual_parameter_arrays_give_the_three_source_head_counts P 2 R (2 * R)).2.1,
    (actual_parameter_arrays_give_the_three_source_head_counts P 2 R (2 * R)).2.2]
  ring

theorem positive_rank_makes_the_source_default_MLP_strictly_larger (P R : ℕ) (positive : 0 < R) :
    allocatedQuadratic P 2 R < allocatedMLP P 2 (2 * R) := by
  rw [matched_feature_dependent_slopes_leave_a_real_parameter_count_gap]
  omega

def activeCells {I : Type*} [Fintype I] (requiresGrad : I → Bool) : ℕ :=
  ∑ cell, if requiresGrad cell = true then 1 else 0

def nonzeroCells {I : Type*} [Fintype I] (values : I → ℝ) : ℕ :=
  ∑ cell, if values cell ≠ 0 then 1 else 0

theorem actual_trainability_filter_distinguishes_allocated_and_frozen_storage
    {I : Type*} [Fintype I] :
    activeCells (fun _ : I => true) = Fintype.card I ∧
      activeCells (fun _ : I => false) = 0 := by simp [activeCells]

theorem zero_initialized_output_cells_remain_allocated_and_trainable (O R : ℕ)
    (outputsPositive : 0 < O) (rankPositive : 0 < R) :
    activeCells (fun _ : Fin O × Fin R => true) = O * R ∧
      nonzeroCells (fun _ : Fin O × Fin R => 0) = 0 ∧ 0 < O * R := by
  refine ⟨?_, ?_, Nat.mul_pos outputsPositive rankPositive⟩
  · simp [activeCells]
  · simp [nonzeroCells]

def nonzeroLinear : Matrix (Fin 2) (Fin 1) ℝ := fun output _ => if output = 0 then 1 else 2

theorem actual_zero_residual_initializations_do_not_make_the_classifier_zero (gelu : ℝ → ℝ) :
    quadraticHead nonzeroLinear (fun _ => 0) (fun _ : Fin 1 => fun _ => 1)
        (fun _ => fun _ => 1) (fun _ _ => 0) (fun _ => 3) 0 = 3 ∧
      mlpHead nonzeroLinear (fun _ => 0) (fun _ : Fin 1 => fun _ => 1) (fun _ => 1)
        (fun _ _ => 0) gelu (fun _ => 3) 0 = 3 := by
  rw [zero_quadratic_output_preserves_the_entire_affine_base,
    zero_MLP_output_preserves_the_entire_affine_base_for_the_actual_GELU_interface]
  norm_num [affine, nonzeroLinear]

theorem source_default_readout_parameter_counts_are_not_exactly_matched :
    allocatedQuadratic 128 2 8 = 2322 ∧ allocatedMLP 128 2 16 = 2354 ∧
      activeCells (fun _ : Fin 2 × Fin 8 => true) = 16 ∧
      activeCells (fun _ : LinearCells 128 2 => false) = 0 := by
  norm_num [allocatedQuadratic, allocatedMLP, QuadraticCells, MLPCells, LinearCells, activeCells]

end
end N2P3.NetworkClassifierHeads
