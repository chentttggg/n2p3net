import N2P3.SourceAuxiliaryRisk
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Field

/-!
Original V06.a L01, fixed risk_slice_contract_02 before new proof. Exact
realized CE values are supplied by the existing finite-score NLL primitive.
Class weights multiply numerator AND denominator; optional row multipliers
multiply only numerator. Batch contexts are retained: no BN/dropout or update
trajectory equivalence follows from a finite loss/reduction identity.
-/
namespace N2P3.TrainingCrossEntropyRisk
open scoped BigOperators
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

def classWeight (positive : ℝ) (label : Bool) : ℝ := if label then positive else 1

theorem actual_binary_class_weights_are_positive (positive : ℝ) (valid : 0<positive) (label : Bool) :
    0<classWeight positive label := by cases label <;> simp [classWeight,valid]

def denominator {I : Type*} [Fintype I] (weight : I → ℝ) : ℝ := ∑ row, weight row
def numerator {I : Type*} [Fintype I] (weight multiplier loss : I → ℝ) : ℝ :=
  ∑ row, weight row*multiplier row*loss row
def risk {I : Type*} [Fintype I] (weight multiplier loss : I → ℝ) : ℝ :=
  numerator weight multiplier loss/denominator weight
def coefficient {I : Type*} [Fintype I] (weight multiplier : I → ℝ) (row : I) : ℝ :=
  weight row*multiplier row/denominator weight

theorem actual_class_denominator_is_positive_on_nonempty_batches
    {I : Type*} [Fintype I] [Nonempty I] (weight : I → ℝ) (positive : ∀ row,0<weight row) :
    0<denominator weight := Finset.sum_pos (fun row _ => positive row) Finset.univ_nonempty

theorem actual_risk_expands_in_its_computed_row_coefficients
    {I : Type*} [Fintype I] (weight multiplier loss : I → ℝ) :
    risk weight multiplier loss=∑ row,coefficient weight multiplier row*loss row := by
  simp only [risk,numerator,coefficient,Finset.sum_div,div_mul_eq_mul_div₀]

theorem computed_coefficient_mass_is_weighted_row_mass_over_CLASS_mass
    {I : Type*} [Fintype I] (weight multiplier : I → ℝ) :
    (∑ row,coefficient weight multiplier row)=(∑ row,weight row*multiplier row)/denominator weight := by
  simp only [coefficient,Finset.sum_div]

theorem unit_row_multipliers_recover_unit_coefficient_mass
    {I : Type*} [Fintype I] (weight : I → ℝ) (nonzero : denominator weight ≠ 0) :
    (∑ row,coefficient weight (fun _ => 1) row)=1 := by
  rw [computed_coefficient_mass_is_weighted_row_mass_over_CLASS_mass]
  simp only [mul_one]
  change denominator weight/denominator weight=1
  exact div_self nonzero

theorem nonnegative_class_and_row_weights_give_nonnegative_risk
    {I : Type*} [Fintype I] (weight multiplier loss : I → ℝ)
    (classNonnegative : ∀ row,0≤weight row) (rowNonnegative : ∀ row,0≤multiplier row)
    (lossNonnegative : ∀ row,0≤loss row) : 0≤risk weight multiplier loss := by
  exact div_nonneg
    (Finset.sum_nonneg (fun row _ => mul_nonneg (mul_nonneg (classNonnegative row)
      (rowNonnegative row)) (lossNonnegative row)))
    (Finset.sum_nonneg (fun row _ => classNonnegative row))

theorem constant_CE_retains_the_computed_coefficient_mass
    {I : Type*} [Fintype I] (weight multiplier : I → ℝ) (value : ℝ) :
    risk weight multiplier (fun _ => value)=(∑ row,coefficient weight multiplier row)*value := by
  rw [actual_risk_expands_in_its_computed_row_coefficients,Finset.sum_mul]

def fiberSum {I B : Type*} [Fintype I] (block : I → B) (values : I → ℝ) (batch : B) : ℝ :=
  ∑ row,if block row=batch then values row else 0

theorem actual_assignment_fibers_sum_each_original_row_exactly_once
    {I B : Type*} [Fintype I] [Fintype B] (block : I → B) (values : I → ℝ) :
    (∑ batch,fiberSum block values batch)=∑ row,values row := by
  unfold fiberSum
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro row _
  simp

def batchNumerator {I B : Type*} [Fintype I] (block : I → B) (weight multiplier loss : I → ℝ) (batch : B) : ℝ :=
  fiberSum block (fun row => weight row*multiplier row*loss row) batch
def batchDenominator {I B : Type*} [Fintype I] (block : I → B) (weight : I → ℝ) (batch : B) : ℝ :=
  fiberSum block weight batch
def batchRisk {I B : Type*} [Fintype I] (block : I → B) (weight multiplier loss : I → ℝ) (batch : B) : ℝ :=
  batchNumerator block weight multiplier loss batch/batchDenominator block weight batch

theorem additive_validation_statistics_recover_the_actual_global_ratio
    {I B : Type*} [Fintype I] [Fintype B] (block : I → B) (weight multiplier loss : I → ℝ) :
    (∑ batch,batchNumerator block weight multiplier loss batch)/
      (∑ batch,batchDenominator block weight batch)=risk weight multiplier loss := by
  simp only [batchNumerator,batchDenominator,actual_assignment_fibers_sum_each_original_row_exactly_once,
    risk,numerator,denominator]

theorem correct_combination_weights_are_batch_CLASS_masses
    {I B : Type*} [Fintype I] [Fintype B] (block : I → B) (weight multiplier loss : I → ℝ)
    (nonzero : ∀ batch,batchDenominator block weight batch ≠ 0) :
    (∑ batch,(batchDenominator block weight batch/denominator weight)*batchRisk block weight multiplier loss batch)=
      risk weight multiplier loss := by
  have each : ∀ batch,(batchDenominator block weight batch/denominator weight)*
      batchRisk block weight multiplier loss batch=batchNumerator block weight multiplier loss batch/denominator weight := by
    intro batch
    unfold batchRisk
    field_simp [nonzero batch]
  simp_rw [each]
  rw [← Finset.sum_div]
  simp only [batchNumerator,actual_assignment_fibers_sum_each_original_row_exactly_once,risk,numerator]

def batchSize {I B : Type*} [Fintype I] (block : I → B) (batch : B) : ℝ := fiberSum block (fun _ => 1) batch

theorem row_size_weights_equal_CLASS_mass_weights_when_local_class_mean_is_constant
    {I B : Type*} [Fintype I] [Fintype B] (block : I → B) (weight : I → ℝ)
    (mean : ℝ) (nonzero : mean ≠ 0)
    (localMean : ∀ batch,batchDenominator block weight batch=mean*batchSize block batch) (batch : B) :
    batchDenominator block weight batch/denominator weight=batchSize block batch/(Fintype.card I : ℝ) := by
  have total : denominator weight=mean*(Fintype.card I : ℝ) := by
    calc
      _ = ∑ batch,batchDenominator block weight batch :=
        (actual_assignment_fibers_sum_each_original_row_exactly_once block weight).symm
      _ = mean*(∑ batch,batchSize block batch) := by simp_rw [localMean]; rw [Finset.mul_sum]
      _ = _ := by simp [batchSize,actual_assignment_fibers_sum_each_original_row_exactly_once]
  rw [localMean,total]
  exact mul_div_mul_left _ _ nonzero

def drawRisk {I : Type*} {batch : ℕ} (rows : Fin batch → I)
    (importance : I → ℝ) (lossBySlot : Fin batch → ℝ) : ℝ :=
  (∑ position,importance (rows position)*lossBySlot position)/(batch : ℝ)

def drawCoefficient {I : Type*} {batch : ℕ} (rows : Fin batch → I)
    (importance : I → ℝ) (row : I) : ℝ :=
  (∑ position,if rows position=row then importance row else 0)/(batch : ℝ)

theorem repeated_CE_draw_slots_retain_their_actual_physical_row_multiplicity
    {I : Type*} {batch : ℕ} (rows : Fin batch → I) (importance : I → ℝ) (row : I) :
    drawCoefficient rows importance row=
      (((Finset.univ.filter (fun position => rows position=row)).card : ℝ)*importance row)/(batch : ℝ) := by
  simp [drawCoefficient,Finset.sum_ite,Finset.sum_const,nsmul_eq_mul]

theorem frozen_row_CE_draw_risk_uses_derived_multiplicity_coefficients
    {I : Type*} [Fintype I] {batch : ℕ} (rows : Fin batch → I) (importance loss : I → ℝ) :
    drawRisk rows importance (fun position => loss (rows position))=
      ∑ row,drawCoefficient rows importance row*loss row := by
  unfold drawRisk drawCoefficient
  simp only [div_mul_eq_mul_div₀,← Finset.sum_div]
  congr 1
  symm
  simp_rw [Finset.sum_mul,ite_mul,zero_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro position _
  simp

def updateMean {B : Type*} [Fintype B] (batchLoss : B → ℝ) : ℝ :=
  (∑ batch,batchLoss batch)/(Fintype.card B : ℝ)
def rowReport {B : Type*} [Fintype B] (sizes : B → ℝ) (batchLoss : B → ℝ) : ℝ :=
  (∑ batch,sizes batch*batchLoss batch)/(∑ batch,sizes batch)

end
end N2P3.TrainingCrossEntropyRisk
