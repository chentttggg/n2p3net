import N2P3.FixedResampleInterface
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic.FinCases

/-! Independent exact fixtures R01-R06 fixed in the source-before-proof
requirements. The four-position fixture exercises only the legal nonempty
quantile primitive, not paired_interval with an illegal public draw count. -/

namespace N2P3.FixedResampleWitnesses

open N2P3.FixedResampleMean N2P3.FixedResampleBatches N2P3.LinearQuantile
open N2P3.RequestedAggregationWeights (mean)

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

def units3 : Fin 3 → ℝ := ![0,3,7]
def repeatedIndices : Fin 3 → Fin 3 := ![0,0,1]

theorem repeated_draw_positions_retain_their_multiplicity :
    drawMean units3 repeatedIndices=1 ∧
      mean ({0,1} : Finset (Fin 3)) units3=3/2 := by
  norm_num [drawMean,mean,units3,repeatedIndices,Fin.sum_univ_succ]

theorem paired_batch_tail_is_one_position : takeCount 4001 4000 4000=1 := by
  norm_num [takeCount]

theorem study_batch_tail_is_one_position : takeCount 1001 1000 1000=1 := by
  norm_num [takeCount]

def unequalStudySizes : Fin 2 → ℕ := ![1,3]
def unequalStudyUnits (study : Fin 2) (_ : Fin (unequalStudySizes study)) : ℝ :=
  if study=0 then 1 else 0

theorem singleton_study_keeps_equal_study_weight_in_every_draw :
    studyDrawMean (Finset.univ : Finset (Fin 2)) unequalStudySizes unequalStudyUnits (fun _ => id)=1/2 ∧
      mean (Finset.univ : Finset (Fin 4)) (![1,0,0,0] : Fin 4 → ℝ)=1/4 := by
  norm_num [studyDrawMean,drawMean,mean,unequalStudyUnits,unequalStudySizes,Fin.sum_univ_succ]

def raw4 : Fin 4 → ℝ := ![12,0,10,2]

def sortingPermutation : Equiv.Perm (Fin 4) where
  toFun := ![1,3,2,0]
  invFun := ![3,0,2,1]
  left_inv := by intro index; fin_cases index <;> rfl
  right_inv := by intro index; fin_cases index <;> rfl

def ordered4 : OrderedSample 4 where
  raw := raw4
  rank := sortingPermutation
  ordered := by
    intro left right ordered
    fin_cases left <;> fin_cases right <;>
      dsimp [raw4,sortingPermutation] at * <;> norm_num [Fin.le_def] at *

theorem sorted_linear_quantile_fixture_has_the_independently_required_endpoints :
    quantile ordered4 (by omega) (1/40) (by norm_num)=3/20 ∧
      quantile ordered4 (by omega) (39/40) (by norm_num)=237/20 := by
  have lowVirtual : virtualIndex 4 (1/40)=3/40 := by norm_num [virtualIndex]
  have highVirtual : virtualIndex 4 (39/40)=117/40 := by norm_num [virtualIndex]
  have lowFloor : ⌊(3/40 : ℝ)⌋₊=0 := (Nat.floor_eq_iff (by norm_num)).mpr (by norm_num)
  have highFloor : ⌊(117/40 : ℝ)⌋₊=2 := (Nat.floor_eq_iff (by norm_num)).mpr (by norm_num)
  norm_num [quantile,fraction,lowerIndex,upperIndex,lowVirtual,highVirtual,lowFloor,highFloor,
    OrderedSample.at,ordered4,raw4,sortingPermutation,Matrix.vecCons,Fin.cons]
  change (37/40 : ℝ)*0+(3/40)*2=3/20 ∧ (3/40 : ℝ)*10+(37/40)*12=237/20
  norm_num

theorem interpolating_unsorted_raw_positions_changes_the_low_endpoint :
    (1-(3/40 : ℝ))*raw4 0+(3/40)*raw4 1=111/10 ∧
      quantile ordered4 (by omega) (1/40) (by norm_num)≠111/10 := by
  constructor
  · norm_num [raw4]
  · rw [sorted_linear_quantile_fixture_has_the_independently_required_endpoints.1]
    norm_num

end
end N2P3.FixedResampleWitnesses
