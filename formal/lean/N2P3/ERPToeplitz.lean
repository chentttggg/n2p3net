import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Data.Real.StarOrdered
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Data.Int.Interval
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.FinCases
import Lean.Elab.Tactic.Omega

/-!
# The actual tapered Toeplitz loop is a finite shift-congruence average

Source: baselines/structured_erp.py:37-48. Coordinates are (time,channel),
with source channel-prime flat index time*C+channel. Both loops and the two
ordered block writes are explicit. The lower write is last, including at
lag zero. Starting storage is arbitrary: no zero value is attributed to
NumPy empty_like. Every coordinate is overwritten by its own lag/time pair.

The general congruence identity requires exact symmetric input. PSD supplies
that symmetry; source floating allclose is not substituted for it. The source
fit path uses float64 covariance. Integer empty_like assignment truncation,
floating sum/division rounding and numerical PSD tolerances are not refined
by the exact-real statements here. No training or stochastic independence is
asserted.
-/

namespace N2P3.ERPToeplitz

open Matrix
open scoped BigOperators Matrix

noncomputable section

abbrev Coordinate (times channels : ℕ) := Fin times × Fin channels
abbrev Covariance (times channels : ℕ) :=
  Matrix (Coordinate times channels) (Coordinate times channels) ℝ

variable {T C : ℕ}

def sourceColumn : Coordinate T C ≃ Fin (T*C) := finProdFinEquiv

theorem source_column_is_time_major (i : Coordinate T C) :
    (sourceColumn i).val=C*i.1.val+i.2.val := by
  simp [sourceColumn,finProdFinEquiv,Nat.add_comm]

def entry (A : Covariance T C) (t s : ℕ) (c d : Fin C) : ℝ :=
  if ht : t < T then if hs : s < T then A (⟨t,ht⟩,c) (⟨s,hs⟩,d) else 0 else 0

theorem entry_fin (A : Covariance T C) (t s : Fin T) (c d : Fin C) :
    entry A t.val s.val c d = A (t,c) (s,d) := by simp [entry]

/-- Literal sum(range(T-lag))/T; no shorter denominator is introduced. -/
def lagAverage (A : Covariance T C) (lag : ℕ) (c d : Fin C) : ℝ :=
  (∑ u ∈ Finset.range (T-lag), entry A u (u+lag) c d) / (T : ℝ)

/-- First assign the upper block, then assign its transpose to the lower one. -/
def pairWrite (A : Covariance T C) (lag time : ℕ) (initial : Covariance T C) :
    Covariance T C := fun i j =>
  if i.1.val = time+lag ∧ j.1.val = time then lagAverage A lag j.2 i.2
  else if i.1.val = time ∧ j.1.val = time+lag then lagAverage A lag i.2 j.2
  else initial i j

def lagLoop (A : Covariance T C) (lag : ℕ) (initial : Covariance T C) : Covariance T C :=
  (List.range (T-lag)).foldl (fun acc time => pairWrite A lag time acc) initial

def sourceLoop (A initial : Covariance T C) : Covariance T C :=
  (List.range T).foldl (fun acc lag => lagLoop A lag acc) initial

def targetLag (i j : Coordinate T C) : ℕ :=
  if i.1.val < j.1.val then j.1.val-i.1.val else i.1.val-j.1.val

def targetTime (i j : Coordinate T C) : ℕ := min i.1.val j.1.val

/-- At equal times the last (transposed lower-block) assignment is observed. -/
def sourceCoordinate (A : Covariance T C) (i j : Coordinate T C) : ℝ :=
  if i.1.val < j.1.val then lagAverage A (targetLag i j) i.2 j.2
  else lagAverage A (targetLag i j) j.2 i.2

theorem target_lag_inside (i j : Coordinate T C) : targetLag i j < T := by
  have hi:=i.1.isLt
  have hj:=j.1.isLt
  unfold targetLag
  split_ifs <;> omega

theorem target_time_inside_lag (i j : Coordinate T C) : targetTime i j < T-targetLag i j := by
  have hi:=i.1.isLt
  have hj:=j.1.isLt
  unfold targetTime targetLag
  split_ifs <;> omega

theorem pairWrite_coordinate (A initial : Covariance T C) (lag time : ℕ)
    (i j : Coordinate T C) :
    pairWrite A lag time initial i j =
      if lag=targetLag i j ∧ time=targetTime i j then sourceCoordinate A i j else initial i j := by
  by_cases hij : i.1.val < j.1.val
  · have hlo : ¬(i.1.val=time+lag ∧ j.1.val=time) := by omega
    have hu : (i.1.val=time ∧ j.1.val=time+lag) ↔
        (lag=j.1.val-i.1.val ∧ time=i.1.val) := by omega
    simp only [pairWrite,if_neg hlo,sourceCoordinate,targetLag,targetTime,
      if_pos hij,min_eq_left (Nat.le_of_lt hij)]
    by_cases h : i.1.val=time ∧ j.1.val=time+lag
    · have hh:=hu.mp h
      rw [if_pos h,if_pos hh,hh.1]
    · rw [if_neg h,if_neg (show ¬(lag=j.1.val-i.1.val ∧ time=i.1.val) from fun hh=>h (hu.mpr hh))]
  · have hji : j.1.val ≤ i.1.val := by omega
    have hl : (i.1.val=time+lag ∧ j.1.val=time) ↔
        (lag=i.1.val-j.1.val ∧ time=j.1.val) := by omega
    simp only [pairWrite,sourceCoordinate,targetLag,targetTime,
      if_neg hij,min_eq_right hji]
    by_cases h : i.1.val=time+lag ∧ j.1.val=time
    · have hh:=hl.mp h
      rw [if_pos h,if_pos hh,hh.1]
    · have hu : ¬(i.1.val=time ∧ j.1.val=time+lag) := by omega
      rw [if_neg h,if_neg hu,if_neg (show ¬(lag=i.1.val-j.1.val ∧ time=j.1.val) from fun hh=>h (hl.mpr hh))]

theorem fold_overwrite_coordinate {M α : Type*} [DecidableEq α]
    (xs : List α) (target : α) (update : α → M → M) (read : M → ℝ)
    (value : ℝ) (initial : M)
    (h : ∀ a m, read (update a m) = if a=target then value else read m) :
    read (xs.foldl (fun m a=>update a m) initial) =
      if target ∈ xs then value else read initial := by
  induction xs generalizing initial with
  | nil => simp
  | cons a tail ih =>
    rw [List.foldl_cons,ih,h]
    by_cases ht : target∈tail
    · simp only [List.mem_cons,ht,or_true,if_true]
    · simp only [List.mem_cons,ht,or_false,if_false]
      by_cases ha : a=target
      · subst a; simp
      · rw [if_neg ha,if_neg (Ne.symm ha)]

theorem lagLoop_coordinate (A initial : Covariance T C) (lag : ℕ) (i j : Coordinate T C) :
    lagLoop A lag initial i j =
      if lag=targetLag i j then sourceCoordinate A i j else initial i j := by
  by_cases hl : lag=targetLag i j
  · subst lag
    have h := fold_overwrite_coordinate (List.range (T-targetLag i j)) (targetTime i j)
      (fun time initial=>pairWrite A (targetLag i j) time initial) (fun m=>m i j)
      (sourceCoordinate A i j) initial (by intro time m; simp [pairWrite_coordinate])
    simpa [lagLoop,List.mem_range,target_time_inside_lag] using h
  · have hfixed : ∀ xs : List ℕ, ∀ initial : Covariance T C,
        xs.foldl (fun acc time=>pairWrite A lag time acc) initial i j=initial i j := by
      intro xs
      induction xs with
      | nil => simp
      | cons time tail ih => intro initial; simp [List.foldl_cons,ih,pairWrite_coordinate,hl]
    simpa [lagLoop,hl] using hfixed (List.range (T-lag)) initial

theorem sourceLoop_all_coordinates (A initial : Covariance T C) (i j : Coordinate T C) :
    sourceLoop A initial i j = sourceCoordinate A i j := by
  have h:=fold_overwrite_coordinate (List.range T) (targetLag i j)
    (fun lag initial=>lagLoop A lag initial) (fun m=>m i j)
    (sourceCoordinate A i j) initial (by intro lag m; exact lagLoop_coordinate A m lag i j)
  simpa [sourceLoop,List.mem_range,target_lag_inside] using h

theorem sourceLoop_independent_of_initial (A initial other : Covariance T C) :
    sourceLoop A initial=sourceLoop A other := by
  ext i j
  rw [sourceLoop_all_coordinates,sourceLoop_all_coordinates]

def shifts (times : ℕ) : Finset ℤ := Finset.Icc (1-(times:ℤ)) ((times:ℤ)-1)

def shiftMatrix (k : ℤ) : Covariance T C := fun i j =>
  if (j.1.val:ℤ)+k=(i.1.val:ℤ) ∧ j.2=i.2 then 1 else 0

def shiftAverage (A : Covariance T C) : Covariance T C :=
  (T:ℝ)⁻¹ • ∑ k ∈ shifts T, shiftMatrix k * A * (shiftMatrix k)ᵀ

theorem shift_congruence_entry (A : Covariance T C) (k : ℤ)
    (t s : Fin T) (c d : Fin C) :
    (shiftMatrix (T:=T) (C:=C) k * A * (shiftMatrix (T:=T) (C:=C) k)ᵀ : Covariance T C) (t,c) (s,d) =
      ∑ v : Fin T, ∑ u : Fin T,
        if (u.val:ℤ)+k=t.val ∧ (v.val:ℤ)+k=s.val then A (u,c) (v,d) else 0 := by
  simp only [Matrix.mul_apply,Matrix.transpose_apply,Fintype.sum_prod_type]
  simp [shiftMatrix,ite_and]
  apply Finset.sum_congr rfl
  intro v _
  by_cases h : (v.val:ℤ)+k=s.val <;> simp [h]

theorem shift_difference_mem (t u : Fin T) : (t.val:ℤ)-u.val ∈ shifts T := by
  have ht:=t.isLt
  have hu:=u.isLt
  simp only [shifts,Finset.mem_Icc]
  omega

theorem sum_shift_congruence_entry (A : Covariance T C) (t s : Fin T) (c d : Fin C) :
    (∑ k ∈ shifts T, shiftMatrix k * A * (shiftMatrix k)ᵀ : Covariance T C) (t,c) (s,d) =
      ∑ v : Fin T, ∑ u : Fin T,
        if (v.val:ℤ)-u.val=(s.val:ℤ)-t.val then A (u,c) (v,d) else 0 := by
  have hk (u v : Fin T) :
      (∑ k ∈ shifts T,
        if (u.val:ℤ)+k=t.val ∧ (v.val:ℤ)+k=s.val then A (u,c) (v,d) else 0) =
      if (v.val:ℤ)-u.val=(s.val:ℤ)-t.val then A (u,c) (v,d) else 0 := by
    rw [Finset.sum_eq_single ((t.val:ℤ)-u.val)]
    · have he : (u.val:ℤ)+((t.val:ℤ)-u.val)=t.val := by ring
      simp only [he,true_and]
      have heq : ((v.val:ℤ)+((t.val:ℤ)-u.val)=s.val) ↔
          (v.val:ℤ)-u.val=(s.val:ℤ)-t.val := by omega
      simp only [heq]
    · intro b _ hb
      have hn : ¬((u.val:ℤ)+b=t.val ∧ (v.val:ℤ)+b=s.val) := by omega
      simp only [if_neg hn]
    · intro hn
      exact False.elim (hn (shift_difference_mem t u))
  simp only [Matrix.sum_apply,shift_congruence_entry]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro v _
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro u _
  exact hk u v

theorem sum_diagonal_upper (A : Covariance T C) (t s : Fin T) (c d : Fin C)
    (h : t.val ≤ s.val) :
    (∑ v : Fin T, ∑ u : Fin T,
      if (v.val:ℤ)-u.val=(s.val:ℤ)-t.val then A (u,c) (v,d) else 0) =
      ∑ u : Fin T, entry A u.val (u.val+(s.val-t.val)) c d := by
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro u _
  by_cases hu : u.val+(s.val-t.val)<T
  · let v : Fin T := ⟨u.val+(s.val-t.val),hu⟩
    rw [Finset.sum_eq_single v]
    · have he : (v.val:ℤ)-u.val=(s.val:ℤ)-t.val := by dsimp [v]; omega
      simp only [if_pos he]
      simp [entry,u.isLt,hu,v]
    · intro b _ hb
      have he : ¬((b.val:ℤ)-u.val=(s.val:ℤ)-t.val) := by
        intro he
        apply hb
        apply Fin.ext
        dsimp [v]
        omega
      simp only [if_neg he]
    · intro hn
      exact False.elim (hn (Finset.mem_univ v))
  · have he (v : Fin T) : ¬((v.val:ℤ)-u.val=(s.val:ℤ)-t.val) := by
      have hv:=v.isLt
      omega
    simp [he,entry,u.isLt,hu]

theorem sum_lag_range (A : Covariance T C) (lag : ℕ) (c d : Fin C) :
    (∑ u : Fin T, entry A u.val (u.val+lag) c d) =
      ∑ u ∈ Finset.range (T-lag), entry A u (u+lag) c d := by
  rw [Fin.sum_univ_eq_sum_range (fun u : ℕ=>entry A u (u+lag) c d)]
  symm
  apply Finset.sum_subset (Finset.range_mono (Nat.sub_le T lag))
  intro u hu hn
  have hl : ¬u+lag<T := by
    simp only [Finset.mem_range] at hu hn
    omega
  simp [entry,hl]

theorem shiftAverage_upper_coordinate (A : Covariance T C) (t s : Fin T) (c d : Fin C)
    (h : t.val ≤ s.val) :
    shiftAverage A (t,c) (s,d)=lagAverage A (s.val-t.val) c d := by
  simp only [shiftAverage,Matrix.smul_apply,smul_eq_mul]
  rw [sum_shift_congruence_entry,sum_diagonal_upper A t s c d h,sum_lag_range]
  rw [lagAverage,div_eq_mul_inv,mul_comm]

theorem shiftAverage_symmetric (A : Covariance T C) (h : Aᵀ=A) :
    (shiftAverage A)ᵀ=shiftAverage A := by
  simp [shiftAverage,Matrix.transpose_smul,Matrix.transpose_sum,Matrix.transpose_mul,
    h,Matrix.mul_assoc]

theorem sourceLoop_equals_shiftAverage (A initial : Covariance T C) (h : Aᵀ=A) :
    sourceLoop A initial=shiftAverage A := by
  ext i j
  rw [sourceLoop_all_coordinates]
  rcases i with ⟨t,c⟩
  rcases j with ⟨s,d⟩
  by_cases ht : t.val<s.val
  · simp only [sourceCoordinate,targetLag,if_pos ht]
    exact (shiftAverage_upper_coordinate A t s c d ht.le).symm
  · have hs : s.val ≤ t.val := by omega
    simp only [sourceCoordinate,targetLag,if_neg ht]
    have he := congrArg (fun M : Covariance T C=>M (t,c) (s,d)) (shiftAverage_symmetric A h)
    change shiftAverage A (s,d) (t,c)=shiftAverage A (t,c) (s,d) at he
    rw [← he,shiftAverage_upper_coordinate A s t d c hs]

theorem shiftMatrix_zero : shiftMatrix (T:=T) (C:=C) 0=1 := by
  ext i j
  simp only [shiftMatrix,add_zero,Int.natCast_inj,Matrix.one_apply]
  have he : (j.1.val=i.1.val ∧ j.2=i.2) ↔ i=j := by
    constructor
    · rintro ⟨ht,hc⟩
      apply Prod.ext
      · exact (Fin.ext ht).symm
      · exact hc.symm
    · intro hh; subst j; simp
  simp only [he]

theorem sumShift_posSemidef (A : Covariance T C) (hA : A.PosSemidef) (s : Finset ℤ) :
    (∑ k ∈ s, shiftMatrix k * A * (shiftMatrix k)ᵀ : Covariance T C).PosSemidef := by
  have hf (k : ℤ) : (shiftMatrix k * A * (shiftMatrix k)ᵀ).PosSemidef := by
    simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using
      hA.mul_mul_conjTranspose_same (shiftMatrix k)
  induction s using Finset.induction_on with
  | empty => simpa using (Matrix.PosSemidef.zero : (0 : Covariance T C).PosSemidef)
  | @insert k s hk ih => rw [Finset.sum_insert hk]; exact (hf k).add ih

theorem shiftAverage_posSemidef (A : Covariance T C) (hA : A.PosSemidef) :
    (shiftAverage A).PosSemidef :=
  (sumShift_posSemidef A hA (shifts T)).smul (inv_nonneg.mpr (Nat.cast_nonneg T))

theorem sourceLoop_posSemidef (A initial : Covariance T C) (hA : A.PosSemidef) :
    (sourceLoop A initial).PosSemidef := by
  have he : Aᵀ=A := by simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using hA.1
  rw [sourceLoop_equals_shiftAverage A initial he]
  exact shiftAverage_posSemidef A hA

theorem zero_shift_inside (hT : 0<T) : (0:ℤ)∈shifts T := by
  simp only [shifts,Finset.mem_Icc]
  omega

theorem shiftAverage_posDef (A : Covariance T C) (hA : A.PosDef) (hT : 0<T) :
    (shiftAverage A).PosDef := by
  have hs := sumShift_posSemidef A hA.posSemidef ((shifts T).erase 0)
  have he : (∑ k ∈ shifts T, shiftMatrix k * A * (shiftMatrix k)ᵀ : Covariance T C) =
      A+∑ k ∈ (shifts T).erase 0, shiftMatrix k*A*(shiftMatrix k)ᵀ := by
    rw [← Finset.add_sum_erase (shifts T) (fun k=>shiftMatrix k*A*(shiftMatrix k)ᵀ)
      (zero_shift_inside hT),shiftMatrix_zero]
    simp
  have hp := hA.add_posSemidef hs
  have ht : (0:ℝ)<T := by exact_mod_cast hT
  unfold shiftAverage
  rw [he]
  exact hp.smul (inv_pos.mpr ht)

theorem sourceLoop_posDef (A initial : Covariance T C) (hA : A.PosDef) (hT : 0<T) :
    (sourceLoop A initial).PosDef := by
  have he : Aᵀ=A := by simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using hA.1
  rw [sourceLoop_equals_shiftAverage A initial he]
  exact shiftAverage_posDef A hA hT

def untapered (A : Covariance T C) : Covariance T C := fun i j =>
  let lag:=targetLag i j
  if i.1.val<j.1.val then
    (∑ u ∈ Finset.range (T-lag),entry A u (u+lag) i.2 j.2)/(T-lag:ℕ)
  else (∑ u ∈ Finset.range (T-lag),entry A u (u+lag) j.2 i.2)/(T-lag:ℕ)

def rankOneVector : Coordinate 3 1 → ℝ := fun i=> ![1,0,1] i.1
def rankOneCovariance : Covariance 3 1 := vecMulVec rankOneVector rankOneVector
def differenceVector : Coordinate 3 1 → ℝ := fun i=> ![1,0,-1] i.1

theorem rankOneCovariance_posSemidef : rankOneCovariance.PosSemidef := by
  simpa only [rankOneCovariance,star_trivial] using
    Matrix.posSemidef_vecMulVec_self_star rankOneVector

theorem source_rank_one_corner : sourceLoop rankOneCovariance 0 (0,0) (2,0)=1/3 := by
  rw [sourceLoop_all_coordinates]
  norm_num [sourceCoordinate,targetLag,lagAverage,entry,rankOneCovariance,vecMulVec,rankOneVector,
    Finset.sum_range_succ]

theorem untapered_rank_one_matrix : untapered rankOneCovariance =
    fun i j=> (!![2/3,0,1;0,2/3,0;1,0,2/3] : Matrix (Fin 3) (Fin 3) ℝ) i.1 j.1 := by
  ext ⟨i,c⟩ ⟨j,d⟩
  fin_cases i <;> fin_cases j <;> fin_cases c <;> fin_cases d <;>
    norm_num [untapered,targetLag,entry,rankOneCovariance,vecMulVec,rankOneVector,
      Finset.sum_range_succ]

theorem untapered_negative_quadratic :
    differenceVector ⬝ᵥ (untapered rankOneCovariance *ᵥ differenceVector) = -2/3 := by
  rw [untapered_rank_one_matrix]
  norm_num [dotProduct,Matrix.mulVec,Fintype.sum_prod_type,Fin.sum_univ_three,
    differenceVector,Fin.sum_univ_one,Matrix.cons_val_two,Matrix.vecHead,Matrix.vecTail]

theorem untapered_not_posSemidef : ¬(untapered rankOneCovariance).PosSemidef := by
  intro h
  have hp:=h.2 differenceVector
  rw [star_trivial,untapered_negative_quadratic] at hp
  norm_num at hp

def missingChannel : Covariance 2 2 := diagonal (fun i=>if i.2=0 then 1 else 0)
def missingDirection : Coordinate 2 2 → ℝ := fun i=>if i=(0,1) then 1 else 0

theorem missingChannel_posSemidef : missingChannel.PosSemidef := by
  apply Matrix.PosSemidef.diagonal
  intro i
  dsimp
  split_ifs <;> norm_num

theorem missingChannel_nonzero : missingChannel≠0 := by
  intro h
  have he:=congrArg (fun A : Covariance 2 2=>A (0,0) (0,0)) h
  norm_num [missingChannel,Matrix.diagonal_apply] at he

theorem source_missing_channel_unchanged : sourceLoop missingChannel 0=missingChannel := by
  ext ⟨i,c⟩ ⟨j,d⟩
  rw [sourceLoop_all_coordinates]
  fin_cases i <;> fin_cases j <;> fin_cases c <;> fin_cases d <;>
    norm_num [sourceCoordinate,targetLag,lagAverage,entry,missingChannel,Matrix.diagonal_apply,
      Finset.sum_range_succ]

theorem missingDirection_nonzero : missingDirection≠0 := by
  intro h
  have he:=congrFun h ((0,1):Coordinate 2 2)
  norm_num [missingDirection] at he

theorem source_missing_channel_not_posDef : ¬(sourceLoop missingChannel 0).PosDef := by
  rw [source_missing_channel_unchanged]
  intro h
  have hp:=h.2 missingDirection missingDirection_nonzero
  norm_num [star_trivial,missingDirection,missingChannel,Matrix.mulVec_diagonal,dotProduct,
    Fintype.sum_prod_type,Fin.sum_univ_two] at hp

def asymmetricBlock : Covariance 1 2 := fun i j=> ( !![1,2;3,4] : Matrix (Fin 2) (Fin 2) ℝ) i.2 j.2

theorem diagonal_last_write_transposes : sourceLoop asymmetricBlock 0 (0,0) (0,1)=3 := by
  rw [sourceLoop_all_coordinates]
  norm_num [sourceCoordinate,targetLag,lagAverage,entry,asymmetricBlock,Finset.sum_range_succ]

end

end N2P3.ERPToeplitz
