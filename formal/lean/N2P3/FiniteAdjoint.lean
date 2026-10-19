import N2P3.FiniteSOS
import Mathlib.LinearAlgebra.Matrix.PosDef

/-!
# The finite zero-state SOS adjoint identity

Derive the causal Toeplitz coefficients from the actual SOS sample recurrence,
then prove reversal-conjugation equals transpose. No Toeplitz, convolution or
adjoint property is assumed of the filter. All statements are over the reals.
-/

namespace N2P3.FiniteAdjoint

open N2P3.FiniteSOS
open scoped BigOperators Matrix ComplexOrder

noncomputable section

def spike {n : ℕ} (j : Fin n) : Samples n := fun i => if i = j then 1 else 0

theorem spike_zero_head (n : ℕ) : spike (0 : Fin (n + 1)) 0 = 1 := by
  simp [spike]

theorem spike_zero_tail (n : ℕ) :
    (fun i : Fin n => spike (0 : Fin (n + 1)) i.succ) = 0 := by
  funext i
  simp [spike]

theorem spike_succ_head {n : ℕ} (j : Fin n) : spike j.succ 0 = 0 := by
  simp [spike, eq_comm]

theorem spike_succ_tail {n : ℕ} (j : Fin n) :
    (fun i => spike j.succ i.succ) = spike j := by
  funext i
  simp [spike]

def stateNext {s : ℕ} (c : Fin s → Section) : State s →ₗ[ℝ] State s :=
  (LinearMap.snd ℝ _ _).comp ((cascade s c).comp (LinearMap.inr ℝ _ _))

def stateRead {s : ℕ} (c : Fin s → Section) : State s →ₗ[ℝ] ℝ :=
  (LinearMap.fst ℝ _ _).comp ((cascade s c).comp (LinearMap.inr ℝ _ _))

/-- Autonomous state evolution, derived from the actual cascade at input zero. -/
def stateAdvance {s : ℕ} (c : Fin s → Section) : ℕ → State s → State s
  | 0, z => z
  | k + 1, z => stateAdvance c k (stateNext c z)

/-- State response on every finite grid equals the corresponding autonomous
trajectory. In particular the coefficient is independent of chosen grid size. -/
theorem stateResponse_coordinate {s : ℕ} (c : Fin s → Section) :
    ∀ (n : ℕ) (i : Fin n) (z : State s),
      stateResponse c n z i = stateRead c (stateAdvance c i.val z)
  | 0, i, _ => Fin.elim0 i
  | n + 1, i, z => by
    refine Fin.cases ?_ (fun j => ?_) i
    · rfl
    · change stateResponse c n (stateNext c z) j =
        stateRead c (stateAdvance c j.val (stateNext c z))
      exact stateResponse_coordinate c n j (stateNext c z)

def impulseHead {s : ℕ} (c : Fin s → Section) : ℝ × State s :=
  cascade s c (1, 0)

/-- Impulse coefficients computed solely from the normalized SOS recurrence. -/
def kernel {s : ℕ} (c : Fin s → Section) : ℕ → ℝ
  | 0 => (impulseHead c).1
  | k + 1 => stateRead c (stateAdvance c k (impulseHead c).2)

def responseMatrix {s : ℕ} (c : Fin s → Section) (n : ℕ) : Matrix (Fin n) (Fin n) ℝ :=
  LinearMap.toMatrix' (response c n)

theorem responseMatrix_apply {s n : ℕ} (c : Fin s → Section) (i j : Fin n) :
    responseMatrix c n i j = response c n (spike j) i := by
  rw [responseMatrix, LinearMap.toMatrix'_apply]
  rfl

theorem responseMatrix_head_zero {s n : ℕ} (c : Fin s → Section) :
    responseMatrix c (n + 1) 0 0 = kernel c 0 := by
  rw [responseMatrix_apply]
  rfl

theorem responseMatrix_head_succ {s n : ℕ} (c : Fin s → Section) (j : Fin n) :
    responseMatrix c (n + 1) 0 j.succ = 0 := by
  rw [responseMatrix_apply]
  change (cascade s c (spike j.succ 0, 0)).1 = 0
  rw [spike_succ_head]
  change (cascade s c 0).1 = 0
  rw [map_zero]
  rfl

theorem responseMatrix_succ_zero {s n : ℕ} (c : Fin s → Section) (i : Fin n) :
    responseMatrix c (n + 1) i.succ 0 = kernel c (i.val + 1) := by
  rw [responseMatrix_apply]
  change (run c n ((fun j => spike (0 : Fin (n + 1)) j.succ),
    (cascade s c (spike (0 : Fin (n + 1)) 0, 0)).2)).1 i = _
  rw [spike_zero_tail, spike_zero_head]
  exact stateResponse_coordinate c n i (impulseHead c).2

theorem responseMatrix_succ_succ {s n : ℕ} (c : Fin s → Section) (i j : Fin n) :
    responseMatrix c (n + 1) i.succ j.succ = responseMatrix c n i j := by
  rw [responseMatrix_apply, responseMatrix_apply]
  change (run c n ((fun k => spike j.succ k.succ),
    (cascade s c (spike j.succ 0, 0)).2)).1 i = _
  rw [spike_succ_tail, spike_succ_head]
  change (run c n (spike j, (cascade s c 0).2)).1 i = _
  rw [map_zero]
  rfl

/-- Causal Toeplitz coefficients are a theorem of the implemented finite
recurrence. The proof is induction on the grid, including an empty grid. -/
theorem responseMatrix_toeplitz {s : ℕ} (c : Fin s → Section) :
    ∀ (n : ℕ) (i j : Fin n), responseMatrix c n i j =
      if j.val ≤ i.val then kernel c (i.val - j.val) else 0
  | 0, i, _ => Fin.elim0 i
  | n + 1, i, j => by
    refine Fin.cases ?_ (fun i' => ?_) i
    · refine Fin.cases ?_ (fun j' => ?_) j
      · simp only [Fin.val_zero, le_refl, ↓reduceIte, Nat.sub_zero]
        exact responseMatrix_head_zero c
      · rw [responseMatrix_head_succ]
        simp
    · refine Fin.cases ?_ (fun j' => ?_) j
      · rw [responseMatrix_succ_zero]
        simp
      · rw [responseMatrix_succ_succ, responseMatrix_toeplitz c n i' j']
        simp

/-- Reversing row and column swaps the causal Toeplitz indices. -/
theorem responseMatrix_reversal {s n : ℕ} (c : Fin s → Section) (i j : Fin n) :
    responseMatrix c n i.rev j.rev = responseMatrix c n j i := by
  rw [responseMatrix_toeplitz, responseMatrix_toeplitz]
  simp only [Fin.val_rev]
  have hi := i.isLt
  have hj := j.isLt
  by_cases h : i.val ≤ j.val
  · have hr : n - (j.val + 1) ≤ n - (i.val + 1) := by omega
    have hd : n - (i.val + 1) - (n - (j.val + 1)) = j.val - i.val := by omega
    simp only [h, hr, ↓reduceIte, hd]
  · have hr : ¬ n - (j.val + 1) ≤ n - (i.val + 1) := by omega
    simp only [h, hr, ↓reduceIte]

theorem reverse_spike {n : ℕ} (j : Fin n) : reverse n (spike j) = spike j.rev := by
  funext i
  simp [reverse, spike, Fin.rev_eq_iff]

/-- The actual finite causal SOS filter satisfies R H R = Hᵀ. -/
theorem reversal_adjoint {s : ℕ} (c : Fin s → Section) (n : ℕ) :
    LinearMap.toMatrix' ((reverse n).comp ((response c n).comp (reverse n))) =
      (responseMatrix c n).transpose := by
  ext i j
  rw [LinearMap.toMatrix'_apply]
  change response c n (reverse n (spike j)) i.rev = responseMatrix c n j i
  rw [reverse_spike]
  simpa only [responseMatrix_apply] using responseMatrix_reversal c i j

/-- Q from `FiniteFilterOperator.zero_state` is exactly HᵀH over the reals. -/
theorem zeroTwoPass_gram {s : ℕ} (c : Fin s → Section) (n : ℕ) :
    LinearMap.toMatrix' (zeroTwoPass c n) =
      (responseMatrix c n).transpose * responseMatrix c n := by
  have he : zeroTwoPass c n =
      ((reverse n).comp ((response c n).comp (reverse n))).comp (response c n) := rfl
  rw [he, LinearMap.toMatrix'_comp, reversal_adjoint]
  rfl

/-- The exact zero-state two-pass matrix is symmetric positive semidefinite;
strict definiteness is deliberately not claimed. -/
theorem zeroTwoPass_psd {s : ℕ} (c : Fin s → Section) (n : ℕ) :
    (LinearMap.toMatrix' (zeroTwoPass c n)).PosSemidef := by
  rw [zeroTwoPass_gram]
  simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using
    Matrix.posSemidef_conjTranspose_mul_self (responseMatrix c n)

end

end N2P3.FiniteAdjoint
