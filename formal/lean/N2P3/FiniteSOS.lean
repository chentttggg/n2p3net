import Mathlib.LinearAlgebra.Matrix.ToLin
import Mathlib.Data.Real.Basic
import Mathlib.Data.Fin.VecNotation
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Linarith
import Lean.Elab.Tactic.Omega
import Mathlib.Tactic.Ring

/-!
# Finite normalized SOS filtering

Real arithmetic specification of SciPy's normalized direct form II transposed
SOS recurrence and `data.preprocess.filter_bidirectional_segment`. Coefficients
and the `sosfilt_zi` array are fixed. Finite identities require neither stability
nor a random-state model. Exact unit-step equilibrium coefficients are derived
below. SciPy floating point and its solver implementation remain refinement
obligations, not assumptions that filtering is linear.
-/

namespace N2P3.FiniteSOS

noncomputable section

open scoped BigOperators

/-- A normalized SOS row [b0,b1,b2,1,a1,a2]. -/
structure Section where
  b0 : ℝ
  b1 : ℝ
  b2 : ℝ
  a1 : ℝ
  a2 : ℝ

/-- SciPy order is section first, then the two DFII-T coordinates. -/
abbrev State (s : ℕ) := Fin s → Fin 2 → ℝ
abbrev Samples (n : ℕ) := Fin n → ℝ

/-- y=b0*x+z0, z0'=b1*x-a1*y+z1, z1'=b2*x-a2*y. -/
def sectionStep (c : Section) : (ℝ × (Fin 2 → ℝ)) →ₗ[ℝ] (ℝ × (Fin 2 → ℝ)) where
  toFun p :=
    let y := c.b0 * p.1 + p.2 0
    (y, ![c.b1 * p.1 - c.a1 * y + p.2 1, c.b2 * p.1 - c.a2 * y])
  map_add' p q := by
    apply Prod.ext
    · simp
      ring
    · funext i
      fin_cases i <;> simp <;> ring
  map_smul' r p := by
    apply Prod.ext
    · simp
      ring
    · funext i
      fin_cases i <;> simp <;> ring

theorem section_output (c : Section) (x : ℝ) (z : Fin 2 → ℝ) :
    (sectionStep c (x, z)).1 = c.b0 * x + z 0 := rfl

theorem section_state0 (c : Section) (x : ℝ) (z : Fin 2 → ℝ) :
    (sectionStep c (x, z)).2 0 =
      c.b1 * x - c.a1 * (c.b0 * x + z 0) + z 1 := rfl

theorem section_state1 (c : Section) (x : ℝ) (z : Fin 2 → ℝ) :
    (sectionStep c (x, z)).2 1 =
      c.b2 * x - c.a2 * (c.b0 * x + z 0) := rfl

def dcGain (c : Section) : ℝ :=
  (c.b0 + c.b1 + c.b2) / (1 + c.a1 + c.a2)

/-- The exact-real unit-step equilibrium solved by `lfilter_zi` for a
normalized section, before cascade gain scaling in `sosfilt_zi`. -/
def sectionZi (c : Section) : Fin 2 → ℝ :=
  ![dcGain c - c.b0, c.b2 - c.a2 * dcGain c]

theorem sectionZi_equilibrium (c : Section) (h : 1 + c.a1 + c.a2 ≠ 0) :
    sectionStep c (1, sectionZi c) = (dcGain c, sectionZi c) := by
  have hb : dcGain c * (1 + c.a1 + c.a2) = c.b0 + c.b1 + c.b2 :=
    div_mul_cancel₀ _ h
  apply Prod.ext
  · change c.b0 * 1 + (dcGain c - c.b0) = dcGain c
    ring
  · funext i
    fin_cases i
    · change c.b1 * 1 - c.a1 * (c.b0 * 1 + (dcGain c - c.b0)) +
        (c.b2 - c.a2 * dcGain c) = dcGain c - c.b0
      nlinarith [hb]
    · change c.b2 * 1 - c.a2 * (c.b0 * 1 + (dcGain c - c.b0)) =
        c.b2 - c.a2 * dcGain c
      ring

theorem sectionZi_scaled_equilibrium (c : Section) (h : 1 + c.a1 + c.a2 ≠ 0)
    (a : ℝ) : sectionStep c (a, a • sectionZi c) =
      (a * dcGain c, a • sectionZi c) := by
  have he : (a, a • sectionZi c) = a • (1, sectionZi c) := by simp
  rw [he, map_smul, sectionZi_equilibrium c h]
  rfl

/-- Feed one sample through all sections, retaining every updated section state. -/
def cascade : (s : ℕ) → (Fin s → Section) →
    (ℝ × State s) →ₗ[ℝ] (ℝ × State s)
  | 0, _ => LinearMap.id
  | s + 1, c =>
    { toFun := fun p =>
        let h := sectionStep (c 0) (p.1, p.2 0)
        let t := cascade s (fun i => c i.succ) (h.1, fun i => p.2 i.succ)
        (t.1, Fin.cons h.2 t.2)
      map_add' := by
        intro p q
        let a := sectionStep (c 0) (p.1, p.2 0)
        let b := sectionStep (c 0) (q.1, q.2 0)
        let u : ℝ × State s := (a.1, fun i => p.2 i.succ)
        let v : ℝ × State s := (b.1, fun i => q.2 i.succ)
        have hh : sectionStep (c 0) ((p + q).1, (p + q).2 0) = a + b :=
          (sectionStep (c 0)).map_add (p.1, p.2 0) (q.1, q.2 0)
        change (_, _) = (_, _)
        rw [hh]
        change (((cascade s (fun i => c i.succ)) (u + v)).1,
          @Fin.cons s (fun _ => Fin 2 → ℝ) (a.2 + b.2)
            ((cascade s (fun i => c i.succ)) (u + v)).2) = _
        simp only [map_add, Prod.fst_add, Prod.snd_add]
        apply Prod.ext
        · rfl
        · funext i
          refine Fin.cases ?_ (fun j => ?_) i <;> rfl
      map_smul' := by
        intro r p
        let a := sectionStep (c 0) (p.1, p.2 0)
        let u : ℝ × State s := (a.1, fun i => p.2 i.succ)
        have hh : sectionStep (c 0) ((r • p).1, (r • p).2 0) = r • a :=
          (sectionStep (c 0)).map_smul r (p.1, p.2 0)
        change (_, _) = (_, _)
        rw [hh]
        change (((cascade s (fun i => c i.succ)) (r • u)).1,
          @Fin.cons s (fun _ => Fin 2 → ℝ) (r • a.2)
            ((cascade s (fun i => c i.succ)) (r • u)).2) = _
        simp only [map_smul]
        apply Prod.ext
        · rfl
        · funext i
          refine Fin.cases ?_ (fun j => ?_) i <;> rfl }

def cascadeGain : (s : ℕ) → (Fin s → Section) → ℝ
  | 0, _ => 1
  | s + 1, c => dcGain (c 0) * cascadeGain s (fun i => c i.succ)

/-- `sosfilt_zi` scales each section's unit equilibrium by the DC response of
all preceding sections. This recursion fixes section/state order as well. -/
def cascadeZi : (s : ℕ) → (Fin s → Section) → State s
  | 0, _ => 0
  | s + 1, c => Fin.cons (sectionZi (c 0))
      (dcGain (c 0) • cascadeZi s (fun i => c i.succ))

theorem cascadeZi_equilibrium : ∀ (s : ℕ) (c : Fin s → Section),
    (∀ i, 1 + (c i).a1 + (c i).a2 ≠ 0) →
    cascade s c (1, cascadeZi s c) = (cascadeGain s c, cascadeZi s c)
  | 0, c, _ => rfl
  | s + 1, c, h => by
    have ht := cascadeZi_equilibrium s (fun i => c i.succ) (fun i => h i.succ)
    have hseed :
        (dcGain (c 0), dcGain (c 0) • cascadeZi s (fun i => c i.succ)) =
          dcGain (c 0) • (1, cascadeZi s (fun i => c i.succ)) := by simp
    have hs : cascade s (fun i => c i.succ)
        (dcGain (c 0), dcGain (c 0) • cascadeZi s (fun i => c i.succ)) =
        (dcGain (c 0) * cascadeGain s (fun i => c i.succ),
          dcGain (c 0) • cascadeZi s (fun i => c i.succ)) := by
      rw [hseed, map_smul, ht]
      rfl
    change ((cascade s (fun i => c i.succ)
        ((sectionStep (c 0) (1, sectionZi (c 0))).1,
          dcGain (c 0) • cascadeZi s (fun i => c i.succ))).1,
      @Fin.cons s (fun _ => Fin 2 → ℝ)
        (sectionStep (c 0) (1, sectionZi (c 0))).2
        (cascade s (fun i => c i.succ)
          ((sectionStep (c 0) (1, sectionZi (c 0))).1,
            dcGain (c 0) • cascadeZi s (fun i => c i.succ))).2) = _
    rw [sectionZi_equilibrium (c 0) (h 0)]
    change ((cascade s (fun i => c i.succ)
        (dcGain (c 0), dcGain (c 0) • cascadeZi s (fun i => c i.succ))).1,
      @Fin.cons s (fun _ => Fin 2 → ℝ) (sectionZi (c 0)) ((cascade s (fun i => c i.succ)
        (dcGain (c 0), dcGain (c 0) • cascadeZi s (fun i => c i.succ))).2)) = _
    rw [hs]
    rfl

/-- Process the finite time grid in order; carries state between samples. -/
def run {s : ℕ} (c : Fin s → Section) : (n : ℕ) →
    (Samples n × State s) →ₗ[ℝ] (Samples n × State s)
  | 0 => LinearMap.id
  | n + 1 =>
    { toFun := fun p =>
        let h := cascade s c (p.1 0, p.2)
        let t := run c n (fun i => p.1 i.succ, h.2)
        (Fin.cons h.1 t.1, t.2)
      map_add' := by
        intro p q
        let a := cascade s c (p.1 0, p.2)
        let b := cascade s c (q.1 0, q.2)
        let u : Samples n × State s := (fun i => p.1 i.succ, a.2)
        let v : Samples n × State s := (fun i => q.1 i.succ, b.2)
        have hh : cascade s c ((p + q).1 0, (p + q).2) = a + b :=
          (cascade s c).map_add (p.1 0, p.2) (q.1 0, q.2)
        change (_, _) = (_, _)
        rw [hh]
        change (@Fin.cons n (fun _ => ℝ) (a.1 + b.1) (run c n (u + v)).1,
          (run c n (u + v)).2) = _
        simp only [map_add, Prod.fst_add, Prod.snd_add]
        apply Prod.ext
        · funext i
          refine Fin.cases ?_ (fun j => ?_) i <;> rfl
        · rfl
      map_smul' := by
        intro r p
        let a := cascade s c (p.1 0, p.2)
        let u : Samples n × State s := (fun i => p.1 i.succ, a.2)
        have hh : cascade s c ((r • p).1 0, (r • p).2) = r • a :=
          (cascade s c).map_smul r (p.1 0, p.2)
        change (_, _) = (_, _)
        rw [hh]
        change (@Fin.cons n (fun _ => ℝ) (r • a.1) (run c n (r • u)).1,
          (run c n (r • u)).2) = _
        simp only [map_smul]
        apply Prod.ext
        · funext i
          refine Fin.cases ?_ (fun j => ?_) i <;> rfl
        · rfl }

/-- Zero-state sample response H. -/
def response {s : ℕ} (c : Fin s → Section) (n : ℕ) : Samples n →ₗ[ℝ] Samples n :=
  (LinearMap.fst ℝ _ _).comp ((run c n).comp (LinearMap.inl ℝ _ _))

/-- State-only sample response O, in exact section/state coordinate order. -/
def stateResponse {s : ℕ} (c : Fin s → Section) (n : ℕ) : State s →ₗ[ℝ] Samples n :=
  (LinearMap.fst ℝ _ _).comp ((run c n).comp (LinearMap.inr ℝ _ _))

theorem run_superposition {s n : ℕ} (c : Fin s → Section)
    (x : Samples n) (z : State s) :
    (run c n (x, z)).1 = response c n x + stateResponse c n z := by
  have h := (run c n).map_add (x, 0) (0, z)
  have hp : ((x, 0) + (0, z) : Samples n × State s) = (x, z) := by simp
  rw [hp] at h
  exact congrArg Prod.fst h

def reverse (n : ℕ) : Samples n →ₗ[ℝ] Samples n where
  toFun x i := x i.rev
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

theorem reverse_involution {n : ℕ} (x : Samples n) : reverse n (reverse n x) = x := by
  funext i
  simp [reverse]

/-- Execute core forward/reverse passes from two specified effective states. -/
def twoPass {s n : ℕ} (c : Fin s → Section) (x : Samples n)
    (left right : State s) : Samples n :=
  reverse n (run c n (reverse n (run c n (x, left)).1, right)).1

def zeroTwoPass {s : ℕ} (c : Fin s → Section) (n : ℕ) :
    Samples n →ₗ[ℝ] Samples n :=
  (reverse n).comp ((response c n).comp ((reverse n).comp (response c n)))

def endpointResponse {s : ℕ} (c : Fin s → Section) (n : ℕ) :
    (State s × State s) →ₗ[ℝ] Samples n :=
  ((reverse n).comp ((response c n).comp ((reverse n).comp
    ((stateResponse c n).comp (LinearMap.fst ℝ _ _))))) +
  ((reverse n).comp ((stateResponse c n).comp (LinearMap.snd ℝ _ _)))

/-- Both effective states are retained; the reverse state is not set to zero. -/
theorem twoPass_decomposition {s n : ℕ} (c : Fin s → Section) (x : Samples n)
    (left right : State s) :
    twoPass c x left right = zeroTwoPass c n x + endpointResponse c n (left, right) := by
  unfold twoPass
  rw [run_superposition, run_superposition]
  simp only [map_add]
  change (zeroTwoPass c n x +
    reverse n (response c n (reverse n (stateResponse c n left)))) +
      reverse n (stateResponse c n right) = _
  exact add_assoc _ _ _

/-- A fixed `sosfilt_zi` table is scaled by the first value, not newly sampled. -/
def initState {s : ℕ} (zi : State s) : ℝ →ₗ[ℝ] State s where
  toFun a := a • zi
  map_add' a b := add_smul a b zi
  map_smul' a b := by simp [mul_smul]

def sampleAt {n : ℕ} (i : Fin n) : Samples n →ₗ[ℝ] ℝ where
  toFun x := x i
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

/-- `2*x[0]-x[1:pad+1][::-1]`; `p ≤ n` is exactly limited reflection. -/
def leftPad (n p : ℕ) (hp : p ≤ n) : Samples (n + 1) →ₗ[ℝ] Samples p where
  toFun x i := 2 * x 0 - x ⟨p - i.val, by omega⟩
  map_add' x y := by funext i; simp; ring
  map_smul' a x := by funext i; simp; ring

/-- `2*x[-1]-x[-pad-1:-1][::-1]`, including the empty pad. -/
def rightPad (n p : ℕ) (hp : p ≤ n) : Samples (n + 1) →ₗ[ℝ] Samples p where
  toFun x i := 2 * x (Fin.last n) - x ⟨n - 1 - i.val, by omega⟩
  map_add' x y := by funext i; simp; ring
  map_smul' a x := by funext i; simp; ring

def leftSeed (n p : ℕ) (hp : p ≤ n) : Samples (n + 1) →ₗ[ℝ] ℝ :=
  if h : p = 0 then sampleAt 0 else
    (sampleAt ⟨0, by omega⟩).comp (leftPad n p hp)

/-- Effective left state after the left pad and before the first core sample. -/
def leftState {s : ℕ} (c : Fin s → Section) (zi : State s)
    (n p : ℕ) (hp : p ≤ n) : Samples (n + 1) →ₗ[ℝ] State s :=
  (LinearMap.snd ℝ _ _).comp ((run c p).comp
    ((leftPad n p hp).prod ((initState zi).comp (leftSeed n p hp))))

def coreRun {s : ℕ} (c : Fin s → Section) (zi : State s)
    (n p : ℕ) (hp : p ≤ n) :
    Samples (n + 1) →ₗ[ℝ] (Samples (n + 1) × State s) :=
  (run c (n + 1)).comp ((LinearMap.id).prod (leftState c zi n p hp))

/-- Filter the reflected right pad from the terminal forward core state. -/
def tailRun {s : ℕ} (c : Fin s → Section) (zi : State s)
    (n p : ℕ) (hp : p ≤ n) :
    Samples (n + 1) →ₗ[ℝ] (Samples p × State s) :=
  (run c p).comp ((rightPad n p hp).prod
    ((LinearMap.snd ℝ _ _).comp (coreRun c zi n p hp)))

/-- Effective right state: initialize from the final filtered tail value, then
filter the reversed filtered tail. The zero-pad branch uses forward[-1]. -/
def rightState {s : ℕ} (c : Fin s → Section) (zi : State s)
    (n p : ℕ) (hp : p ≤ n) : Samples (n + 1) →ₗ[ℝ] State s :=
  if h : p = 0 then
    (initState zi).comp ((sampleAt (Fin.last n)).comp
      ((LinearMap.fst ℝ _ _).comp (coreRun c zi n p hp)))
  else
    let tail := (LinearMap.fst ℝ _ _).comp (tailRun c zi n p hp)
    (LinearMap.snd ℝ _ _).comp ((run c p).comp
      (((reverse p).comp tail).prod
        ((initState zi).comp ((sampleAt ⟨p - 1, by omega⟩).comp tail))))

/-- Actual finite observed-odd-reflection algorithm, for every legal pad length.
Each call constructs both endpoint states solely from this segment. -/
def paddedFilter {s : ℕ} (c : Fin s → Section) (zi : State s)
    (n p : ℕ) (hp : p ≤ n) : Samples (n + 1) →ₗ[ℝ] Samples (n + 1) :=
  (reverse (n + 1)).comp ((LinearMap.fst ℝ _ _).comp ((run c (n + 1)).comp
    (((reverse (n + 1)).comp ((LinearMap.fst ℝ _ _).comp
      (coreRun c zi n p hp))).prod (rightState c zi n p hp))))

theorem paddedFilter_add {s n p : ℕ} (c : Fin s → Section) (zi : State s)
    (hp : p ≤ n) (x y : Samples (n + 1)) :
    paddedFilter c zi n p hp (x + y) =
      paddedFilter c zi n p hp x + paddedFilter c zi n p hp y :=
  (paddedFilter c zi n p hp).map_add x y

theorem paddedFilter_smul {s n p : ℕ} (c : Fin s → Section) (zi : State s)
    (hp : p ≤ n) (a : ℝ) (x : Samples (n + 1)) :
    paddedFilter c zi n p hp (a • x) = a • paddedFilter c zi n p hp x :=
  (paddedFilter c zi n p hp).map_smul a x

/-- F(x)=Qx+B*a(x), with the actual two deterministic endpoint states. -/
theorem paddedFilter_decomposition {s n p : ℕ} (c : Fin s → Section) (zi : State s)
    (hp : p ≤ n) (x : Samples (n + 1)) :
    paddedFilter c zi n p hp x = zeroTwoPass c (n + 1) x +
      endpointResponse c (n + 1)
        (leftState c zi n p hp x, rightState c zi n p hp x) :=
  twoPass_decomposition c x _ _

theorem zeroPad_leftState {s n : ℕ} (c : Fin s → Section) (zi : State s)
    (x : Samples (n + 1)) :
    leftState c zi n 0 (Nat.zero_le n) x = x 0 • zi := by
  rfl

theorem zeroPad_rightState {s n : ℕ} (c : Fin s → Section) (zi : State s)
    (x : Samples (n + 1)) :
    rightState c zi n 0 (Nat.zero_le n) x =
      (run c (n + 1) (x, x 0 • zi)).1 (Fin.last n) • zi := by
  rfl

/-- A one-sample segment admits only pad=0; no out-of-range reflection occurs. -/
theorem oneSample_pad (limit : ℕ) : min limit (1 - 1) = 0 := by simp

theorem oneSample_filter {s : ℕ} (c : Fin s → Section) (zi : State s)
    (x : Samples 1) :
    paddedFilter c zi 0 0 (Nat.zero_le 0) x =
      twoPass c x (x 0 • zi) ((run c 1 (x, x 0 • zi)).1 0 • zi) := by
  rfl

/-- Standard section/state flattened coordinates for one channel. -/
abbrev EndpointCoordinate (s : ℕ) := (Fin s × Fin 2) ⊕ (Fin s × Fin 2)

def packEndpoints {s : ℕ} (z : State s × State s) : EndpointCoordinate s → ℝ :=
  Sum.elim (fun j => z.1 j.1 j.2) (fun j => z.2 j.1 j.2)

def unpackEndpoints (s : ℕ) :
    (EndpointCoordinate s → ℝ) →ₗ[ℝ] (State s × State s) where
  toFun a := (fun j k => a (.inl (j, k)), fun j k => a (.inr (j, k)))
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

theorem unpack_pack {s : ℕ} (z : State s × State s) :
    unpackEndpoints s (packEndpoints z) = z := rfl

/-- Exactly 4*S basis columns; this theorem makes no rank assertion. -/
theorem endpoint_coordinate_count (s : ℕ) :
    Fintype.card (EndpointCoordinate s) = 4 * s := by
  simp only [EndpointCoordinate, Fintype.card_sum, Fintype.card_prod, Fintype.card_fin]
  omega

/-- Flattening is first all left sections, then all right sections, with each
section's state 0 before state 1: NumPy transpose(0,2,1).reshape(...). -/
def flatCoordinate (s : ℕ) : EndpointCoordinate s ≃ Fin (s * 2 + s * 2) :=
  (Equiv.sumCongr finProdFinEquiv finProdFinEquiv).trans finSumFinEquiv

theorem flatCoordinate_left {s : ℕ} (j : Fin s) (k : Fin 2) :
    (flatCoordinate s (.inl (j, k))).val = 2 * j.val + k.val := by
  simp [flatCoordinate, finProdFinEquiv]
  omega

theorem flatCoordinate_right {s : ℕ} (j : Fin s) (k : Fin 2) :
    (flatCoordinate s (.inr (j, k))).val = 2 * s + 2 * j.val + k.val := by
  simp [flatCoordinate, finProdFinEquiv]
  omega

def basisMatrix {s : ℕ} (c : Fin s → Section) (n : ℕ) :
    Matrix (Fin n) (EndpointCoordinate s) ℝ :=
  LinearMap.toMatrix' ((endpointResponse c n).comp (unpackEndpoints s))

theorem basisMatrix_mulVec {s n : ℕ} (c : Fin s → Section) (z : State s × State s) :
    (basisMatrix c n).mulVec (packEndpoints z) = endpointResponse c n z := by
  rw [basisMatrix, LinearMap.toMatrix'_mulVec]
  rfl

/-- The dense basis relation uses the same effective states as execution. -/
theorem paddedFilter_matrix_decomposition {s n p : ℕ} (c : Fin s → Section)
    (zi : State s) (hp : p ≤ n) (x : Samples (n + 1)) :
    paddedFilter c zi n p hp x = zeroTwoPass c (n + 1) x +
      (basisMatrix c (n + 1)).mulVec
        (packEndpoints (leftState c zi n p hp x, rightState c zi n p hp x)) := by
  rw [basisMatrix_mulVec, paddedFilter_decomposition]

/-- Padding endpoint states contain no additive independent noise realization. -/
theorem zeroInput_endpoints {s n p : ℕ} (c : Fin s → Section) (zi : State s)
    (hp : p ≤ n) :
    (leftState c zi n p hp 0, rightState c zi n p hp 0) = (0, 0) := by
  simp only [map_zero]

/-- Filtering segments separately reinitializes them separately. A preceding or
following segment cannot affect a segment whose own samples are unchanged. -/
def resetSegments {s : ℕ} {ι : Type*} (c : Fin s → Section) (zi : State s)
    (n limit : ι → ℕ) (x : (i : ι) → Samples (n i + 1)) :
    (i : ι) → Samples (n i + 1) :=
  fun i => paddedFilter c zi (n i) (min (limit i) (n i)) (Nat.min_le_right _ _) (x i)

theorem reset_segment_noninterference {s : ℕ} {ι : Type*}
    (c : Fin s → Section) (zi : State s) (n limit : ι → ℕ)
    (x y : (i : ι) → Samples (n i + 1)) (i : ι) (h : x i = y i) :
    resetSegments c zi n limit x i = resetSegments c zi n limit y i := by
  unfold resetSegments
  rw [h]

theorem oneSample_reset {s : ℕ} (c : Fin s → Section) (zi : State s)
    (limit : ℕ) (x : Samples 1) :
    paddedFilter c zi 0 (min limit 0) (Nat.min_le_right _ _) x =
      paddedFilter c zi 0 0 (Nat.zero_le 0) x := by
  simp

/-- Even a nonzero endpoint coordinate can be invisible on a one-sample grid.
Consequently 4*S columns must never be described as rank 4*S. -/
theorem oneSample_endpoint_kernel (c : Fin 1 → Section) :
    ∃ z : State 1 × State 1, z ≠ 0 ∧ endpointResponse c 1 z = 0 := by
  let z : State 1 × State 1 := ((fun _ => ![0, 1]), 0)
  refine ⟨z, ?_, ?_⟩
  · intro hz
    have hz1 := congrArg (fun w : State 1 × State 1 => w.1 0 1) hz
    norm_num [z] at hz1
  · funext i
    fin_cases i
    simp [z, endpointResponse, response, stateResponse, run, cascade,
      sectionStep, reverse]

end

end N2P3.FiniteSOS
