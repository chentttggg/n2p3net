import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum
import Lean.Elab.Tactic.Omega

/-!
# Source loops for the three TemporalNoise whitening modes

Source: temporal_noise.py:99-136,347-373. Signals are total coordinate
representatives of a FINITE parsed samples-by-columns array. Only indices in
the explicit output windows are observations. The source domain requires a
positive column count, n >= max(1,p) for stationary/lattice, and n >= p+1 for
conditional prefix. Natural subtraction is used only at the same valid lag
coordinates as the source; invalid public input arrays are outside this layer.

The stationary prefix uses explicit lower triangular forward substitution,
not an arbitrary supplied whitening function. Its root must be bound by the
constructor to the unit-innovation Cholesky factor; the algebra below also
states the nonzero diagonal condition for the triangular equations. It does
not prove a SciPy Cholesky or Lyapunov solver correct.

The lattice is the actual forward/backward stage loop: forward shifts left,
backward retains its coordinate and loses its last row through the decreasing
remaining length. Prefix rows use exp(-logVariance/2); final rows use the last
next_forward directly. Log variances are explicit suffix sums from the actual
reflections. No opaque lattice map or whitening/stability identity is a field.
Strict |k|<1 gives positive 1-k^2; the source's stricter numerical margin,
finite exponentials/condition diagnostics and fitted PARCOR provenance remain
constructor/refinement obligations. In particular source allclose between AR
coefficients and expanded PARCOR coefficients is not exact equality.

All arithmetic is exact real, with no IEEE/NumPy array refinement, overflow,
fit or physical prefix-authority conclusion. Diagnostic innovation variance
is accepted as an argument but never divides the observations a second time.
-/

namespace N2P3.TemporalWhitening

open scoped BigOperators

noncomputable section

abbrev Signal (C : Type*) := ℕ → C → ℝ
abbrev Rows (C : Type*) := List (C → ℝ)

variable {C : Type*}

def shift (offset : ℕ) (x : Signal C) : Signal C := fun t c => x (offset + t) c

def combine (a b : ℝ) (x y : Signal C) : Signal C :=
  fun t c => a * x t c + b * y t c

def window : ℕ → Signal C → Rows C
  | 0, _ => []
  | n + 1, x => x 0 :: window n (shift 1 x)

def rowAt : Rows C → ℕ → C → ℝ
  | [], _, _ => 0
  | x :: _, 0, c => x c
  | _ :: tail, t + 1, c => rowAt tail t c

theorem window_length (n : ℕ) (x : Signal C) : (window n x).length = n := by
  induction n generalizing x with
  | zero => rfl
  | succ n ih => simp [window, ih]

theorem rowAt_window (n i : ℕ) (x : Signal C) (c : C) :
    rowAt (window n x) i c = if i < n then x i c else 0 := by
  induction n generalizing i x with
  | zero => simp [window, rowAt]
  | succ n ih =>
      cases i with
      | zero => simp [window, rowAt]
      | succ i => simp [window, rowAt, ih, shift, Nat.add_comm]

theorem rowAt_append_left (xs ys : Rows C) (i : ℕ) (c : C) (h : i < xs.length) :
    rowAt (xs ++ ys) i c = rowAt xs i c := by
  induction xs generalizing i with
  | nil => simp at h
  | cons x tail ih =>
      cases i with
      | zero => rfl
      | succ i => exact ih i (by simpa using h)

theorem rowAt_append_right (xs ys : Rows C) (i : ℕ) (c : C) :
    rowAt (xs ++ ys) (xs.length + i) c = rowAt ys i c := by
  induction xs with
  | nil => simp
  | cons x tail ih =>
      simp only [List.cons_append, List.length_cons]
      rw [show tail.length+1+i = (tail.length+i)+1 by omega]
      exact ih

/-- Each source lag subtracts from the same retained result slice. -/
def lagLoop : List ℝ → ℕ → ℕ → Signal C → Signal C → Signal C
  | [], _, _, _, result => result
  | coefficient :: rest, lag, p, x, result =>
      lagLoop rest (lag + 1) p x
        (fun i c => result i c - coefficient * x (p + i - lag) c)

def lagTerms : List ℝ → ℕ → ℕ → Signal C → ℕ → C → List ℝ
  | [], _, _, _, _, _ => []
  | coefficient :: rest, lag, p, x, i, c =>
      coefficient * x (p + i - lag) c :: lagTerms rest (lag + 1) p x i c

theorem lagLoop_subtracts_each_lag (coefficients : List ℝ) (lag p i : ℕ)
    (x result : Signal C) (c : C) :
    lagLoop coefficients lag p x result i c =
      result i c - (lagTerms coefficients lag p x i c).sum := by
  induction coefficients generalizing lag result with
  | nil => simp [lagLoop, lagTerms]
  | cons k tail ih => simp [lagLoop, lagTerms, ih]; ring

def arTail (coefficients : List ℝ) (x : Signal C) : Signal C :=
  lagLoop coefficients 1 coefficients.length x (shift coefficients.length x)

def conditional (coefficients : List ℝ) (n : ℕ) (x : Signal C) : Rows C :=
  window (n - coefficients.length) (arTail coefficients x)

theorem conditional_output_length (coefficients : List ℝ) (n : ℕ) (x : Signal C) :
    (conditional coefficients n x).length = n - coefficients.length :=
  window_length _ _

theorem conditional_coordinate (coefficients : List ℝ) (n i : ℕ) (x : Signal C)
    (c : C) (hi : i < n - coefficients.length) :
    rowAt (conditional coefficients n x) i c = x (coefficients.length + i) c -
      (lagTerms coefficients 1 coefficients.length x i c).sum := by
  rw [conditional, rowAt_window, if_pos hi, arTail, lagLoop_subtracts_each_lag]
  rfl

theorem arTail_nil (x : Signal C) : arTail [] x = x := by
  funext i c
  simp [arTail, lagLoop, shift]

theorem arTail_one (a : ℝ) (x : Signal C) (i : ℕ) (c : C) :
    arTail [a] x i c = x (i + 1) c - a * x i c := by
  simp [arTail, lagLoop, shift, Nat.add_comm]

theorem lagLoop_combine (coefficients : List ℝ) (lag p : ℕ) (a b : ℝ)
    (x y u v : Signal C) :
    lagLoop coefficients lag p (combine a b x y) (combine a b u v) =
      combine a b (lagLoop coefficients lag p x u) (lagLoop coefficients lag p y v) := by
  induction coefficients generalizing lag u v with
  | nil => rfl
  | cons k tail ih =>
      have he : (fun i c => combine a b u v i c - k * combine a b x y (p+i-lag) c) =
          combine a b (fun i c => u i c - k*x (p+i-lag) c)
            (fun i c => v i c - k*y (p+i-lag) c) := by
        funext i c
        simp only [combine]
        ring
      simpa only [lagLoop, he] using ih (lag+1)
        (fun i c => u i c-k*x (p+i-lag) c) (fun i c => v i c-k*y (p+i-lag) c)

theorem arTail_combine (coefficients : List ℝ) (a b : ℝ) (x y : Signal C) :
    arTail coefficients (combine a b x y) =
      combine a b (arTail coefficients x) (arTail coefficients y) := by
  exact lagLoop_combine coefficients 1 coefficients.length a b x y
    (shift coefficients.length x) (shift coefficients.length y)

theorem lagTerms_causal (coefficients : List ℝ) (lag p i : ℕ) (x y : Signal C) (c : C)
    (h : ∀ t ≤ p+i, x t c = y t c) :
    lagTerms coefficients lag p x i c = lagTerms coefficients lag p y i c := by
  induction coefficients generalizing lag with
  | nil => rfl
  | cons k tail ih => simp [lagTerms, h _ (Nat.sub_le _ _), ih]

theorem arTail_causal (coefficients : List ℝ) (i : ℕ) (x y : Signal C) (c : C)
    (h : ∀ t ≤ coefficients.length+i, x t c = y t c) :
    arTail coefficients x i c = arTail coefficients y i c := by
  simp only [arTail, lagLoop_subtracts_each_lag, shift]
  rw [h _ (le_refl _), lagTerms_causal _ _ _ _ _ _ _ h]

/-- Exact lower-triangular forward substitution, one solved row at a time. -/
def prefixSolve (root : ℕ → ℕ → ℝ) (x : Signal C) (t : ℕ) (c : C) : ℝ :=
  (x t c - ∑ j : Fin t, root t j.val * prefixSolve root x j.val c) / root t t
termination_by t
decreasing_by exact j.isLt

theorem prefixSolve_zero (root : ℕ → ℕ → ℝ) (x : Signal C) (c : C) :
    prefixSolve root x 0 c = x 0 c / root 0 0 := by
  rw [prefixSolve]
  simp

theorem prefixSolve_triangular_equation (root : ℕ → ℕ → ℝ) (x : Signal C)
    (t : ℕ) (c : C) (h : root t t ≠ 0) :
    root t t * prefixSolve root x t c +
      (∑ j : Fin t, root t j.val * prefixSolve root x j.val c) = x t c := by
  rw [prefixSolve]
  have he := mul_div_cancel₀
    (x t c - ∑ j : Fin t, root t j.val * prefixSolve root x j.val c) h
  nlinarith

theorem prefixSolve_causal (root : ℕ → ℕ → ℝ) (x y : Signal C) (t : ℕ) (c : C)
    (h : ∀ i ≤ t, x i c = y i c) : prefixSolve root x t c = prefixSolve root y t c := by
  induction t using Nat.strong_induction_on with
  | h t ih =>
      rw [prefixSolve, prefixSolve, h t (le_refl _)]
      congr 2
      apply Finset.sum_congr rfl
      intro j _
      rw [ih j.val j.isLt (fun i hi => h i (by omega))]

theorem prefixSolve_combine (root : ℕ → ℕ → ℝ) (a b : ℝ) (x y : Signal C)
    (t : ℕ) (c : C) : prefixSolve root (combine a b x y) t c =
      a * prefixSolve root x t c + b * prefixSolve root y t c := by
  induction t using Nat.strong_induction_on with
  | h t ih =>
      rw [prefixSolve, prefixSolve, prefixSolve]
      have hs : (∑ j : Fin t, root t j.val * prefixSolve root (combine a b x y) j.val c) =
          a * (∑ j : Fin t, root t j.val * prefixSolve root x j.val c) +
          b * (∑ j : Fin t, root t j.val * prefixSolve root y j.val c) := by
        rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
        apply Finset.sum_congr rfl
        intro j _
        rw [ih j.val j.isLt]
        ring
      rw [hs]
      simp only [combine]
      ring

def stationary (coefficients : List ℝ) (root : ℕ → ℕ → ℝ) (n : ℕ)
    (x : Signal C) : Rows C :=
  window coefficients.length (prefixSolve root x) ++ conditional coefficients n x

theorem stationary_output_length (coefficients : List ℝ) (root : ℕ → ℕ → ℝ)
    (n : ℕ) (x : Signal C) (h : coefficients.length ≤ n) :
    (stationary coefficients root n x).length = n := by
  simp only [stationary, List.length_append, window_length, conditional_output_length]
  omega

theorem stationary_tail_is_conditional (coefficients : List ℝ) (root : ℕ → ℕ → ℝ)
    (n : ℕ) (x : Signal C) :
    (stationary coefficients root n x).drop coefficients.length = conditional coefficients n x := by
  exact List.drop_left' (window_length coefficients.length (prefixSolve root x))

theorem stationary_prefix_coordinate (coefficients : List ℝ) (root : ℕ → ℕ → ℝ)
    (n i : ℕ) (x : Signal C) (c : C) (h : i < coefficients.length) :
    rowAt (stationary coefficients root n x) i c = prefixSolve root x i c := by
  rw [stationary, rowAt_append_left _ _ _ _ (by simpa [window_length] using h), rowAt_window,
    if_pos h]

theorem stationary_tail_coordinate (coefficients : List ℝ) (root : ℕ → ℕ → ℝ)
    (n i : ℕ) (x : Signal C) (c : C) :
    rowAt (stationary coefficients root n x) (coefficients.length+i) c =
      rowAt (conditional coefficients n x) i c := by
  simpa only [window_length] using rowAt_append_right
    (window coefficients.length (prefixSolve root x)) (conditional coefficients n x) i c

def logVariance (reflections : List ℝ) : ℝ :=
  (reflections.map (fun k => -Real.log (1-k^2))).sum

def logVarianceAt (reflections : List ℝ) (t : ℕ) : ℝ := logVariance (reflections.drop t)

def prefixScale (reflections : List ℝ) (t : ℕ) : ℝ :=
  Real.exp (-logVarianceAt reflections t / 2)

theorem logVariance_nil : logVariance [] = 0 := by simp [logVariance]

theorem logVariance_cons (k : ℝ) (tail : List ℝ) :
    logVariance (k::tail) = -Real.log (1-k^2) + logVariance tail := by simp [logVariance]

theorem prefixScale_positive (reflections : List ℝ) (t : ℕ) : 0 < prefixScale reflections t :=
  Real.exp_pos _

theorem strict_reflection_has_positive_variance_factor (k : ℝ) (h : |k| < 1) :
    0 < 1-k^2 := by
  have ha := abs_lt.mp h
  nlinarith [sq_nonneg k]

theorem innovation_variance_positive (reflections : List ℝ) (t : ℕ) :
    0 < Real.exp (logVarianceAt reflections t) := Real.exp_pos _

/-- Reverse cumulative sum of -log(1-k^2), evaluated from the right. -/
def suffixLogs : List ℝ → List ℝ
  | [] => []
  | k::tail => (-Real.log (1-k^2) + (suffixLogs tail).headD 0) :: suffixLogs tail

theorem suffixLogs_length (ks : List ℝ) : (suffixLogs ks).length = ks.length := by
  induction ks with
  | nil => rfl
  | cons k tail ih => simp [suffixLogs, ih]

theorem suffixLogs_head (ks : List ℝ) : (suffixLogs ks).headD 0 = logVariance ks := by
  induction ks with
  | nil => simp [suffixLogs, logVariance]
  | cons k tail ih => simp only [suffixLogs, List.headD_cons, logVariance_cons, ih]

theorem suffixLogs_lookup (ks : List ℝ) (t : ℕ) :
    (suffixLogs ks).getD t 0 = logVarianceAt ks t := by
  induction ks generalizing t with
  | nil => simp [suffixLogs, logVarianceAt, logVariance]
  | cons k tail ih =>
      cases t with
      | zero => simpa [logVarianceAt] using suffixLogs_head (k::tail)
      | succ t => simpa [suffixLogs, logVarianceAt] using ih t

def varianceProduct : List ℝ → ℝ
  | [] => 1
  | k::tail => (1-k^2)⁻¹ * varianceProduct tail

theorem innovation_variance_is_actual_parcor_product (ks : List ℝ)
    (h : ∀ k ∈ ks, |k| < 1) : Real.exp (logVariance ks) = varianceProduct ks := by
  induction ks with
  | nil => simp [logVariance_nil, varianceProduct]
  | cons k tail ih =>
      have hk := strict_reflection_has_positive_variance_factor k (h k (by simp))
      have ht : ∀ r ∈ tail, |r| < 1 := fun r hr => h r (by simp [hr])
      rw [logVariance_cons, Real.exp_add, Real.exp_neg, Real.exp_log hk, ih ht]
      rfl

def forward (k : ℝ) (f b : Signal C) : Signal C := fun t c => f t c + k*b t c
def backward (k : ℝ) (f b : Signal C) : Signal C := fun t c => b t c + k*f t c

/-- `remaining` is the exact current forward/backward array length.
Dropping next_backward's last row changes this bound without shifting it. -/
def latticeLoop (allReflections : List ℝ) : List ℝ → ℕ → ℕ → Signal C → Signal C → Rows C
  | [], _, _, _, _ => []
  | k::rest, stage, remaining, f, b =>
      let nf := forward k f b
      let nb := backward k f b
      match rest with
      | [] => window remaining nf
      | next::tail =>
          (fun c => nf 0 c * prefixScale allReflections stage) ::
          latticeLoop allReflections (next::tail) (stage+1) (remaining-1) (shift 1 nf) nb

def lattice (reflections : List ℝ) (n : ℕ) (x : Signal C) : Rows C :=
  match reflections with
  | [] => window n x
  | k::tail => (fun c => x 0 c * prefixScale (k::tail) 0) ::
      latticeLoop (k::tail) (k::tail) 1 (n-1) (shift 1 x) x

theorem latticeLoop_single (all : List ℝ) (k : ℝ) (stage remaining : ℕ) (f b : Signal C) :
    latticeLoop all [k] stage remaining f b = window remaining (forward k f b) := rfl

theorem latticeLoop_prefix_step (all : List ℝ) (k next : ℝ) (tail : List ℝ)
    (stage remaining : ℕ) (f b : Signal C) :
    latticeLoop all (k::next::tail) stage remaining f b =
      (fun c => forward k f b 0 c * prefixScale all stage) ::
        latticeLoop all (next::tail) (stage+1) (remaining-1)
          (shift 1 (forward k f b)) (backward k f b) := rfl

theorem latticeLoop_length (all ks : List ℝ) (stage remaining : ℕ) (f b : Signal C)
    (hne : ks ≠ []) (h : ks.length ≤ remaining+1) :
    (latticeLoop all ks stage remaining f b).length = remaining := by
  induction ks generalizing stage remaining f b with
  | nil => contradiction
  | cons k rest ih =>
      cases rest with
      | nil => simp [latticeLoop, window_length]
      | cons next tail =>
          have hr : 0 < remaining := by simp at h; omega
          simp only [latticeLoop, List.length_cons]
          rw [ih (stage+1) (remaining-1) (shift 1 (forward k f b)) (backward k f b)
            (by simp) (by simp at h ⊢; omega)]
          omega

theorem lattice_output_length (ks : List ℝ) (n : ℕ) (x : Signal C)
    (h : ks.length ≤ n) : (lattice ks n x).length = n := by
  cases ks with
  | nil => exact window_length _ _
  | cons k tail =>
      have hn : 0 < n := by simp at h; omega
      simp only [lattice, List.length_cons]
      rw [latticeLoop_length _ _ _ _ _ _ (by simp) (by simp at h ⊢; omega)]
      omega

theorem forward_combine (k a b : ℝ) (f g u v : Signal C) :
    forward k (combine a b f g) (combine a b u v) =
      combine a b (forward k f u) (forward k g v) := by
  funext t c
  simp only [forward, combine]
  ring

theorem backward_combine (k a b : ℝ) (f g u v : Signal C) :
    backward k (combine a b f g) (combine a b u v) =
      combine a b (backward k f u) (backward k g v) := by
  funext t c
  simp only [backward, combine]
  ring

theorem shift_combine (offset : ℕ) (a b : ℝ) (x y : Signal C) :
    shift offset (combine a b x y) = combine a b (shift offset x) (shift offset y) := rfl

theorem latticeLoop_combine (all ks : List ℝ) (stage remaining i : ℕ) (a b : ℝ)
    (f g u v : Signal C) (c : C) :
    rowAt (latticeLoop all ks stage remaining (combine a b f g) (combine a b u v)) i c =
      a*rowAt (latticeLoop all ks stage remaining f u) i c +
      b*rowAt (latticeLoop all ks stage remaining g v) i c := by
  induction ks generalizing stage remaining i f g u v with
  | nil => simp [latticeLoop, rowAt]
  | cons k rest ih =>
      cases rest with
      | nil =>
          simp only [latticeLoop, forward_combine, rowAt_window]
          split_ifs <;> simp [combine]
      | cons next tail =>
          cases i with
          | zero => simp [latticeLoop, rowAt, forward, combine]; ring
          | succ i =>
              simp only [latticeLoop, rowAt, forward_combine, backward_combine, shift_combine]
              exact ih (stage+1) (remaining-1) i _ _ _ _

theorem lattice_combine (ks : List ℝ) (n i : ℕ) (a b : ℝ) (x y : Signal C) (c : C) :
    rowAt (lattice ks n (combine a b x y)) i c =
      a*rowAt (lattice ks n x) i c + b*rowAt (lattice ks n y) i c := by
  cases ks with
  | nil =>
      simp only [lattice, rowAt_window]
      split_ifs <;> simp [combine]
  | cons k tail =>
      cases i with
      | zero => simp [lattice, rowAt, combine]; ring
      | succ i =>
          simpa only [lattice, rowAt, shift_combine] using
            latticeLoop_combine (k::tail) (k::tail) 1 (n-1) i a b
              (shift 1 x) (shift 1 y) x y c

theorem latticeLoop_causal (all ks : List ℝ) (stage remaining i : ℕ)
    (f g u v : Signal C) (c : C)
    (hf : ∀ t ≤ i, f t c = g t c) (hb : ∀ t ≤ i, u t c = v t c) :
    rowAt (latticeLoop all ks stage remaining f u) i c =
      rowAt (latticeLoop all ks stage remaining g v) i c := by
  induction ks generalizing stage remaining i f g u v with
  | nil => rfl
  | cons k rest ih =>
      cases rest with
      | nil =>
          simp only [latticeLoop, rowAt_window]
          split_ifs <;> simp [forward, hf i (le_refl _), hb i (le_refl _)]
      | cons next tail =>
          cases i with
          | zero => simp [latticeLoop, rowAt, forward, hf 0 (by omega), hb 0 (by omega)]
          | succ i =>
              simp only [latticeLoop, rowAt]
              apply ih (stage+1) (remaining-1) i
              · intro t ht
                simp [shift, forward, hf (1+t) (by omega), hb (1+t) (by omega)]
              · intro t ht
                simp [backward, hf t (by omega), hb t (by omega)]

theorem lattice_causal (ks : List ℝ) (n i : ℕ) (x y : Signal C) (c : C)
    (h : ∀ t ≤ i, x t c = y t c) :
    rowAt (lattice ks n x) i c = rowAt (lattice ks n y) i c := by
  cases ks with
  | nil => simp only [lattice, rowAt_window]; split_ifs <;> simp [h i (le_refl _)]
  | cons k tail =>
      cases i with
      | zero => simp [lattice, rowAt, h 0 (le_refl _)]
      | succ i =>
          simp only [lattice, rowAt]
          apply latticeLoop_causal
          · intro t ht
            exact h (1+t) (by omega)
          · intro t ht
            exact h t (by omega)

theorem lattice_two_stage_tail (k l : ℝ) (n i : ℕ) (x : Signal C) (c : C)
    (hi : i < n-2) :
    rowAt (lattice [k,l] n x) (i+2) c =
      x (i+2) c + k*(1+l)*x (i+1) c + l*x i c := by
  simp only [lattice, latticeLoop, rowAt, rowAt_window]
  have he : n-1-1 = n-2 := by omega
  rw [he, if_pos hi]
  simp only [shift, forward, backward,
    show 1+i=i+1 by omega, show 1+(i+1)=i+2 by omega]
  ring

/-- Prefix rows and final buffer are both derived by the same actual stages. -/
def latticePrefix (all : List ℝ) : List ℝ → ℕ → Signal C → Signal C → Rows C
  | [], _, _, _ => []
  | k::rest, stage, f, b =>
      match rest with
      | [] => []
      | next::tail => (fun c => forward k f b 0 c * prefixScale all stage) ::
          latticePrefix all (next::tail) (stage+1) (shift 1 (forward k f b)) (backward k f b)

def finalForward : List ℝ → Signal C → Signal C → Signal C
  | [], f, _ => f
  | k::rest, f, b =>
      match rest with
      | [] => forward k f b
      | next::tail => finalForward (next::tail) (shift 1 (forward k f b)) (backward k f b)

theorem latticePrefix_length (all ks : List ℝ) (stage : ℕ) (f b : Signal C) :
    (latticePrefix all ks stage f b).length = ks.length-1 := by
  induction ks generalizing stage f b with
  | nil => rfl
  | cons k rest ih =>
      cases rest with
      | nil => rfl
      | cons next tail => simp [latticePrefix, ih]

theorem latticeLoop_prefix_and_final_tail (all ks : List ℝ) (stage remaining : ℕ)
    (f b : Signal C) (hne : ks ≠ []) (h : ks.length ≤ remaining+1) :
    latticeLoop all ks stage remaining f b = latticePrefix all ks stage f b ++
      window (remaining-(ks.length-1)) (finalForward ks f b) := by
  induction ks generalizing stage remaining f b with
  | nil => contradiction
  | cons k rest ih =>
      cases rest with
      | nil => simp [latticeLoop, latticePrefix, finalForward]
      | cons next tail =>
          simp only [latticeLoop, latticePrefix, finalForward, List.cons_append]
          rw [ih (stage+1) (remaining-1) (shift 1 (forward k f b)) (backward k f b)
            (by simp) (by simp at h ⊢; omega)]
          have he : remaining-1-((next::tail).length-1) =
              remaining-((k::next::tail).length-1) := by simp; omega
          rw [he]

theorem lattice_all_stages_prefix_and_tail (ks : List ℝ) (n : ℕ) (x : Signal C)
    (hne : ks ≠ []) (h : ks.length ≤ n) :
    lattice ks n x = (fun c => x 0 c * prefixScale ks 0) ::
      (latticePrefix ks ks 1 (shift 1 x) x ++
        window (n-ks.length) (finalForward ks (shift 1 x) x)) := by
  cases ks with
  | nil => contradiction
  | cons k tail =>
      simp only [lattice]
      rw [latticeLoop_prefix_and_final_tail _ _ _ _ _ _ (by simp) (by simp at h ⊢; omega)]
      have he : n-1-((k::tail).length-1)=n-(k::tail).length := by simp; omega
      rw [he]

inductive Mode where
  | stationary | conditionalPrefix | stationaryInnovations
  deriving DecidableEq

def whiten (mode : Mode) (coefficients reflections : List ℝ) (root : ℕ → ℕ → ℝ)
    (_diagnosticVariance : ℝ) (n : ℕ) (x : Signal C) : Rows C :=
  match mode with
  | .stationary => stationary coefficients root n x
  | .conditionalPrefix => conditional coefficients n x
  | .stationaryInnovations => lattice reflections n x

theorem diagnostic_variance_does_not_rescale (mode : Mode) (coefficients reflections : List ℝ)
    (root : ℕ → ℕ → ℝ) (q r : ℝ) (n : ℕ) (x : Signal C) :
    whiten mode coefficients reflections root q n x =
      whiten mode coefficients reflections root r n x := rfl

theorem conditional_order_zero (n : ℕ) (x : Signal C) : conditional [] n x = window n x := by
  simp [conditional, arTail_nil]

theorem stationary_order_zero (root : ℕ → ℕ → ℝ) (n : ℕ) (x : Signal C) :
    stationary [] root n x = window n x := by
  simp [stationary, window, conditional_order_zero]

theorem lattice_order_zero (n : ℕ) (x : Signal C) : lattice [] n x = window n x := rfl

theorem conditional_combine (coefficients : List ℝ) (n i : ℕ)
    (a b : ℝ) (x y : Signal C) (c : C) :
    rowAt (conditional coefficients n (combine a b x y)) i c =
      a*rowAt (conditional coefficients n x) i c + b*rowAt (conditional coefficients n y) i c := by
  simp only [conditional, rowAt_window, arTail_combine]
  split_ifs <;> simp [combine]

theorem conditional_causal (coefficients : List ℝ) (n i : ℕ)
    (x y : Signal C) (c : C) (h : ∀ t ≤ coefficients.length+i, x t c = y t c) :
    rowAt (conditional coefficients n x) i c = rowAt (conditional coefficients n y) i c := by
  simp only [conditional, rowAt_window]
  split_ifs
  · exact arTail_causal coefficients i x y c h
  · rfl

theorem stationary_combine (coefficients : List ℝ) (root : ℕ → ℕ → ℝ) (n i : ℕ)
    (a b : ℝ) (x y : Signal C) (c : C) :
    rowAt (stationary coefficients root n (combine a b x y)) i c =
      a*rowAt (stationary coefficients root n x) i c +
      b*rowAt (stationary coefficients root n y) i c := by
  by_cases hi : i < coefficients.length
  · rw [stationary_prefix_coordinate _ _ _ _ _ _ hi,
      stationary_prefix_coordinate _ _ _ _ _ _ hi,
      stationary_prefix_coordinate _ _ _ _ _ _ hi]
    exact prefixSolve_combine root a b x y i c
  · have he : coefficients.length+(i-coefficients.length)=i := by omega
    rw [← he, stationary_tail_coordinate, stationary_tail_coordinate, stationary_tail_coordinate]
    exact conditional_combine coefficients n _ a b x y c

theorem stationary_causal (coefficients : List ℝ) (root : ℕ → ℕ → ℝ) (n i : ℕ)
    (x y : Signal C) (c : C) (h : ∀ t ≤ i, x t c = y t c) :
    rowAt (stationary coefficients root n x) i c =
      rowAt (stationary coefficients root n y) i c := by
  by_cases hi : i < coefficients.length
  · rw [stationary_prefix_coordinate _ _ _ _ _ _ hi,
      stationary_prefix_coordinate _ _ _ _ _ _ hi]
    exact prefixSolve_causal root x y i c h
  · have he : coefficients.length+(i-coefficients.length)=i := by omega
    rw [← he, stationary_tail_coordinate, stationary_tail_coordinate]
    apply conditional_causal
    simpa only [he] using h

def InputAdmissible (mode : Mode) (order n columns : ℕ) : Prop :=
  0 < columns ∧ match mode with
    | .conditionalPrefix => order+1 ≤ n
    | _ => max 1 order ≤ n

def outputRows (mode : Mode) (order n : ℕ) : ℕ :=
  match mode with
  | .conditionalPrefix => n-order
  | _ => n

def observationTime (mode : Mode) (order i : ℕ) : ℕ :=
  match mode with
  | .conditionalPrefix => order+i
  | _ => i

theorem whiten_output_length (mode : Mode) (coefficients reflections : List ℝ)
    (root : ℕ → ℕ → ℝ) (variance : ℝ) (n columns : ℕ) (x : Signal (Fin columns))
    (h : InputAdmissible mode coefficients.length n columns)
    (hr : reflections.length = coefficients.length) :
    (whiten mode coefficients reflections root variance n x).length =
      outputRows mode coefficients.length n := by
  cases mode with
  | stationary =>
      apply stationary_output_length
      simp only [InputAdmissible] at h
      omega
  | conditionalPrefix => exact conditional_output_length _ _ _
  | stationaryInnovations =>
      apply lattice_output_length
      simp only [InputAdmissible] at h
      omega

theorem whiten_combine (mode : Mode) (coefficients reflections : List ℝ)
    (root : ℕ → ℕ → ℝ) (variance a b : ℝ) (n i : ℕ) (x y : Signal C) (c : C) :
    rowAt (whiten mode coefficients reflections root variance n (combine a b x y)) i c =
      a*rowAt (whiten mode coefficients reflections root variance n x) i c +
      b*rowAt (whiten mode coefficients reflections root variance n y) i c := by
  cases mode
  · exact stationary_combine _ _ _ _ _ _ _ _ _
  · exact conditional_combine _ _ _ _ _ _ _ _
  · exact lattice_combine _ _ _ _ _ _ _ _

theorem whiten_within_record_causal (mode : Mode) (coefficients reflections : List ℝ)
    (root : ℕ → ℕ → ℝ) (variance : ℝ) (n i : ℕ) (x y : Signal C) (c : C)
    (h : ∀ t ≤ observationTime mode coefficients.length i, x t c = y t c) :
    rowAt (whiten mode coefficients reflections root variance n x) i c =
      rowAt (whiten mode coefficients reflections root variance n y) i c := by
  cases mode
  · exact stationary_causal _ _ _ _ _ _ _ h
  · exact conditional_causal _ _ _ _ _ _ h
  · exact lattice_causal _ _ _ _ _ _ h

theorem whiten_columns_independent (mode : Mode) (coefficients reflections : List ℝ)
    (root : ℕ → ℕ → ℝ) (variance : ℝ) (n i : ℕ) (x y : Signal C) (c : C)
    (h : ∀ t, x t c = y t c) :
    rowAt (whiten mode coefficients reflections root variance n x) i c =
      rowAt (whiten mode coefficients reflections root variance n y) i c :=
  whiten_within_record_causal _ _ _ _ _ _ _ _ _ _ (fun t _ => h t)

/-- Nonempty AR(1) unit-innovation stationary covariance is 25/16, root 5/4. -/
def fixtureRoot : ℕ → ℕ → ℝ := fun _ _ => 5/4
def fixtureOnes : Signal (Fin 1) := fun _ _ => 1

theorem nonempty_fixture_covariance_equation :
    (25/16 : ℝ) * (1-(3/5 : ℝ)^2) = 1 ∧ (5/4 : ℝ)^2=25/16 ∧ 0<(5/4 : ℝ) := by
  norm_num

theorem nonempty_fixture_admissible (mode : Mode) : InputAdmissible mode 1 8 1 := by
  cases mode <;> norm_num [InputAdmissible]

theorem nonempty_stationary_first_row :
    rowAt (stationary [3/5] fixtureRoot 8 fixtureOnes) 0 0 = 4/5 := by
  rw [stationary_prefix_coordinate _ _ _ _ _ _ (by simp)]
  rw [prefixSolve_zero]
  norm_num [fixtureRoot, fixtureOnes]

theorem nonempty_stationary_interior_row :
    rowAt (stationary [3/5] fixtureRoot 8 fixtureOnes) 4 0 = 2/5 := by
  change rowAt (stationary [3/5] fixtureRoot 8 fixtureOnes) (([3/5] : List ℝ).length+3) 0 = 2/5
  rw [stationary_tail_coordinate, conditional_coordinate _ _ _ _ _ (by simp)]
  norm_num [lagTerms, fixtureOnes]

theorem fresh_start_differs_from_record_continuation :
    rowAt (stationary [3/5] fixtureRoot 8 fixtureOnes) 0 0 ≠
      rowAt (stationary [3/5] fixtureRoot 8 fixtureOnes) 4 0 := by
  rw [nonempty_stationary_first_row, nonempty_stationary_interior_row]
  norm_num

def twoStageFixture : Signal (Fin 1) := fun t _ => t+1

theorem nonempty_two_stage_tail_value :
    rowAt (lattice [1/2,1/4] 5 twoStageFixture) 2 0 = 9/2 := by
  change rowAt (lattice [1/2,1/4] 5 twoStageFixture) (0+2) 0 = 9/2
  rw [lattice_two_stage_tail _ _ _ _ _ _ (by norm_num)]
  norm_num [twoStageFixture]

end

end N2P3.TemporalWhitening
