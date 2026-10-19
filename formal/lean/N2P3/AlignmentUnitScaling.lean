import N2P3.OcularPointwise
import Mathlib.Data.Real.Sqrt
import Mathlib.Tactic.FieldSimp

/-!
# Relative EA operator mathematics and voltage-unit compensation

Source: euclidean_alignment.py:80-103,209-227,248-266. The empirical object
is an UNCENTERED second moment. Its trace/channel scale carries V²; the
relative spectrum is dimensionless. The same normalized operator/eigenbasis
can be used after a positive change of voltage units. Reconstruction then
scales the reference by k² and inverse by 1/k, preserving component values.
Eigenpair computation, clipping, and floating-point rounding remain native
boundaries; exact reconstruction is separated from library execution.
-/
namespace N2P3.AlignmentUnitScaling
open scoped BigOperators
noncomputable section
variable {N C T : Type*} [Fintype N] [Fintype C] [Fintype T]

def secondMoment (x : N → C → T → ℝ) (c d : C) : ℝ :=
  (∑ n, ∑ t, x n c t*x n d t)/((Fintype.card N : ℝ)*(Fintype.card T : ℝ))

omit [Fintype C] in
theorem common_voltage_scaling_squares_the_empirical_second_moment
    (x : N → C → T → ℝ) (k : ℝ) (c d : C) :
    secondMoment (fun n c t => k*x n c t) c d=k^2*secondMoment x c d := by
  have products : (∑ n, ∑ t, (k*x n c t)*(k*x n d t))=
      k^2*(∑ n, ∑ t, x n c t*x n d t) := by
    calc
      _ = ∑ n, ∑ t, k^2*(x n c t*x n d t) := by
        apply Finset.sum_congr rfl
        intro n _
        apply Finset.sum_congr rfl
        intro t _
        ring
      _ = _ := by simp_rw [← Finset.mul_sum]
  unfold secondMoment
  rw [products]
  ring

def energyScale (S : C → C → ℝ) : ℝ := (∑ c, S c c)/(Fintype.card C : ℝ)

def normalized (S : C → C → ℝ) (c d : C) : ℝ := S c d/energyScale S

theorem trace_per_channel_scale_transforms_quadratically (S : C → C → ℝ) (k : ℝ) :
    energyScale (fun c d => k^2*S c d)=k^2*energyScale S := by
  simp only [energyScale,← Finset.mul_sum]
  ring

theorem positive_unit_change_preserves_relative_covariance
    (S : C → C → ℝ) (k : ℝ) (hk : 0 < k) (hs : 0 < energyScale S) (c d : C) :
    normalized (fun c d => k^2*S c d) c d=normalized S c d := by
  unfold normalized
  rw [trace_per_channel_scale_transforms_quadratically]
  field_simp [ne_of_gt hk,ne_of_gt hs]

def shrunkEigenvalue (eigenvalue shrinkage : ℝ) : ℝ :=
  (1-shrinkage)*eigenvalue+shrinkage

def regularizedVariance (eigenvalue floor scale : ℝ) : ℝ := max eigenvalue floor*scale

theorem positive_relative_floor_and_signal_energy_give_positive_variance
    (eigenvalue floor scale : ℝ) (hf : 0 < floor) (hs : 0 < scale) :
    0 < regularizedVariance eigenvalue floor scale :=
  mul_pos (lt_of_lt_of_le hf (le_max_right eigenvalue floor)) hs

theorem reconstructed_reference_variance_scales_quadratically
    (eigenvalue floor scale k : ℝ) :
    regularizedVariance eigenvalue floor (k^2*scale)=k^2*regularizedVariance eigenvalue floor scale := by
  unfold regularizedVariance
  ring

theorem positive_unit_change_scales_spectral_square_root
    (variance k : ℝ) (hk : 0 < k) : Real.sqrt (k^2*variance)=k*Real.sqrt variance := by
  rw [Real.sqrt_mul (sq_nonneg k),Real.sqrt_sq (le_of_lt hk)]

theorem positive_unit_change_divides_the_spectral_inverse
    (variance k : ℝ) (hk : 0 < k) :
    1/Real.sqrt (k^2*variance)=(1/Real.sqrt variance)/k := by
  rw [positive_unit_change_scales_spectral_square_root variance k hk]
  simp only [mul_inv_rev,div_eq_mul_inv,one_mul]

def spectralEntry (Q : C → C → ℝ) (diagonal : C → ℝ) (c d : C) : ℝ :=
  ∑ j, Q c j*diagonal j*Q d j

theorem common_spectral_scaling_scales_every_reconstructed_entry
    (Q : C → C → ℝ) (diagonal : C → ℝ) (k : ℝ) (c d : C) :
    spectralEntry Q (fun j => k*diagonal j) c d=k*spectralEntry Q diagonal c d := by
  unfold spectralEntry
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  ring

def transform (W : C → C → ℝ) (x : C → ℝ) (c : C) : ℝ := ∑ d, W c d*x d

theorem inverse_unit_scaling_compensates_a_common_input_unit_change
    (W : C → C → ℝ) (x : C → ℝ) (k : ℝ) (hk : k ≠ 0) (c : C) :
    transform (fun c d => W c d/k) (fun d => k*x d) c=transform W x c := by
  unfold transform
  apply Finset.sum_congr rfl
  intro d _
  field_simp

omit [Fintype N] [Fintype T] in
theorem frozen_transform_only_reads_the_current_row_and_sample
    (W : C → C → ℝ) (x y : N → C → T → ℝ) (n : N) (t : T)
    (same : ∀ c, x n c t=y n c t) (c : C) :
    transform W (fun d => x n d t) c=transform W (fun d => y n d t) c := by
  simp only [same]

theorem regularized_reference_need_not_exactly_whiten_empirical_energy :
    1/regularizedVariance (shrunkEigenvalue (2/5) (4/5)) (1/1000000) (5/2)=5/11 ∧
    4/regularizedVariance (shrunkEigenvalue (8/5) (4/5)) (1/1000000) (5/2)=10/7 ∧
    (5/11 : ℝ) ≠ 1 := by
  norm_num [regularizedVariance,shrunkEigenvalue]

theorem the_uncentered_second_moment_is_not_DC_translation_invariant :
    secondMoment (fun (_ : Fin 2) (_ : Fin 1) (_ : Fin 1) => (1 : ℝ)) 0 0=1 ∧
    secondMoment (fun (_ : Fin 2) (_ : Fin 1) (_ : Fin 1) => (2 : ℝ)) 0 0=4 := by
  norm_num [secondMoment,Fin.sum_univ_succ]

end
end N2P3.AlignmentUnitScaling
