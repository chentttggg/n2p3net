import N2P3.GaussianJointLaw
import Mathlib.Tactic.Linarith

/-!
Strict false controls for actual joint/marginal/posterior law constructions.
The one-coordinate instances satisfy the algebraic source covariance/ridge
contracts; they are not assertions of complete EEG-record admission. Every
example must be rejected with exactly one unsolved False goal.
-/

namespace N2P3.GaussianJointLaw.Negative

open Matrix MeasureTheory ProbabilityTheory
open N2P3.ScaleGauge N2P3.GaussianConditioning N2P3.MatrixNormalCoordinates N2P3.BayesDensity
open scoped Matrix BigOperators ENNReal ComplexOrder ProbabilityTheory

private abbrev Scalar := Fin 1
private abbrev Mat := Matrix Scalar Scalar ℝ
private abbrev X := Coeff Scalar Scalar

private theorem scalar_volume : (volume : Measure X) Set.univ = ∞ := by
  simp [X, Coeff, volume_pi]

private theorem scalar_joint_mass : jointMeasure (0 : Mat) 1 (1 : Mat) (1 : Mat) Set.univ = 1 :=
  joint_mass_one (0 : Mat) (1 : Mat) (1 : Mat) (1 : Mat) (1 : Mat) 1
    (by simp) (by simp) Matrix.PosDef.one (by norm_num) Matrix.PosDef.one

private theorem scalar_evidence_mass : evidenceMeasure (0 : Mat) 1 (1 : Mat) (1 : Mat) Set.univ = 1 := by
  haveI := evidence_probability (0 : Mat) (1 : Mat) (1 : Mat) (1 : Mat) (1 : Mat) 1
    (by simp) (by simp) Matrix.PosDef.one (by norm_num) Matrix.PosDef.one
  exact measure_univ

private theorem scalar_jacobian :
    jacobianNormalizer (1 : Mat) (1 : Mat) = (Real.sqrt (2 * Real.pi))⁻¹ := by
  simp [jacobianNormalizer, whiteningMatrix]

private theorem scalar_datafactor : dataFactor (0 : Mat) (0 : Mat) (1 : Mat) (1 : Mat) 1 = 1 := by
  simp [dataFactor, constantEnergy, mean, rowCov, precision, weightedEnergy]

private theorem scalar_evidence_at_zero : evidenceNN (0 : Mat) 1 (1 : Mat) (1 : Mat) (0 : X) =
    ENNReal.ofReal ((Real.sqrt (2 * Real.pi))⁻¹) := by
  have h := evidenceNN_as_explicit_normalizer (0 : Mat) (1 : Mat) (1 : Mat)
    (1 : Mat) (1 : Mat) 1 (1 : Mat) (by simp [precision]) (by simp) (by simp)
    Matrix.PosDef.one (by norm_num) Matrix.PosDef.one (0 : X)
  change evidenceNN (0 : Mat) 1 (1 : Mat) (1 : Mat) (0 : X) = _ at h
  have hprior : priorRowFactor (p := Scalar) 1 = (1 : Mat) := by simp [priorRowFactor]
  have hdata : dataFactor (0 : Mat) (matricize (0 : X)) (1 : Mat) (1 : Mat) 1 = 1 := by
    change dataFactor (0 : Mat) (0 : Mat) (1 : Mat) (1 : Mat) 1 = 1
    exact scalar_datafactor
  simp only [hprior, inv_one, scalar_jacobian, hdata, one_div, inv_inv] at h
  have hpos : 0 < Real.sqrt (2 * Real.pi) := Real.sqrt_pos.mpr (mul_pos (by norm_num) Real.pi_pos)
  have hs : (Real.sqrt (2 * Real.pi))⁻¹ * (Real.sqrt (2 * Real.pi))⁻¹ * Real.sqrt (2 * Real.pi) =
      (Real.sqrt (2 * Real.pi))⁻¹ := by
    rw [mul_assoc, inv_mul_cancel₀ (ne_of_gt hpos), mul_one]
  rwa [hs] at h

private theorem scalar_evidence_at_zero_ne_one :
    evidenceNN (0 : Mat) 1 (1 : Mat) (1 : Mat) (0 : X) ≠ 1 := by
  rw [scalar_evidence_at_zero]
  intro h
  have hsqrt : 1 < Real.sqrt (2 * Real.pi) := by
    apply Real.lt_sqrt_of_sq_lt
    nlinarith [Real.two_le_pi]
  have hpos : 0 < Real.sqrt (2 * Real.pi) := lt_trans (by norm_num) hsqrt
  have hreal := congrArg ENNReal.toReal h
  rw [ENNReal.toReal_ofReal (inv_nonneg.mpr hpos.le), ENNReal.toReal_one] at hreal
  exact (ne_of_lt (inv_lt_one_of_one_lt₀ hsqrt)) hreal

/-- False: the normalized joint probability is ambient product Lebesgue volume. -/
example : jointMeasure (0 : Mat) 1 (1 : Mat) (1 : Mat) =
    (volume : Measure X).prod (volume : Measure X) := by
  have hne : jointMeasure (0 : Mat) 1 (1 : Mat) (1 : Mat) ≠
      (volume : Measure X).prod (volume : Measure X) := by
    intro h
    have hu := congrArg (fun μ : Measure (X × X) => μ Set.univ) h
    change jointMeasure (0 : Mat) 1 (1 : Mat) (1 : Mat) Set.univ =
      ((volume : Measure X).prod (volume : Measure X)) Set.univ at hu
    have hp : ((volume : Measure X).prod (volume : Measure X)) Set.univ = ∞ := by
      rw [← Set.univ_prod_univ, Measure.prod_prod, scalar_volume]
      norm_num
    rw [scalar_joint_mass, hp] at hu
    norm_num at hu
  simp only [hne]

/-- False: a properly normalized data marginal can be replaced with raw volume. -/
example : evidenceMeasure (0 : Mat) 1 (1 : Mat) (1 : Mat) = (volume : Measure X) := by
  have hne : evidenceMeasure (0 : Mat) 1 (1 : Mat) (1 : Mat) ≠ (volume : Measure X) := by
    intro h
    have hu := congrArg (fun μ : Measure X => μ Set.univ) h
    change evidenceMeasure (0 : Mat) 1 (1 : Mat) (1 : Mat) Set.univ =
      (volume : Measure X) Set.univ at hu
    rw [scalar_evidence_mass, scalar_volume] at hu
    exact ENNReal.one_ne_top hu
  simp only [hne]

/-- False: an unnormalized slice of the joint is already a posterior probability.
Its mass is the nonunit evidence at Y=0, so division by that evidence is essential. -/
example : ((volume : Measure X).withDensity
    (fun x => jointNN (0 : Mat) 1 (1 : Mat) (1 : Mat) ((0 : X), x))) Set.univ = 1 := by
  rw [withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ]
  change evidenceNN (0 : Mat) 1 (1 : Mat) (1 : Mat) (0 : X) = 1
  simp only [scalar_evidence_at_zero_ne_one]

end N2P3.GaussianJointLaw.Negative
