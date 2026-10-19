import N2P3.RetainedGaussianLaw

/-!
Two deliberate false assertions about the actual retained Gaussian measure.
The first changes the derived normalized mass to two. The second invents unit
mass outside the nuisance-retained observation support. Each must reduce to
one unsolved `False` goal; a failed import/type check is not a valid negative.
-/

namespace N2P3.NegativeRetainedGaussianLaw

open Matrix MeasureTheory
open N2P3.RetainedGaussianLaw N2P3.RetainedMatrixMeasure N2P3.RetainedMeasure
open scoped Matrix ENNReal

noncomputable section

variable {n q m r c : Type*}
variable [Fintype n] [Fintype q] [Fintype m] [Fintype r] [Fintype c]
variable [DecidableEq n] [DecidableEq q] [DecidableEq m] [DecidableEq r] [DecidableEq c]

example (B : Matrix n m ℝ) (hB : Bᵀ * B = 1)
    (U : Matrix n r ℝ) (H : Matrix m m ℝ) (L Sigma : Matrix c c ℝ) (M : Matrix n c ℝ)
    (hH : H * Hᵀ = N2P3.GaussianLowRank.fullCovariance (Bᵀ * U))
    (hL : L * Lᵀ = Sigma) (hSigma : Sigma.PosDef) :
    (∫⁻ x, ENNReal.ofReal (candidatePDF B H L M x) ∂coordinateVolume (channelLift B)) = 2 := by
  rw [candidateDensity_mass_one B hB U H L Sigma M hH hL hSigma]
  norm_num

example (Q : Matrix n q ℝ) (B : Matrix n m ℝ) (hB : Bᵀ * B = 1)
    (hPB : B * Bᵀ = N2P3.NuisanceProjection.projection Q)
    (H : Matrix m m ℝ) (L : Matrix c c ℝ) (M : Matrix n c ℝ) :
    candidateLaw B H L M {x | N2P3.NuisanceProjection.projection Q *
      N2P3.MatrixNormalCoordinates.matricize x ≠ N2P3.MatrixNormalCoordinates.matricize x} = 1 := by
  rw [candidateLaw_supported Q B hB hPB H L M]
  norm_num

end

end N2P3.NegativeRetainedGaussianLaw
