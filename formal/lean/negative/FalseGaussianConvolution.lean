import N2P3.GaussianConvolution

/-! Two deliberate failures: replacing the genuine predictive integral with
the plug-in variance-one peak, and dropping predictive mass normalization. -/

namespace N2P3.NegativeGaussianConvolution

open MeasureTheory N2P3.GaussianConvolution

-- Marginalizing W with variance one adds its uncertainty to noise variance one.
-- The centered predictive peak is lower than the variance-one plug-in peak.
example : (∫ x, scalarModel.jointAtData 0 x) = scalarModel.priorPDF 0 := by
  simp only [scalar_plug_in_peak_wrong]

-- A missing factor of two cannot be called a normalized predictive density.
example : (∫ y, scalarModel.predictivePDF y) = (2 : ℝ) := by
  rw [scalarModel.predictive_integral_one]
  norm_num

end N2P3.NegativeGaussianConvolution
