import N2P3.RawScope
import N2P3.FiniteSOS
import N2P3.Covariance

/-!
# A composed authorized-input, finite-filter and covariance slice

This composes the proved extraction and actual finite SOS maps. Reading any
additional source metadata, labels, globals or files remains a Python caller
refinement obligation; it is not hidden in the mathematical interface.
-/

namespace N2P3.FirstSlice

open Matrix MeasureTheory ProbabilityTheory
open scoped BigOperators ProbabilityTheory

variable {nSamples nEvents k n s : ℕ} {Event Label Ω : Type*}

def selectedSamples (scope : RawScope.Scope nSamples nEvents k)
    (width : scope.stop - scope.first = n + 1)
    (record : RawScope.SourceRecord nSamples nEvents ℝ Event Label) :
    FiniteSOS.Samples (n + 1) :=
  fun i => record.samples (RawScope.sampleIndex scope (Fin.cast width.symm i))

def extractionMatrix (scope : RawScope.Scope nSamples nEvents k)
    (width : scope.stop - scope.first = n + 1) :
    Matrix (Fin (n + 1)) (Fin nSamples) ℝ :=
  fun i j => if j = RawScope.sampleIndex scope (Fin.cast width.symm i) then 1 else 0

theorem extractionMatrix_mulVec (scope : RawScope.Scope nSamples nEvents k)
    (width : scope.stop - scope.first = n + 1) (x : Fin nSamples → ℝ) :
    extractionMatrix scope width *ᵥ x =
      fun i => x (RawScope.sampleIndex scope (Fin.cast width.symm i)) := by
  funext i
  simp [extractionMatrix, Matrix.mulVec, dotProduct]

theorem selectedSamples_is_extraction (scope : RawScope.Scope nSamples nEvents k)
    (width : scope.stop - scope.first = n + 1)
    (record : RawScope.SourceRecord nSamples nEvents ℝ Event Label) :
    selectedSamples scope width record = extractionMatrix scope width *ᵥ record.samples := by
  rw [extractionMatrix_mulVec]
  rfl

def isolatedFilter (scope : RawScope.Scope nSamples nEvents k)
    (width : scope.stop - scope.first = n + 1) (c : Fin s → FiniteSOS.Section)
    (zi : FiniteSOS.State s) (limit : ℕ)
    (record : RawScope.SourceRecord nSamples nEvents ℝ Event Label) :
    FiniteSOS.Samples (n + 1) :=
  FiniteSOS.paddedFilter c zi n (min limit n) (Nat.min_le_right _ _)
    (selectedSamples scope width record)

/-- A real source sample outside the chosen absolute support cannot affect this
actual finite SOS composition, including both observed padding initial states. -/
theorem isolated_filter_noninterference
    (scope : RawScope.Scope nSamples nEvents k)
    (width : scope.stop - scope.first = n + 1) (c : Fin s → FiniteSOS.Section)
    (zi : FiniteSOS.State s) (limit : ℕ)
    (a b : RawScope.SourceRecord nSamples nEvents ℝ Event Label)
    (hSamples : ∀ j, scope.first ≤ scope.origin + j.val →
      scope.origin + j.val < scope.stop → a.samples j = b.samples j) :
    isolatedFilter scope width c zi limit a = isolatedFilter scope width c zi limit b := by
  have hinput : selectedSamples scope width a = selectedSamples scope width b := by
    funext i
    have hj := RawScope.sampleIndex_in_scope scope (Fin.cast width.symm i)
    exact hSamples _ hj.1 hj.2
  unfold isolatedFilter
  rw [hinput]

/-- Explicitly fixed coefficients and zi give a full source-to-output matrix,
including extraction first; it is not a crop of globally filtered data. -/
noncomputable def sourceFilterMatrix (scope : RawScope.Scope nSamples nEvents k)
    (width : scope.stop - scope.first = n + 1) (c : Fin s → FiniteSOS.Section)
    (zi : FiniteSOS.State s) (limit : ℕ) :
    Matrix (Fin (n + 1)) (Fin nSamples) ℝ :=
  LinearMap.toMatrix'
    (FiniteSOS.paddedFilter c zi n (min limit n) (Nat.min_le_right _ _)) *
      extractionMatrix scope width

theorem sourceFilterMatrix_mulVec (scope : RawScope.Scope nSamples nEvents k)
    (width : scope.stop - scope.first = n + 1) (c : Fin s → FiniteSOS.Section)
    (zi : FiniteSOS.State s) (limit : ℕ)
    (record : RawScope.SourceRecord nSamples nEvents ℝ Event Label) :
    sourceFilterMatrix scope width c zi limit *ᵥ record.samples =
      isolatedFilter scope width c zi limit record := by
  unfold sourceFilterMatrix isolatedFilter
  rw [← Matrix.mulVec_mulVec, LinearMap.toMatrix'_mulVec,
    ← selectedSamples_is_extraction scope width record]

/-- True covariance of the composed legal source/filter map, including every
deterministic observed padding state and all correlations in the source. -/
theorem isolated_filter_covariance [MeasurableSpace Ω]
    (mu : Measure Ω) [IsProbabilityMeasure mu]
    (scope : RawScope.Scope nSamples nEvents k)
    (width : scope.stop - scope.first = n + 1) (c : Fin s → FiniteSOS.Section)
    (zi : FiniteSOS.State s) (limit : ℕ)
    (X : Fin nSamples → Ω → ℝ) (hX : ∀ i, MemLp (X i) 2 mu) :
    Covariance.covarianceMatrix mu
      (Covariance.linearImage (sourceFilterMatrix scope width c zi limit) X) =
      sourceFilterMatrix scope width c zi limit * Covariance.covarianceMatrix mu X *
        (sourceFilterMatrix scope width c zi limit)ᵀ := by
  exact Covariance.covariance_linear_image mu X hX _

end N2P3.FirstSlice
