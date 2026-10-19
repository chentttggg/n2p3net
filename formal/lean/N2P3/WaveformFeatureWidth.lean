import N2P3.ERPState

/-!
# Physical feature width of the current waveform-LDA caller

Sources: baselines/features.py downsampled_time_points/downsample_epochs and
transfer/calibrated.py CalibratedPredictor waveform-LDA constructor branch.
The source computes min(T, int(round(T * 20.0 / sfreq))) and multiplies by
the selected tensor's numel, or the physical channel count when absent.

Arithmetic exposes the implicit integer conversion, multiplication, division
and Python round/int step separately, including failures. It does not replace
binary floating arithmetic or ties-to-even round by exact-real rounding.
The rational fixture implements only exactly integral quotient rounding and
is a nonempty operation-domain witness, not an implementation of Python round.

FeatureFields are decoded shape observations and the single scale > 0/all
operation result. The guard below describes the Boolean admissibility test;
it does not assert evaluation order or complete constructor acceptance. Earlier
selector validation, tensor cloning, preprocessing admission and later finite
state checks must be composed separately. No whole-downsample, whole-feature
or whole-constructor success is assumed as a primitive.
-/

namespace N2P3.WaveformFeatureWidth

open N2P3.StateLedger

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map

structure Arithmetic (Scalar : Type) where
  fromNat : ℕ → Check Scalar
  twenty : Scalar
  multiply : Scalar → Scalar → Check Scalar
  divide : Scalar → Scalar → Check Scalar
  roundInt : Scalar → Check ℤ

def downsampleLength {Scalar : Type} (ops : Arithmetic Scalar)
    (times : ℕ) (sfreq : Scalar) : Check ℤ := do
  let timeValue ← ops.fromNat times
  let numerator ← ops.multiply timeValue ops.twenty
  let quotient ← ops.divide numerator sfreq
  let rounded ← ops.roundInt quotient
  pure (min (times : ℤ) rounded)

theorem length_from_individual_operations {Scalar : Type} (ops : Arithmetic Scalar)
    (times : ℕ) (sfreq timeValue numerator quotient : Scalar) (rounded : ℤ)
    (hconvert : ops.fromNat times = .ok timeValue)
    (hmul : ops.multiply timeValue ops.twenty = .ok numerator)
    (hdiv : ops.divide numerator sfreq = .ok quotient)
    (hround : ops.roundInt quotient = .ok rounded) :
    downsampleLength ops times sfreq = .ok (min (times : ℤ) rounded) := by
  simp [downsampleLength, hconvert, hmul, hdiv, hround]

theorem conversion_failure_propagates {Scalar : Type} (ops : Arithmetic Scalar)
    (times : ℕ) (sfreq : Scalar) (error : LedgerFault)
    (h : ops.fromNat times = .error error) :
    downsampleLength ops times sfreq = .error error := by
  simp [downsampleLength, h]

theorem division_failure_propagates {Scalar : Type} (ops : Arithmetic Scalar)
    (times : ℕ) (sfreq timeValue numerator : Scalar) (error : LedgerFault)
    (hconvert : ops.fromNat times = .ok timeValue)
    (hmul : ops.multiply timeValue ops.twenty = .ok numerator)
    (hdiv : ops.divide numerator sfreq = .error error) :
    downsampleLength ops times sfreq = .error error := by
  simp [downsampleLength, hconvert, hmul, hdiv]

theorem no_upsampling_for_any_rounded_integer (times : ℕ) (rounded : ℤ) :
    min (times : ℤ) rounded ≤ times := min_le_left _ _

theorem rounded_length_at_least_original_keeps_original (times : ℕ) (rounded : ℤ)
    (h : (times : ℤ) ≤ rounded) : min (times : ℤ) rounded = times := min_eq_left h

theorem shorter_rounded_length_is_used (times : ℕ) (rounded : ℤ)
    (h : rounded ≤ (times : ℤ)) : min (times : ℤ) rounded = rounded := min_eq_right h

def selectedCount (physicalChannels : ℕ) (selected : Option (List ℕ)) : ℕ :=
  match selected with
  | none => physicalChannels
  | some channels => channels.length

def expectedWidth {Scalar : Type} (ops : Arithmetic Scalar) (physicalChannels times : ℕ)
    (sfreq : Scalar) (selected : Option (List ℕ)) : Check ℤ :=
  (downsampleLength ops times sfreq).map (fun width => (selectedCount physicalChannels selected : ℤ) * width)

theorem absent_selector_uses_physical_channels (channels : ℕ) : selectedCount channels none = channels := rfl

theorem present_selector_uses_its_actual_count (channels : ℕ) (selected : List ℕ) :
    selectedCount channels (some selected) = selected.length := rfl

theorem permutation_preserves_width_only {Scalar : Type} (ops : Arithmetic Scalar)
    (channels times : ℕ) (sfreq : Scalar) (first second : List ℕ)
    (h : first.length = second.length) :
    expectedWidth ops channels times sfreq (some first) =
      expectedWidth ops channels times sfreq (some second) := by
  simp [expectedWidth, selectedCount, h]

theorem width_from_individual_operations {Scalar : Type} (ops : Arithmetic Scalar)
    (channels times : ℕ) (sfreq : Scalar) (selected : Option (List ℕ))
    (timeValue numerator quotient : Scalar) (rounded : ℤ)
    (hconvert : ops.fromNat times = .ok timeValue)
    (hmul : ops.multiply timeValue ops.twenty = .ok numerator)
    (hdiv : ops.divide numerator sfreq = .ok quotient)
    (hround : ops.roundInt quotient = .ok rounded) :
    expectedWidth ops channels times sfreq selected =
      .ok ((selectedCount channels selected : ℤ) * min (times : ℤ) rounded) := by
  rw [expectedWidth, length_from_individual_operations ops times sfreq timeValue numerator quotient rounded
    hconvert hmul hdiv hround]
  rfl

structure FeatureFields where
  coefficientShape : List ℕ
  meanShape : List ℕ
  scaleShape : List ℕ
  scaleAllPositive : Bool
  interceptNumel : ℕ

def featureConditions (fields : FeatureFields) (expected : ℤ) : Prop :=
  fields.coefficientShape.map Int.ofNat = [expected] ∧
  fields.meanShape = fields.coefficientShape ∧
  fields.scaleShape = fields.coefficientShape ∧
  fields.scaleAllPositive = true ∧ fields.interceptNumel = 1

def featureGuard (fields : FeatureFields) (expected : ℤ) : Check Unit :=
  if featureConditions fields expected then .ok ()
  else .error (.sourceReject "calibrated LDA feature state has invalid shape or scale")

theorem independent_feature_fields_pass (fields : FeatureFields) (expected : ℤ)
    (hc : fields.coefficientShape.map Int.ofNat = [expected])
    (hm : fields.meanShape = fields.coefficientShape)
    (hs : fields.scaleShape = fields.coefficientShape)
    (hp : fields.scaleAllPositive = true) (hi : fields.interceptNumel = 1) :
    featureGuard fields expected = .ok () := by
  simp [featureGuard, featureConditions, hc, hm, hs, hp, hi]

theorem physical_width_mismatch_refuses (fields : FeatureFields) (expected : ℤ)
    (h : fields.coefficientShape.map Int.ofNat ≠ [expected]) :
    featureGuard fields expected =
      .error (.sourceReject "calibrated LDA feature state has invalid shape or scale") := by
  simp [featureGuard, featureConditions, h]

theorem successful_guard_forces_physical_width (fields : FeatureFields) (expected : ℤ)
    (h : featureGuard fields expected = .ok ()) :
    fields.coefficientShape.map Int.ofNat = [expected] := by
  by_cases hc : featureConditions fields expected
  · exact hc.1
  · simp [featureGuard, hc] at h

theorem nonpositive_scale_refuses (fields : FeatureFields) (expected : ℤ)
    (h : fields.scaleAllPositive = false) :
    featureGuard fields expected =
      .error (.sourceReject "calibrated LDA feature state has invalid shape or scale") := by
  simp [featureGuard, featureConditions, h]

def waveformWidthGuard {Scalar : Type} (ops : Arithmetic Scalar)
    (channels times : ℕ) (sfreq : Scalar) (selected : Option (List ℕ))
    (fields : FeatureFields) : Check Unit := do
  let expected ← expectedWidth ops channels times sfreq selected
  featureGuard fields expected

def integralFixtureArithmetic : Arithmetic ℚ where
  fromNat n := .ok n
  twenty := 20
  multiply a b := .ok (a*b)
  divide a b := if b = 0 then .error (.fieldFailure "division:ZeroDivisionError") else .ok (a/b)
  roundInt q := if q.den = 1 then .ok q.num else .error (.fieldFailure "round:nonintegral fixture domain")

def sixFields : FeatureFields := ⟨[6], [6], [6], true, 1⟩
def fourFields : FeatureFields := ⟨[4], [4], [4], true, 1⟩
def obsoleteTwoFields : FeatureFields := ⟨[2], [2], [2], true, 1⟩

theorem nonempty_exact_integer_arithmetic_length :
    downsampleLength integralFixtureArithmetic 25 250 = .ok 2 := by
  norm_num [downsampleLength, integralFixtureArithmetic]

theorem nonempty_no_upsampling_length :
    downsampleLength integralFixtureArithmetic 2 10 = .ok 2 := by
  norm_num [downsampleLength, integralFixtureArithmetic]

theorem absent_selector_expected_six :
    expectedWidth integralFixtureArithmetic 3 25 250 none = .ok 6 := by
  rw [expectedWidth, nonempty_exact_integer_arithmetic_length]
  norm_num [selectedCount]

theorem ordered_subset_expected_four :
    expectedWidth integralFixtureArithmetic 3 25 250 (some [2,0]) = .ok 4 := by
  rw [expectedWidth, nonempty_exact_integer_arithmetic_length]
  norm_num [selectedCount]

theorem nonempty_physical_feature_guard_passes :
    waveformWidthGuard integralFixtureArithmetic 3 25 250 none sixFields = .ok () := by
  rw [waveformWidthGuard, absent_selector_expected_six]
  norm_num [featureGuard, featureConditions, sixFields]

theorem nonempty_selected_feature_guard_passes :
    waveformWidthGuard integralFixtureArithmetic 3 25 250 (some [2,0]) fourFields = .ok () := by
  rw [waveformWidthGuard, ordered_subset_expected_four]
  norm_num [featureGuard, featureConditions, fourFields]

theorem matching_old_shape_fields_do_not_suffice :
    waveformWidthGuard integralFixtureArithmetic 3 25 250 none obsoleteTwoFields =
      .error (.sourceReject "calibrated LDA feature state has invalid shape or scale") := by
  rw [waveformWidthGuard, absent_selector_expected_six]
  norm_num [featureGuard, featureConditions, obsoleteTwoFields]

theorem selected_subset_cannot_reuse_all_channel_width :
    waveformWidthGuard integralFixtureArithmetic 3 25 250 (some [2,0]) sixFields =
      .error (.sourceReject "calibrated LDA feature state has invalid shape or scale") := by
  rw [waveformWidthGuard, ordered_subset_expected_four]
  norm_num [featureGuard, featureConditions, sixFields]

theorem equal_numel_does_not_replace_coefficient_rank :
    featureGuard ⟨[2,3], [2,3], [2,3], true, 1⟩ 6 =
      .error (.sourceReject "calibrated LDA feature state has invalid shape or scale") := by
  norm_num [featureGuard, featureConditions]

theorem zero_sfreq_operation_failure_is_not_constructor_success :
    waveformWidthGuard integralFixtureArithmetic 3 25 0 none sixFields =
      .error (.fieldFailure "division:ZeroDivisionError") := by
  norm_num [waveformWidthGuard, expectedWidth, downsampleLength, integralFixtureArithmetic]

end
end N2P3.WaveformFeatureWidth
