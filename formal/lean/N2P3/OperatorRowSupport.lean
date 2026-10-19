import N2P3.OperatorGeometry
import N2P3.StateLedger

/-!
# Current normalized per-row operator-LDA support gate

Source: operator_lda.py:_validate_support, current SHA
41e555fd7c888f588112f14d96f3741d75f158d58c60793484094f4b65a93b4f.
This is the v3 normalized-row implementation, separately identified from the
historical batch-Frobenius predicate in OperatorGeometry and the intermediate
axis-only implementation. Existing historical proof/evidence is unchanged.

The finite-real graph computes each row's max-absolute scale, division,
right-multiplied projection residual, two row norms, max/tiny correction,
strict comparison and the final any. The norm is a small per-vector numerical
operation, with finite/NaN/+infinity results distinguished in the comparison.
Exact arithmetic/division/matmul are proved at their mathematical layer; actual
NumPy/BLAS rounding, underflow, overflow and dependence on a backend's batch
layout are not refined. Thus append/subset statements concern this literal
pointwise computation graph, not universal bit-level IEEE stability.

Empty batch with positive feature width takes no row operations and any([])
is false. Zero feature width instead fails at np.max's empty reduction axis,
even for a zero-row matrix. This is a numerical-operation failure, not a new
source validation refusal. Upstream centering and final affine overflow are
outside this helper's guarantee.
-/

namespace N2P3.OperatorRowSupport

open N2P3.StateLedger N2P3.OperatorGeometry

noncomputable section
attribute [local instance] Classical.propDecidable

abbrev Row (d : ℕ) := Fin d → ℝ

inductive NormResult where
  | finite (value : ℝ)
  | nan
  | positiveInfinity

/-- np.greater returns false if either operand is NaN. -/
def normGreater : NormResult → NormResult → Bool
  | .nan, _ | _, .nan => false
  | .positiveInfinity, .finite _ => true
  | .finite _, .positiveInfinity | .positiveInfinity, .positiveInfinity => false
  | .finite left, .finite right => decide (right < left)

def normLimit (normalizedNorm : NormResult) (tinyOverScale : ℝ) : NormResult :=
  match normalizedNorm with
  | .finite value => .finite (max value tinyOverScale * supportTolerance)
  | .nan => .nan
  | .positiveInfinity => .positiveInfinity

abbrev NormOperation (d : ℕ) := Row d → NormResult

def exactNorm {d : ℕ} (x : Row d) : NormResult := .finite (vectorNorm x)

/-- Abs values are nonnegative, so zero is a neutral seed for a nonempty
max-absolute reduction. The zero-axis case is handled before this graph. -/
def rowMaxAbs {d : ℕ} (x : Row d) : ℝ :=
  (List.ofFn (fun j => |x j|)).foldl max 0

def rowScale {d : ℕ} (x : Row d) : ℝ := max (rowMaxAbs x) supportTiny

def normalizedRow {d : ℕ} (x : Row d) : Row d := fun j => x j / rowScale x

def rowProject {d : ℕ} (projector : Matrix (Fin d) (Fin d) ℝ) (x : Row d) : Row d :=
  fun j => ∑ k, x k * projector k j

def rowResidual {d : ℕ} (projector : Matrix (Fin d) (Fin d) ℝ) (x : Row d) : Row d :=
  fun j => x j - rowProject projector x j

structure RowTrace (d : ℕ) where
  scale : ℝ
  normalized : Row d
  residual : Row d
  normalizedNorm : NormResult
  residualNorm : NormResult
  limit : NormResult
  tooLarge : Bool

def evaluateRow {d : ℕ} (norm : NormOperation d)
    (projector : Matrix (Fin d) (Fin d) ℝ) (centered : Row d) : RowTrace d :=
  let scale := rowScale centered
  let normalized := normalizedRow centered
  let residual := rowResidual projector normalized
  let normalizedNorm := norm normalized
  let residualNorm := norm residual
  let limit := normLimit normalizedNorm (supportTiny / scale)
  ⟨scale, normalized, residual, normalizedNorm, residualNorm, limit,
    normGreater residualNorm limit⟩

def sourceAny {d : ℕ} (norm : NormOperation d)
    (projector : Matrix (Fin d) (Fin d) ℝ) (rows : List (Row d)) : Bool :=
  (rows.map (fun row => (evaluateRow norm projector row).tooLarge)).any id

def validateSupport {d : ℕ} (norm : NormOperation d)
    (projector : Matrix (Fin d) (Fin d) ℝ) (rows : List (Row d)) : Check Unit :=
  if d = 0 then .error (.fieldFailure "np.max empty last axis")
  else if sourceAny norm projector rows then
    .error (.sourceReject "frozen covariance identifiable support")
  else .ok ()

theorem source_any_is_literal_row_comparison {d : ℕ} (norm : NormOperation d)
    (projector : Matrix (Fin d) (Fin d) ℝ) (rows : List (Row d)) :
    sourceAny norm projector rows = rows.any (fun row =>
      normGreater (norm (rowResidual projector (normalizedRow row)))
        (normLimit (norm (normalizedRow row)) (supportTiny / rowScale row))) := by
  simp [sourceAny, evaluateRow]

theorem source_any_append {d : ℕ} (norm : NormOperation d)
    (projector : Matrix (Fin d) (Fin d) ℝ) (first second : List (Row d)) :
    sourceAny norm projector (first ++ second) =
      (sourceAny norm projector first || sourceAny norm projector second) := by
  simp [sourceAny]

theorem source_any_empty {d : ℕ} (norm : NormOperation d)
    (projector : Matrix (Fin d) (Fin d) ℝ) : sourceAny norm projector [] = false := rfl

theorem source_any_has_computed_row_witness {d : ℕ} (norm : NormOperation d)
    (projector : Matrix (Fin d) (Fin d) ℝ) (rows : List (Row d)) :
    sourceAny norm projector rows = true ↔
      ∃ row ∈ rows, normGreater (norm (rowResidual projector (normalizedRow row)))
        (normLimit (norm (normalizedRow row)) (supportTiny / rowScale row)) = true := by
  simp [source_any_is_literal_row_comparison]

theorem append_cannot_remove_a_computed_refusal {d : ℕ} (norm : NormOperation d)
    (projector : Matrix (Fin d) (Fin d) ℝ) (first second : List (Row d)) :
    sourceAny norm projector first = true → sourceAny norm projector (first ++ second) = true := by
  intro h
  simp [source_any_append, h]

theorem prepend_cannot_remove_a_computed_refusal {d : ℕ} (norm : NormOperation d)
    (projector : Matrix (Fin d) (Fin d) ℝ) (first second : List (Row d)) :
    sourceAny norm projector second = true → sourceAny norm projector (first ++ second) = true := by
  intro h
  simp [source_any_append, h]

theorem containing_a_refused_subset_refuses {d : ℕ} (norm : NormOperation d)
    (projector : Matrix (Fin d) (Fin d) ℝ) (small large : List (Row d))
    (hsubset : ∀ row ∈ small, row ∈ large) :
    sourceAny norm projector small = true → sourceAny norm projector large = true := by
  intro h
  obtain ⟨row, hr, hbad⟩ := (source_any_has_computed_row_witness norm projector small).1 h
  exact (source_any_has_computed_row_witness norm projector large).2 ⟨row, hsubset row hr, hbad⟩

theorem source_any_permutation_invariant {d : ℕ} (norm : NormOperation d)
    (projector : Matrix (Fin d) (Fin d) ℝ) (first second : List (Row d))
    (h : first.Perm second) : sourceAny norm projector first = sourceAny norm projector second := by
  apply Bool.eq_iff_iff.mpr
  simp only [source_any_has_computed_row_witness]
  constructor
  · rintro ⟨row, hr, hb⟩
    exact ⟨row, (h.mem_iff).1 hr, hb⟩
  · rintro ⟨row, hr, hb⟩
    exact ⟨row, (h.mem_iff).2 hr, hb⟩

theorem positive_width_empty_batch_passes (d : ℕ) (hd : 0 < d) (norm : NormOperation d)
    (projector : Matrix (Fin d) (Fin d) ℝ) : validateSupport norm projector [] = .ok () := by
  simp [validateSupport, Nat.ne_of_gt hd, source_any_empty]

theorem zero_width_is_empty_reduction_failure (norm : NormOperation 0)
    (projector : Matrix (Fin 0) (Fin 0) ℝ) (rows : List (Row 0)) :
    validateSupport norm projector rows = .error (.fieldFailure "np.max empty last axis") := by
  simp [validateSupport]

theorem append_refusal_is_an_or {d : ℕ} (hd : 0 < d) (norm : NormOperation d)
    (projector : Matrix (Fin d) (Fin d) ℝ) (first second : List (Row d)) :
    validateSupport norm projector (first ++ second) =
      .error (.sourceReject "frozen covariance identifiable support") ↔
    validateSupport norm projector first = .error (.sourceReject "frozen covariance identifiable support") ∨
    validateSupport norm projector second = .error (.sourceReject "frozen covariance identifiable support") := by
  simp [validateSupport, Nat.ne_of_gt hd, source_any_append]
  cases sourceAny norm projector first <;> cases sourceAny norm projector second <;> simp

theorem accepted_batch_subsets_pass {d : ℕ} (hd : 0 < d) (norm : NormOperation d)
    (projector : Matrix (Fin d) (Fin d) ℝ) (small large : List (Row d))
    (hsubset : ∀ row ∈ small, row ∈ large)
    (hlarge : validateSupport norm projector large = .ok ()) :
    validateSupport norm projector small = .ok () := by
  have hn : sourceAny norm projector large = false := by
    simpa [validateSupport, Nat.ne_of_gt hd] using hlarge
  have hs : sourceAny norm projector small = false := by
    cases hsmall : sourceAny norm projector small with
    | false => rfl
    | true =>
      have hbad := containing_a_refused_subset_refuses norm projector small large hsubset hsmall
      simp [hn] at hbad
  simp [validateSupport, Nat.ne_of_gt hd, hs]

theorem nan_comparison_is_not_a_new_refusal (limit : NormResult) :
    normGreater .nan limit = false := by cases limit <;> rfl

theorem support_tiny_positive : 0 < supportTiny := by
  exact pow_pos (by norm_num : (0 : ℝ) < 1 / 2) _

theorem max_fold_dominates_seed (seed : ℝ) (values : List ℝ) :
    seed ≤ values.foldl max seed := by
  induction values generalizing seed with
  | nil => exact le_rfl
  | cons value values ih =>
    exact le_trans (le_max_left seed value) (ih (max seed value))

theorem max_fold_dominates_member (seed : ℝ) (values : List ℝ) (value : ℝ)
    (h : value ∈ values) : value ≤ values.foldl max seed := by
  induction values generalizing seed with
  | nil => simp at h
  | cons head tail ih =>
    rcases List.mem_cons.mp h with rfl | ht
    · exact le_trans (le_max_right seed value) (max_fold_dominates_seed (max seed value) tail)
    · exact ih (max seed head) ht

theorem row_max_abs_dominates_every_coordinate {d : ℕ} (row : Row d) (j : Fin d) :
    |row j| ≤ rowMaxAbs row := by
  apply max_fold_dominates_member
  simp only [List.mem_ofFn]
  exact ⟨j, rfl⟩

theorem row_scale_positive {d : ℕ} (row : Row d) : 0 < rowScale row :=
  lt_of_lt_of_le support_tiny_positive (le_max_right _ _)

theorem exact_normalized_coordinates_are_bounded {d : ℕ} (row : Row d) (j : Fin d) :
    |normalizedRow row j| ≤ 1 := by
  have hs := row_scale_positive row
  rw [normalizedRow, abs_div, abs_of_pos hs]
  exact (div_le_one hs).2 (le_trans (row_max_abs_dominates_every_coordinate row j) (le_max_left _ _))

theorem normalized_residual_is_scaled_raw_residual {d : ℕ}
    (projector : Matrix (Fin d) (Fin d) ℝ) (row : Row d) :
    rowResidual projector (normalizedRow row) = fun j => rowResidual projector row j / rowScale row := by
  funext j
  simp only [rowResidual, normalizedRow, rowProject]
  rw [show (∑ k, row k / rowScale row * projector k j) =
      (∑ k, row k * projector k j) / rowScale row by
    simp_rw [div_mul_eq_mul_div]
    rw [Finset.sum_div]]
  exact (sub_div _ _ _).symm

theorem vector_norm_positive_divisor {d : ℕ} (row : Row d) (scale : ℝ) (hs : 0 < scale) :
    vectorNorm (fun j => row j / scale) = vectorNorm row / scale := by
  simp only [vectorNorm, div_pow, ← Finset.sum_div]
  rw [Real.sqrt_div, Real.sqrt_sq (le_of_lt hs)]
  exact Finset.sum_nonneg (fun _ _ => sq_nonneg _)

def rawRowViolation {d : ℕ} (projector : Matrix (Fin d) (Fin d) ℝ) (row : Row d) : Prop :=
  vectorNorm (rowResidual projector row) > max (vectorNorm row) supportTiny * supportTolerance

theorem exact_normalized_comparison_preserves_relative_geometry {d : ℕ}
    (projector : Matrix (Fin d) (Fin d) ℝ) (row : Row d) :
    (evaluateRow exactNorm projector row).tooLarge = true ↔ rawRowViolation projector row := by
  have hs := row_scale_positive row
  have hnorm := vector_norm_positive_divisor row (rowScale row) hs
  have hres := vector_norm_positive_divisor (rowResidual projector row) (rowScale row) hs
  have hmax : max (vectorNorm row / rowScale row) (supportTiny / rowScale row) =
      max (vectorNorm row) supportTiny / rowScale row := by
    by_cases h : vectorNorm row ≤ supportTiny
    · rw [max_eq_right ((div_le_div_iff_of_pos_right hs).2 h), max_eq_right h]
    · have hr : supportTiny ≤ vectorNorm row := le_of_not_ge h
      rw [max_eq_left ((div_le_div_iff_of_pos_right hs).2 hr), max_eq_left hr]
  simp only [evaluateRow, exactNorm, normLimit, normGreater, decide_eq_true_eq]
  rw [normalized_residual_is_scaled_raw_residual, hres]
  change vectorNorm (rowResidual projector row) / rowScale row >
    max (vectorNorm (normalizedRow row)) (supportTiny / rowScale row) * supportTolerance ↔ _
  rw [show vectorNorm (normalizedRow row) = vectorNorm row / rowScale row by exact hnorm, hmax]
  rw [div_mul_eq_mul_div]
  change max (vectorNorm row) supportTiny * supportTolerance / rowScale row <
      vectorNorm (rowResidual projector row) / rowScale row ↔
    max (vectorNorm row) supportTiny * supportTolerance < vectorNorm (rowResidual projector row)
  exact div_lt_div_iff_of_pos_right hs

theorem exact_supported_row_has_no_computed_violation {d : ℕ}
    (projector : Matrix (Fin d) (Fin d) ℝ) (row : Row d)
    (h : rowResidual projector row = 0) : (evaluateRow exactNorm projector row).tooLarge = false := by
  have hbound : 0 ≤ max (vectorNorm row) supportTiny * supportTolerance :=
    mul_nonneg (le_trans (Real.sqrt_nonneg _) (le_max_left _ _))
      (by norm_num [supportTolerance])
  have hn : ¬rawRowViolation projector row := by
    simp only [rawRowViolation, h, vectorNorm, Pi.zero_apply, zero_pow (by decide : (2 : ℕ) ≠ 0),
      Finset.sum_const_zero, Real.sqrt_zero]
    exact not_lt_of_ge hbound
  cases hb : (evaluateRow exactNorm projector row).tooLarge with
  | false => rfl
  | true => exact False.elim (hn ((exact_normalized_comparison_preserves_relative_geometry projector row).1 hb))

def axisProjector : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, 0]
def unsupportedRow : Row 2 := ![0, 1]
def largeSupportedRow : Row 2 := ![100000000, 0]

theorem unsupported_row_residual_is_itself : rowResidual axisProjector unsupportedRow = unsupportedRow := by
  funext j
  fin_cases j <;> norm_num [rowResidual, rowProject, axisProjector, unsupportedRow, Fin.sum_univ_two]

theorem unsupported_row_has_computed_violation :
    (evaluateRow exactNorm axisProjector unsupportedRow).tooLarge = true := by
  apply (exact_normalized_comparison_preserves_relative_geometry axisProjector unsupportedRow).2
  rw [rawRowViolation, unsupported_row_residual_is_itself]
  norm_num [vectorNorm, unsupportedRow, Fin.sum_univ_two,
    max_eq_left support_tiny_below_one, supportTolerance]

theorem large_supported_row_has_zero_residual : rowResidual axisProjector largeSupportedRow = 0 := by
  funext j
  fin_cases j <;> norm_num [rowResidual, rowProject, axisProjector, largeSupportedRow, Fin.sum_univ_two]

theorem large_supported_row_alone_passes :
    validateSupport exactNorm axisProjector [largeSupportedRow] = .ok () := by
  have h := exact_supported_row_has_no_computed_violation axisProjector largeSupportedRow
    large_supported_row_has_zero_residual
  simp [validateSupport, sourceAny, h]

theorem unsupported_row_alone_refuses :
    validateSupport exactNorm axisProjector [unsupportedRow] =
      .error (.sourceReject "frozen covariance identifiable support") := by
  simp [validateSupport, sourceAny, unsupported_row_has_computed_violation]

theorem unsupported_row_plus_arbitrary_append_refuses (more : List (Row 2)) :
    validateSupport exactNorm axisProjector (unsupportedRow :: more) =
      .error (.sourceReject "frozen covariance identifiable support") := by
  simp [validateSupport, sourceAny, unsupported_row_has_computed_violation]

theorem unsupported_plus_large_supported_refuses :
    validateSupport exactNorm axisProjector [unsupportedRow, largeSupportedRow] =
      .error (.sourceReject "frozen covariance identifiable support") :=
  unsupported_row_plus_arbitrary_append_refuses [largeSupportedRow]

theorem historical_batch_dilution_and_current_row_refusal_are_distinct :
    OperatorGeometry.batchAccepted axisBasis dilutedNullRow ∧
    validateSupport exactNorm axisProjector [unsupportedRow, largeSupportedRow] =
      .error (.sourceReject "frozen covariance identifiable support") :=
  ⟨extra_large_row_admits_same_null_row, unsupported_plus_large_supported_refuses⟩

end

end N2P3.OperatorRowSupport
