import Mathlib.Order.Interval.Set.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.NormNum

/-!
Deliberately false authorization/support statements. The verifier must reject
this file. The accepted RawScope module proves the corresponding counterexamples.
-/

namespace N2P3.RawScope.Negative

/-- False: the closed source-grid final sample can be omitted from extraction. -/
example : (1 : ℕ) ∉ Set.Ico 0 (1 + 1) := by
  norm_num

/-- False: a calibration epoch tail always excludes the next unlabelled flash. -/
example : ¬ ((1 : ℕ) < 2) := by
  norm_num

/-- False: zeros appended beyond observed support preserve the observed mean. -/
example : (2 : ℝ) / 1 = (2 + 0) / 2 := by
  norm_num

end N2P3.RawScope.Negative
