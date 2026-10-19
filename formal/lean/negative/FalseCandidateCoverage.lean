import N2P3.CandidateCoverage

/-! Five deliberate false contracts. Each must be rejected at a `False` goal. -/

namespace N2P3.NegativeCandidateCoverage

open N2P3.CandidateCoverage

-- Positive membership for all 36 hypotheses does not certify twelve flash counts.
example : membershipReason 6 36 (physicalMatrix onlyRows) = none → PhysicalComplete onlyRows := by
  intro _
  simp only [only_rows_physically_incomplete]

-- Even 120 events and twenty positive events per cell do not imply 10 of each flash.
example : (∀ cell, columnCount (physicalMatrix biasedSchedule) cell = 20) →
    PhysicalComplete biasedSchedule := by
  intro _
  simp only [biased_physical_incomplete]

-- Surplus physical events must also be rejected; an empty missing ledger is insufficient.
example : missingKeys (fun _ : Flash => 10) (fun _ => some 11) = ∅ →
    Complete (fun _ : Flash => 10) (fun _ => some 11) := by
  intro _
  simp only [all_eleven_is_incomplete]

-- When all four predicates fail, incomplete coverage precedes the scope refusal.
example : allFailureGeometry.reason (fun _ => some 0) = some .outsideScope := by
  rw [all_failures_keep_coverage_priority.2.2.2]
  have hne : ViewRefusal.partialCoverage ≠ ViewRefusal.outsideScope := by decide
  simp only [Option.some.injEq, hne]

-- Offline whole-scope filtering does not make a view available at its own stop.
example : delayedGeometry.available = delayedGeometry.stop := by
  norm_num [delayedGeometry, Geometry.available, Geometry.stop]

end N2P3.NegativeCandidateCoverage
