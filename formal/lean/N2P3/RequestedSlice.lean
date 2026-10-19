import N2P3.RequestedOutcomeCounts
import N2P3.RequestedOutcomeRates
import N2P3.RequestedCoverage
import N2P3.RequestedConstructorFields
import N2P3.RequestedAUC
import N2P3.RequestedNLL
import N2P3.RequestedStudyStatistics

/-! V05 requested-denominator and metric specifications. Native code is
source-audited separately. The actual near-zero coefficient source branch is
false_claim_with_evidence; it is not covered by the exact-zero corollary. -/
