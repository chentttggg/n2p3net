import N2P3.ERPTemporalIO

namespace N2P3.NegativeERPTemporalIO

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPTemporalIO

example : (finalState (p:=fixtureIO) fixtureWrite).world.replacements=1 := by
  rw [permission_errors_retry_until_seventh_attempt.1]
  have h : (7 : ℕ) ≠ 1 := by decide
  simp only [h]

example : (finalState (p:=fixtureIO none) (fixtureWrite none)).world.replacements=8 := by
  rw [seventh_permission_failure_has_no_eighth_attempt.1]
  have h : (7 : ℕ) ≠ 8 := by decide
  simp only [h]

example : (finalState (p:=fixtureIO (some 7) (some (.os "fsync")))
    (fixtureWrite (some 7) (some (.os "fsync")))).world.events=
      ["mkdir","temporaryName","open","dumpJSONNoNan","newline","flush","fsync","unlink"] := by
  rw [fsync_failure_closes_then_cleans_without_publication.1]
  have h : (["mkdir","temporaryName","open","dumpJSONNoNan","newline","flush","fsync","close","unlink"] : List String) ≠
      ["mkdir","temporaryName","open","dumpJSONNoNan","newline","flush","fsync","unlink"] := by decide
  simp only [h]

example : (match fixtureWrite (some 7) (some (.os "fsync")) (some (.os "unlink")) with
    | .ok _ _ => none | .error fault _ => some fault)=some (.os "unlink") := by
  change some (Fault.os "fsync")=some (Fault.os "unlink")
  have h : some (Fault.os "fsync") ≠ some (Fault.os "unlink") := by decide
  simp only [h]

example : (match fixtureWrite (some 7) (some (.os "fsync")) (some (.base "unlink interrupt")) with
    | .ok _ _ => none | .error fault _ => some fault)=some (.os "fsync") := by
  change some (Fault.base "unlink interrupt")=some (Fault.os "fsync")
  have h : some (Fault.base "unlink interrupt") ≠ some (Fault.os "fsync") := by decide
  simp only [h]

example : (finalState (p:=fixtureIO (some 1)) (fixtureWrite (some 1))).world.targetOverwritten=false := by
  rw [atomic_replace_does_not_recheck_output_nonexistence]
  simp

end N2P3.NegativeERPTemporalIO
