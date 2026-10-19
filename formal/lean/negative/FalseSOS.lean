import N2P3.FiniteSOS
import Mathlib.Tactic.NormNum

/-!
Deliberately false finite-SOS claims. The verifier must reject this file after
the accepted module is built; an import failure is not a successful control.
-/

namespace N2P3.FiniteSOS.Negative

def identitySection : Section := ⟨1, 0, 0, 0, 0⟩

/-- False: the right endpoint state can always be dropped from two passes. -/
example :
    twoPass (fun _ : Fin 1 => identitySection) (0 : Samples 1) 0
      (fun _ => ![1, 0]) =
    zeroTwoPass (fun _ : Fin 1 => identitySection) 1 0 := by
  funext i
  fin_cases i
  norm_num [twoPass, zeroTwoPass, run, cascade, sectionStep, reverse,
    response, identitySection]

/-- False: a unit delay coordinate is observable in one sample. -/
example :
    (endpointResponse (fun _ : Fin 1 => identitySection) 1
      ((fun _ => ![0, 1]), 0)) 0 ≠ 0 := by
  norm_num [endpointResponse, run, cascade, sectionStep, reverse,
    response, stateResponse, identitySection]

end N2P3.FiniteSOS.Negative
