import N2P3Pilot

open N2P3Pilot
open scoped Matrix

-- Deliberately false: counts 2 vs 1, evidence 1, offset 2 reverses the ranking.
example :
    membershipScore (!![1, 0; 1, 0; 0, 1]) (fun _ => true)
      (![1, 1, 1]) (fun _ => 2) (0 : Fin 2) >
    membershipScore (!![1, 0; 1, 0; 0, 1]) (fun _ => true)
      (![1, 1, 1]) (fun _ => 2) (1 : Fin 2) := by
  norm_num [membershipScore, Fin.sum_univ_succ]
