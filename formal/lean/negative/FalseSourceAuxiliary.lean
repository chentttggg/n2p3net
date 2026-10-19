import N2P3.SourceAuxiliaryChain

/-! Source-first wrong alternatives with inhabited finite sample sets. -/
namespace N2P3.NegativeSourceAuxiliary
open N2P3.SourceAuxiliaryRisk N2P3.SourceAuxiliaryMMD
open scoped BigOperators
noncomputable section

-- A repeated physical row's last occurrence is not retained.
example : First (fun _ : Fin 2 => 7) 1 := by
  have h := later_duplicate_excluded (fun _ : Fin 2 => 7) 0 1 (by decide) rfl
  simp only [h]

-- Cross-domain alone does not suffice: participant is the same.
example : Positive (fun _ : Fin 2 => True) (fun _ => 0) (fun i => i.val) (fun _ => 0) 0 1 := by
  norm_num [Positive]

-- Cross-person alone does not suffice: domain is the same.
example : Positive (fun _ : Fin 2 => True) (fun _ => 0) (fun _ => 0) (fun i => i.val) 0 1 := by
  norm_num [Positive]

-- A different-class pair remains negative in the same domain and participant.
example : ¬ Negative (fun _ : Fin 2 => True) (fun i => i.val) 0 1 := by
  norm_num [Negative]

example : normalizedLength 0 (1/10) = 1 := by
  norm_num [normalizedLength]

example : normalizedLength (1/20) (1/10) = 1 := by
  norm_num [normalizedLength]

-- An active singleton candidate and positive has zero loss.
example : anchorRisk (fun _ : Fin 1 => (3:ℝ)) Finset.univ ≠ 0 := by
  rw [(singleton_anchor_is_active_zero 3).1]
  norm_num

-- Empirical off-diagonal MMD for identical nonconstant clouds can be negative.
example : 0 ≤ empiricalMMD
    (fun i j => scalarRBF 1 (twoPoints 1 i) (twoPoints 1 j))
    (fun i j => scalarRBF 1 (twoPoints 1 i) (twoPoints 1 j))
    (fun i j => scalarRBF 1 (twoPoints 1 i) (twoPoints 1 j)) := by
  have h := not_le_of_gt (actual_identical_two_point_rbf_mmd_negative 1 1 (by norm_num) (by norm_num))
  simp only [h]

-- Inactive cells cannot double the denominator.
example : activeMean ({0} : Finset (Fin 2)) (fun _ => (2:ℝ)) = 1 := by
  norm_num [activeMean]

-- Proposal correction may not omit division by the positive proposal mass.
example : (∑ i : Fin 2, (if i=0 then (1/4:ℝ) else 3/4) * (1/2) *
    (if i=0 then 1 else 4)) = ∑ i : Fin 2, (1/2:ℝ) * (if i=0 then 1 else 4) := by
  norm_num [Fin.sum_univ_two]

end
end N2P3.NegativeSourceAuxiliary
