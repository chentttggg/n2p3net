import N2P3.LatencyReadoutComposition
import Mathlib.Tactic.FinCases

/-! Independent L01-L10 alternatives fixed in the source-before-proof contract.
No fixture imports Torch, fits a model or reads EEG. -/
namespace N2P3.LatencyReadoutWitnesses
open N2P3.LatencyWindowBank N2P3.LatencyContrastAttention N2P3.LatencyReadoutComposition
open scoped BigOperators
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

def miniConfig : Config := ⟨1000,0,0,2,0,2,[0,1,2]⟩
def rawConfig : Config := {miniConfig with sfreq := 100, tmin := -1/5}

private lemma miniCandidate0 : candidate miniConfig 3 0={0,1} := by
  ext t
  fin_cases t <;> norm_num [candidate,window,timeMs,miniConfig,Fin.ext_iff]

private lemma miniCandidate1 : candidate miniConfig 3 1={1,2} := by
  ext t
  fin_cases t <;> norm_num [candidate,window,timeMs,miniConfig,Fin.ext_iff]

private lemma miniCandidate2 : candidate miniConfig 3 2={2} := by
  ext t
  fin_cases t <;> norm_num [candidate,window,timeMs,miniConfig,Fin.ext_iff]

theorem first_ST_bin_is_timestamped_at_its_center_instead_of_its_first_sample :
    timeMs (featureConfig rawConfig 4) 0= -185 ∧ timeMs rawConfig 0= -200 := by
  norm_num [timeMs,featureConfig,rawConfig]

theorem the_literal_half_open_window_excludes_its_upper_sample :
    window miniConfig 3 0 2={0,1} ∧ (2 : Fin 3) ∉ window miniConfig 3 0 2 := by
  have setEq : window miniConfig 3 0 2={0,1} := by
    ext t
    fin_cases t <;> norm_num [window,timeMs,miniConfig,Fin.ext_iff]
  exact ⟨setEq,by rw [setEq]; norm_num [Fin.ext_iff]⟩

theorem the_active_bank_keeps_available_two_sample_windows_and_refuses_a_one_sample_reference :
    activeOffsets miniConfig 3=[0,1] ∧
      build {miniConfig with referenceStart := 2, referenceEnd := 3} 3=none := by
  constructor
  · change [0,1,2].filter (fun offset => decide (2 ≤ (candidate miniConfig 3 offset).card))=[0,1]
    norm_num [miniCandidate0,miniCandidate1,miniCandidate2,Fin.ext_iff]
  · apply too_short_reference_refuses_the_bank
    have refEq : reference {miniConfig with referenceStart := 2, referenceEnd := 3} 3={2} := by
      ext t
      fin_cases t <;> norm_num [reference,window,timeMs,miniConfig,Fin.ext_iff]
    rw [refEq]
    norm_num

theorem a_valid_shifted_candidate_can_include_a_sample_outside_the_base_window :
    (2 : Fin 3) ∈ candidate miniConfig 3 1 ∧ (2 : Fin 3) ∉ candidate miniConfig 3 0 := by
  rw [miniCandidate0,miniCandidate1]
  norm_num [Fin.ext_iff]

theorem cached_shape_mismatch_does_not_silently_rebuild_for_another_length :
    weightsFor (some 3) miniConfig 4=none ∧ weightsFor none miniConfig 4=build miniConfig 4 := by
  simp [weightsFor]

def ref4 : Finset (Fin 4) := {0,1}
def candidates4 (_ : Fin 1) : Finset (Fin 4) := {2,3}
def signal4 (_ : Fin 1) (t : Fin 4) : ℝ :=
  if t.val=0 then 1 else if t.val=1 then 3 else if t.val=2 then 10 else 14

private lemma referenceMean (x : Fin 4 → ℝ) : mean ref4 x=(x 0+x 1)/2 := by
  rw [actual_selected_window_mean_is_the_selected_sum_over_its_own_count]
  norm_num [ref4,Fin.ext_iff]

private lemma candidateMean (x : Fin 4 → ℝ) : mean (candidates4 0) x=(x 2+x 3)/2 := by
  rw [actual_selected_window_mean_is_the_selected_sum_over_its_own_count]
  norm_num [candidates4,Fin.ext_iff]

private lemma oneCandidate {C : ℕ} (x : Fin C → Fin 4 → ℝ) (c : Fin C) :
    LatencyContrastAttention.forward ref4 candidates4 x (fun _ => 0) 1 c=
      mean (candidates4 0) (x c)-mean ref4 (x c) := by
  simp [LatencyContrastAttention.forward,zero_query_pools_the_arithmetic_average_of_contrasts,contrast]

theorem the_actual_single_candidate_output_is_evidence_mean_minus_reference_mean :
    mean ref4 (signal4 0)=2 ∧ mean (candidates4 0) (signal4 0)=12 ∧
      LatencyContrastAttention.forward ref4 candidates4 signal4 (fun _ => 0) 1 0=10 := by
  rw [oneCandidate]
  rw [referenceMean,candidateMean,candidateMean,referenceMean]
  norm_num [signal4]

def twoContrasts (k : Fin 2) (_ : Fin 1) : ℝ := (![2,6] : Fin 2 → ℝ) k

theorem zero_query_initialization_averages_distinct_latency_contrasts :
    attention twoContrasts (fun _ => 0) 1 0=1/2 ∧
      attention twoContrasts (fun _ => 0) 1 1=1/2 ∧
      pooled twoContrasts (fun _ => 0) 1 0=4 := by
  simp only [zero_query_initialization_gives_uniform_latency_attention,
    zero_query_pools_the_arithmetic_average_of_contrasts]
  norm_num [twoContrasts,Fin.sum_univ_succ]

theorem actual_temperature_scale_uses_the_square_root_of_channel_count :
    temperatureScale 2 4=4 := by
  have sqrtFour : Real.sqrt (4 : ℝ)=2 := by
    rw [show (4 : ℝ)=(2 : ℝ)^2 by norm_num,Real.sqrt_sq (by norm_num)]
  norm_num [temperatureScale,sqrtFour]

def changedReferenceSignal (_ : Fin 1) (t : Fin 4) : ℝ :=
  if t.val=0 then 3 else if t.val=1 then 5 else if t.val=2 then 10 else 14

theorem reference_only_edits_change_the_actual_pool_even_with_unchanged_evidence :
    LatencyContrastAttention.forward ref4 candidates4 changedReferenceSignal (fun _ => 0) 1 0=8 ∧
      LatencyContrastAttention.forward ref4 candidates4 signal4 (fun _ => 0) 1 0=10 := by
  rw [oneCandidate,oneCandidate]
  rw [candidateMean,referenceMean,candidateMean,referenceMean]
  norm_num [signal4,changedReferenceSignal]

def orderedBranchSignals (branch c : Fin 2) (t : Fin 4) : ℝ :=
  ((branch.val*2+c.val+1 : ℕ) : ℝ)*signal4 0 t

theorem actual_branch_major_channel_concatenation_keeps_different_coordinates :
    packed ref4 candidates4 orderedBranchSignals (fun _ _ => 0) 1 1=20 ∧
      packed ref4 candidates4 orderedBranchSignals (fun _ _ => 0) 1 2=30 := by
  have first := branch_split_and_concatenation_recover_the_same_source_branch_channel
    ref4 candidates4 orderedBranchSignals (fun _ _ => 0) 1 (0 : Fin 2) (1 : Fin 2)
  have second := branch_split_and_concatenation_recover_the_same_source_branch_channel
    ref4 candidates4 orderedBranchSignals (fun _ _ => 0) 1 (1 : Fin 2) (0 : Fin 2)
  have firstIndex : finProdFinEquiv ((0 : Fin 2),(1 : Fin 2))=(1 : Fin 4) := rfl
  have secondIndex : finProdFinEquiv ((1 : Fin 2),(0 : Fin 2))=(2 : Fin 4) := rfl
  rw [firstIndex] at first
  rw [secondIndex] at second
  rw [first,second]
  rw [oneCandidate,oneCandidate]
  rw [candidateMean,referenceMean,candidateMean,referenceMean]
  norm_num [signal4,orderedBranchSignals]

end
end N2P3.LatencyReadoutWitnesses
