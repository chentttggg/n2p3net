import N2P3.SelectionERPCaller

/-! Source-first wrong alternatives. The numeric-operation fixture below has
only the independently fixed identity/zero covariance operation domain; it is
not an implementation or a fitted SelectionERP model. -/
namespace N2P3.FalseSelectionERP
open N2P3.SelectionERPSupport N2P3.SelectionERPEvidence N2P3.SelectionERPCaller
open N2P3.CandidateCoverage N2P3.CandidateTie
open scoped BigOperators
noncomputable section

example : stopExclusive-firstOffset=357 := by norm_num [stopExclusive,firstOffset]

example : (windowIndex constructedGeometry (Fin.last 119) (Fin.last 357)).val=1429 := by
  rw [the_last_raw_window_coordinate_is_plus306]
  norm_num [constructedGeometry]

example : (Fintype.card (Fin 178))=179 := by norm_num

example : 0 ≤ physicalTime 25 := by norm_num [physicalTime]

example : physicalTime 26 < 0 := by norm_num [physicalTime]

example : N2P3.ERPFeatures.retainedSamples 178 4=178 := by
  norm_num [N2P3.ERPFeatures.retainedSamples]

example : (finProdFinEquiv ((1 : Fin 8),(0 : Fin 28))).val=1 := by
  norm_num [finProdFinEquiv]

example : (N2P3.ERPFeatures.sourceColumn ((1 : Fin 44),(1 : Fin 8))).val=45 := by
  norm_num [N2P3.ERPFeatures.sourceColumn,finProdFinEquiv]

example : familyBaselineAllowed .toeplitz .meanOnly := by simp [familyBaselineAllowed]

example : familyBaselineAllowed .xdawn .none := by simp [familyBaselineAllowed]

example : completeCalibrationFlag .numericZero=true := by norm_num [completeCalibrationFlag]

example : (∑ _group ∈ trainingGroups (0 : Fin 6),(120:ℕ))=800 := by
  rw [(actual_e2_complete_group_label_accounting 0).1]
  norm_num

example : physicalEvidence indexedCompleteSchedule (fun _ => 0) 1 0=-1 := by
  rw [physical_complete_llr_prior_is_one_common_twenty_event_offset _ actual120_position_witness_is_physically_complete]
  norm_num [physicalEvidence,evidence,N2P3Pilot.membershipScore]

-- A valid membership matrix with unequal candidate counts can reverse order.
def unequalMembership : Fin 3 → Fin 2 → ℤ := !![1,0;1,0;0,1]

example : evidence unequalMembership (fun _ => 1) 2 0 > evidence unequalMembership (fun _ => 1) 2 1 := by
  norm_num [evidence,N2P3Pilot.membershipScore,unequalMembership,Fin.sum_univ_succ]

example : PhysicalComplete onlyRows := by
  simp only [only_rows_physically_incomplete]

example : (readout nearTieScores).winnerIndex=some 1 := by
  change winner nearTieScores sourceTolerance=some 1
  rw [nearTie_abstains]
  simp

-- Covariance before tangent coordinates is a real source operation boundary.
def fixtureOps : NumericalOperations := by
  classical
  exact {
    fft20Hz := fun _ _ => 0
    cast32 := id
    choleskyAdmission := fun covariance => if covariance=1 then .ok () else .error .covariance
    riemannTangent := fun reference covariance => if reference=1 ∧ covariance=1 then .ok (fun _ => 0) else .error .tangent
  }

example : xdawnLogit fixtureOps pulseXDawnState pulseXDawnEpoch=.ok 0 := by
  rw [nonempty_xdawn_affine_output_from_independent_identity_cholesky_and_tangent_atoms fixtureOps
    (by simp [fixtureOps]) (by simp [fixtureOps])]
  simp

example : oasCovariance (0 : Matrix (Fin 16) (Fin 178) ℝ)=1 := by
  rw [actual_time_constant_augmented_population_has_zero_oas_covariance]
  have different : (0 : Matrix (Fin 16) (Fin 16) ℝ) ≠ 1 := zero_ne_one
  simp only [different]

def recordFixture : PhysicalRecord 1429 119 8 := ⟨constructedGeometry,0,fun _ => false⟩
def zeroFFT : TraceFFT := fun _ _ => 0

example : scorePhysicalRecord zeroFFT fixtureOps (.toeplitz (zeroToeplitz 8 0)) .none 1
    {recordFixture with targetMask := fun _ => true}
    (indexedPhysicalMatrix indexedCompleteSchedule) 0 ≠
      scorePhysicalRecord zeroFFT fixtureOps (.toeplitz (zeroToeplitz 8 0)) .none 1
        recordFixture (indexedPhysicalMatrix indexedCompleteSchedule) 0 := by
  rw [actual_prediction_arguments_exclude_the_held_target_mask]
  simp

def frozenWord : List String := (List.range 36).map (fun index => toString index)
def frozenContext : FrozenContext :=
  ⟨256,["c0","c1","c2","c3","c4","c5","c6","c7"],frozenWord,["calibration"],[]⟩
def shortenedWordContext : PredictionContext :=
  ⟨256,frozenContext.channels,frozenWord.take 35,"held","opaque_actual_digest"⟩

example : scoreWithContext (fun _ => true) frozenContext shortenedWordContext zeroFFT fixtureOps
    (.toeplitz (zeroToeplitz 8 0)) .none (fun _ : Fin 120 => fun _ _ => 0)
    (indexedPhysicalMatrix indexedCompleteSchedule) 0=.ok (readout (fun _ => 0)) := by
  have different : shortenedWordContext.vocabulary ≠ frozenContext.vocabulary := by
    intro same
    have lengths := congrArg List.length same
    norm_num [shortenedWordContext,frozenContext,frozenWord] at lengths
  rw [changing_frozen_word_or_order_refuses_before_the_numeric_score _ _ _ _ _ _ _ _ _ _ different]
  simp

-- A later covariance refusal blocks the entire tangent batch; row-wise
-- interleaving must not expose an earlier row's hypothetical tangent error.
def twoCovarianceAdmissions (row : Fin 2) : NumericCheck Unit :=
  if row=0 then .ok () else .error .covariance

example : (do
    let _ ← collectRows 2 twoCovarianceAdmissions
    (Except.error NumericFailure.tangent : NumericCheck ℝ))=.error .tangent := by
  simp [collectRows,twoCovarianceAdmissions,Except.bind,Bind.bind]

end
end N2P3.FalseSelectionERP
