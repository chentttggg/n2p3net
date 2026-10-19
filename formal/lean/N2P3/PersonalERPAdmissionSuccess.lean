import N2P3.PersonalERPPublicInput

/-!
# Complete personal ERP constructor from independent fields and atomic facts

This reaches an actual successful Predictor, rather than an unresolved ERP
continuation. Coefficients/intercept, dimensions, selector, recipe fields and
per-tensor clone/NumPy/finiteness operations are independent inputs. No whole
reader/constructor/validator-success premise or fitted-score assumption occurs.
This known-object success route does not establish the old AdmissionRuntime's
general malformed-artifact/NumPy exception priority; the fuller constructor
connection separately preserves all NumPy calls before artifact mapping reads.
-/

namespace N2P3.PersonalERPAdmissionSuccess
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalState
open N2P3.PersonalERPData N2P3.PersonalERPScalar N2P3.PersonalERPAdmission
open N2P3.PersonalERPRecipe N2P3.PersonalERPScalarFields N2P3.PersonalERPAdmissionRuntime

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def selector (indices : List ℕ) : TorchTensor :=
  ⟨"torch.int64",⟨.integer,[indices.length],indices.map (fun i : ℕ => Atom.integer (i : ℤ)),by simp⟩⟩

def coefficientTorch {d : ℕ} (values : Fin d → ℝ) : TorchTensor :=
  ⟨"torch.float64",PersonalERPImageOps.image1 .floating (fun i => Atom.real (values i))⟩

def interceptTorch (value : ℝ) : TorchTensor :=
  ⟨"torch.float64",⟨.floating,[],[.real value],rfl⟩⟩

def torchState {d : ℕ} (indices : List ℕ) (coefficient : Fin d → ℝ) (intercept : ℝ) : TorchState :=
  ⟨[("waveform_channels",selector indices),("erp.coefficient",coefficientTorch coefficient),
    ("erp.intercept",interceptTorch intercept)],by simp⟩

theorem strict_selector_atoms (indices : List ℕ) :
    (indices.map (fun i : ℕ => Atom.integer (i : ℤ))).mapM
      (fun atom => do let i ← liftCheck (strictInteger atom); pure i.toNat)=.ok indices := by
  rw [List.mapM_map]
  change indices.mapM (fun i : ℕ => (Except.ok i : Result ℕ))=.ok indices
  induction indices with
  | nil => rfl
  | cons head tail ih => simp [ih]

theorem selected_from_independent_fields (p : PersonalERPAdmissionRuntime.Primitives)
    (indices : List ℕ) (contract : Fields) {d : ℕ} (coefficient : Fin d → ℝ) (intercept : ℝ) (C : ℕ)
    (hn : indices ≠ []) (hc : contract.get "n_channels"=some (.integer C))
    (hu : p.selectedUniqueCount (selector indices)=.ok indices.length)
    (hneg : p.selectedAnyNegative (selector indices)=.ok false)
    (hrange : p.selectedAnyTooHigh (selector indices) (.integer C)=.ok false) :
    PersonalERPAdmissionRuntime.selected p (torchState indices coefficient intercept).entries contract=.ok indices := by
  have hm := strict_selector_atoms indices
  simp only [selector] at hu hneg hrange
  simp [PersonalERPAdmissionRuntime.selected,torchState,requiredTorch,lookupTorch,selector,
    require,liftCheck,PersonalERPModelRuntime.required,StateLedger.required,hc,hu,hneg,hrange,hn]
  simpa only [List.mapM_map,Function.comp_def] using hm

theorem current_reader_from_independent_linear_fields (p : PersonalERPAdmissionRuntime.Primitives)
    (f : ERPFields) (indices : List ℕ) (T : ℕ) (baseline : Value)
    (coefficient : Fin (indices.length*(T/f.width)) → ℝ) (intercept : ℝ)
    (hw : 0 < f.width) (hn : indices ≠ []) (ht : 0 < T) (hb : 0 < T/f.width)
    (hf : f.family ≠ .temporal ∧ f.family ≠ .xdawn)
    (hbase : f.family ≠ .toeplitz ∨ baseline=.text "none") :
    CurrentERPState.fromState p.original.temporal p.original.reference
      (plainRecord f.family f.width 4 indices.length T baseline) (linearState coefficient intercept)=
      .ok (plainModel f indices.length T 4 baseline coefficient intercept) := by
  unfold CurrentERPState.fromState
  rw [plain_family_constructs p.original.temporal p.original.reference f.family f.width 4
    indices.length T baseline coefficient intercept hw (by norm_num) (List.length_pos_iff.mpr hn)
    ht hb hf hbase]
  exact CurrentERPState.other_families_current_guard_preserves _ hf.2

theorem numpy_state_from_individual_conversions (p : PersonalERPAdmission.Primitives) (indices : List ℕ)
    {d : ℕ} (coefficient : Fin d → ℝ) (intercept : ℝ)
    (hcoefficient : p.numpy (coefficientTorch coefficient)=.ok (coefficientTensor coefficient))
    (hintercept : p.numpy (interceptTorch intercept)=.ok (interceptTensor intercept)) :
    erpNumpyState p (torchState indices coefficient intercept).entries=.ok (linearState coefficient intercept) := by
  simp [torchState,erpNumpyState,stripERP,hcoefficient,hintercept,linearState]

theorem final_state_finite_from_each_tensor (p : PersonalERPAdmission.Primitives)
    (state : List (String × TorchTensor)) (finite : ∀ entry ∈ state, p.tensorIsFinite entry.2=.ok true) :
    allStateFinite p state=.ok () := by
  induction state with
  | nil => rfl
  | cons head tail ih =>
    have hh := finite head (by simp)
    have ht : ∀ entry ∈ tail, p.tensorIsFinite entry.2=.ok true := by
      intro entry member
      exact finite entry (by simp [member])
    simp [allStateFinite,hh,ih ht,ensure]

theorem complete_constructor_from_independent_fields_and_atoms
    (p : PersonalERPAdmissionRuntime.Primitives) (f : ERPFields) (checks : Conditions f)
    (atoms : ScalarAtoms p.recipeScalar f.raw) (authority : String) (contract rawArgs : Fields)
    (C T : ℕ) (indices : List ℕ) (baseline : Value)
    (coefficient : Fin (indices.length*(T/f.width)) → ℝ) (intercept : ℝ) (requested device : String)
    (hAuthority : authority ≠ "")
    (hArgs : contract.get "preprocessing"=some (.object rawArgs))
    (hKnown : fieldKeys rawArgs ⊆ preprocessingKeywords)
    (hChannels : contract.get "n_channels"=some (.integer C))
    (hTimes : contract.get "n_times"=some (.integer T))
    (hBaseline : rawArgs.get "baseline_mode"=some baseline)
    (hn : indices ≠ []) (ht : 0 < T) (hb : 0 < T/f.width)
    (hf : f.family ≠ .temporal ∧ f.family ≠ .xdawn)
    (hbase : f.family ≠ .toeplitz ∨ baseline=.text "none")
    (hJSON : p.original.finiteJSON (.object (noQualityMetadata authority contract (recordFields f.raw)
      (plainRecord f.family f.width 4 indices.length T baseline)))=
      .ok (.object (noQualityMetadata authority contract (recordFields f.raw)
        (plainRecord f.family f.width 4 indices.length T baseline))))
    (hCopies : ∀ entry ∈ (torchState indices coefficient intercept).entries,
      p.original.detachCPUClone entry.2=.ok entry.2)
    (hDevice : p.original.device requested=.ok device)
    (hUnique : p.selectedUniqueCount (selector indices)=.ok indices.length)
    (hNegative : p.selectedAnyNegative (selector indices)=.ok false)
    (hRange : p.selectedAnyTooHigh (selector indices) (.integer C)=.ok false)
    (hCoefficient : p.original.numpy (coefficientTorch coefficient)=.ok (coefficientTensor coefficient))
    (hIntercept : p.original.numpy (interceptTorch intercept)=.ok (interceptTensor intercept))
    (hFinite : ∀ entry ∈ (torchState indices coefficient intercept).entries,
      p.original.tensorIsFinite entry.2=.ok true) :
    constructERP p (.object (noQualityMetadata authority contract (recordFields f.raw)
      (plainRecord f.family f.width 4 indices.length T baseline)))
      (torchState indices coefficient intercept) requested=
      .ok ⟨noQualityMetadata authority contract (recordFields f.raw)
        (plainRecord f.family f.width 4 indices.length T baseline),
        (torchState indices coefficient intercept).entries,device,none,
        ⟨f.recipe,indices,plainModel f indices.length T 4 baseline coefficient intercept⟩⟩ := by
  rw [prefix_from_independent_fields_reaches_actual_ERP_continuation p authority contract
    (recordFields f.raw) (plainRecord f.family f.width 4 indices.length T baseline) rawArgs
    (torchState indices coefficient intercept) requested device hAuthority hArgs hKnown hJSON hCopies hDevice]
  have hs := selected_from_independent_fields p indices contract coefficient intercept C hn hChannels hUnique hNegative hRange
  have hnumpy := numpy_state_from_individual_conversions p.original indices coefficient intercept hCoefficient hIntercept
  have hreader := current_reader_from_independent_linear_fields p f indices T baseline coefficient intercept
    checks.width hn ht hb hf hbase
  have hrecipe := canonical_kwargs_from_independent_fields_and_atoms p.recipeScalar f checks hf.1 atoms
  have hfinite := final_state_finite_from_each_tensor p.original _ hFinite
  simp [finishERP,noQualityMetadata,PersonalERPModelRuntime.required,PersonalERPModelRuntime.text,
    PersonalERPModelRuntime.objectAt,StateLedger.required,PersonalERPExecution.text,ERPState.objectAt,
    Fields.ofList,hs,hnumpy,hreader,hrecipe,hTimes,hBaseline,
    plainModel,ERPFields.recipe,usesSource,modelKind,require,liftCheck,hfinite,
    PersonalAdmission.PyEq,PersonalAdmission.normalizeValue,fieldDefault,Fields.get]

end
end N2P3.PersonalERPAdmissionSuccess
