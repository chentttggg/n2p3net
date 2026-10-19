import N2P3.PersonalERPAdmissionRuntime
import N2P3.PersonalERPGuardedViews
import N2P3.PersonalERPRawInputTail

/-!
# Personal input on the actual Timeline-constructor route

Sources: calibrated.py:546-659,752-766. Independent dataset arrays reach their
literal validator before any N/C/T Context exists. The actual constructor and
the input continuation share the execution primitives. Timeline coordinate
facts are obtained separately from its actual constructor, never from an
assumed success of its later validator. This covers constructor-corresponding
timelines only. Arbitrary mutated timelines can validate with unequal unused
field lengths and are NOT covered by this downstream Context. Full raw-session,
arbitrary-mutated-object and fit-producer paths remain separate obligations;
this connection does not close P06.
-/

namespace N2P3.PersonalERPPublicInput
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open N2P3.PersonalERPData N2P3.PersonalERPScalar N2P3.PersonalERPAdmission
open N2P3.PersonalERPArrayIngress

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Operations where
  constructor : PersonalERPAdmissionRuntime.Primitives
  arrays : PersonalERPArrayIngress.Operations
  sharedExecution : arrays.input=constructor.execution

def constructAndValidate (p : Operations) (metadata : Value) (state : TorchState) (device : String)
    (raw : RawDataset) (timeline : ERPTemporalNPZ.Timeline)
    (arrays : timeline.arrays=raw.timeline.arrays) (complete : timeline.complete=raw.timeline.complete)
    (online : timeline.onlineCausal=raw.timeline.onlineCausal) :
    Result (Predictor × (Σ N C T, PersonalERPInput.Context N C T timeline.count)) := do
  let model ← PersonalERPAdmissionRuntime.constructERP p.constructor metadata state device
  let target ← PersonalERPGuardedViews.validateInput p.arrays model.metadata (predictorBasis model)
    raw timeline arrays complete online
  pure (model,target)

/-- Broader public ndarray route. This retains arbitrary raw timeline fields
and has no Timeline-constructor or common-length premise. -/
def constructAndValidateRaw (p : Operations) (metadata : Value) (state : TorchState) (device : String)
    (raw : RawDataset) : Result (Predictor × PersonalERPRawInputTail.State raw) := do
  let model ← PersonalERPAdmissionRuntime.constructERP p.constructor metadata state device
  let target ← PersonalERPRawInputTail.validateInput p.arrays model.metadata (predictorBasis model) raw
  pure (model,target)

/-- Independent constructor-prefix fields feed the real ERP continuation,
then the real raw-array input route. Neither constructor nor validator
success is a hypothesis. Primitive JSON/copy/device facts stay individual. -/
theorem independent_fields_feed_actual_constructor_and_raw_input (p : Operations)
    (authority : String) (contract recipe artifact rawArgs : Fields) (state : TorchState)
    (requested actualDevice : String) (raw : RawDataset) (timeline : ERPTemporalNPZ.Timeline)
    (arrays : timeline.arrays=raw.timeline.arrays) (complete : timeline.complete=raw.timeline.complete)
    (online : timeline.onlineCausal=raw.timeline.onlineCausal)
    (hAuthority : authority ≠ "") (hArgs : contract.get "preprocessing"=some (.object rawArgs))
    (hKnown : fieldKeys rawArgs ⊆ preprocessingKeywords)
    (hJSON : p.constructor.original.finiteJSON
      (.object (PersonalERPAdmissionRuntime.noQualityMetadata authority contract recipe artifact))=
      .ok (.object (PersonalERPAdmissionRuntime.noQualityMetadata authority contract recipe artifact)))
    (hCopies : ∀ entry ∈ state.entries, p.constructor.original.detachCPUClone entry.2=.ok entry.2)
    (hDevice : p.constructor.original.device requested=.ok actualDevice) :
    constructAndValidate p (.object (PersonalERPAdmissionRuntime.noQualityMetadata authority contract recipe artifact))
      state requested raw timeline arrays complete online=
      (do
        let model ← PersonalERPAdmissionRuntime.finishERP p.constructor
          (PersonalERPAdmissionRuntime.noQualityMetadata authority contract recipe artifact)
          state.entries actualDevice none contract rawArgs
        let target ← PersonalERPGuardedViews.validateInput p.arrays model.metadata (predictorBasis model)
          raw timeline arrays complete online
        pure (model,target)) := by
  unfold constructAndValidate
  rw [PersonalERPAdmissionRuntime.prefix_from_independent_fields_reaches_actual_ERP_continuation
    p.constructor authority contract recipe artifact rawArgs state requested actualDevice
    hAuthority hArgs hKnown hJSON hCopies hDevice]

end
end N2P3.PersonalERPPublicInput
