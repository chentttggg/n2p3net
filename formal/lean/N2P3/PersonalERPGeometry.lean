import N2P3.PersonalERPSession

/-!
# Literal callback geometry branches for the saved final montage

Source: calibrated.py:736-750; channel.py:70-156,207-263,366-515,581-620.
Only the two actually used resolver routes are modeled: explicit BIDS EEG
coordinates in head frame without a montage, and standard_1005 registration
without ICP/spherical fallback. Library montage IO and each vector/matrix or
float32 operation remain separate numerical boundaries. Neither a complete
layout resolver nor a physical-recording-validity predicate is supplied.
-/

namespace N2P3.PersonalERPGeometry

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPExecution N2P3.PersonalERPRaw N2P3.PersonalERPModels

noncomputable section

attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

abbrev Vector := Fin 3 → ℝ
abbrev Matrix := Fin 3 → Fin 3 → ℝ

structure MontageImage where
  positions : List (String × ArrayImage Atom)
  frame : String
  fiducials : List (String × ArrayImage Atom)

structure Primitives where
  session : PersonalERPSession.Primitives
  loadMontage : String → Check MontageImage
  lower : String → String
  asVector : ArrayImage Atom → Check Vector
  subtract : Vector → Vector → Check Vector
  add : Vector → Vector → Check Vector
  scale : ℝ → Vector → Check Vector
  divide : Vector → ℝ → Check Vector
  dot : Vector → Vector → Check ℝ
  cross : Vector → Vector → Check Vector
  norm : Vector → Check ℝ
  matrixProductTranspose : Matrix → Check Matrix
  allclose : Matrix → Matrix → ℝ → Check Bool
  determinant : Matrix → Check ℝ
  determinantClose : ℝ → Check Bool
  matvec : Matrix → Vector → Check Vector
  float32 : Vector → Check Vector
  embedding : List Vector → Check (ArrayImage Atom)

def canonicalUnique (p : Primitives) (names : List String) : Check (List String) := do
  let names ← names.mapM p.session.clock.bids.models.input.canonicalChannel
  ensure (names.Nodup) "geometry canonical unique channels"
  pure names

def putPosition (name : String) (position : Vector) (positions : List (String × Vector)) : List (String × Vector) :=
  positions.filter (fun row => decide (row.1 ≠ name)) ++ [(name,position)]

def montagePositions (p : Primitives) : List (String × ArrayImage Atom) → List (String × Vector) →
    Check (List (String × Vector))
  | [],positions => .ok positions
  | (name,image) :: tail,positions => do
    let key ← p.session.clock.bids.models.input.canonicalChannel name
    let positions ← if image.shape=[3] ∧ finiteArray image then do
      let vector ← p.asVector image
      pure (putPosition key vector positions)
      else pure positions
    montagePositions p tail positions

def fiducials (p : Primitives) : List (String × ArrayImage Atom) → Check (List (String × Vector))
  | [] => .ok []
  | (name,image) :: tail => do
    ensure (image.shape=[3] ∧ finiteArray image) "finite metre fiducial three vector"
    let vector ← p.asVector image
    let rest ← fiducials p tail
    pure ((name,vector)::rest)

def vectorAt (positions : List (String × Vector)) (key : String) : Check Vector :=
  match positions.lookup key with
  | none => .error (.sourceReject "geometry required measured coordinate present")
  | some vector => .ok vector

structure Transform where
  rotation : Matrix
  translation : Vector

def fiducialTransform (p : Primitives) (positions : List (String × Vector)) : Check Transform := do
  let lpa ← vectorAt positions "lpa"
  let rpa ← vectorAt positions "rpa"
  let nasion ← vectorAt positions "nasion"
  let xAxis ← p.subtract rpa lpa
  let xNorm ← p.norm xAxis
  ensure (1/1000000 < xNorm) "geometry noncoincident auricular fiducials"
  let xAxis ← p.divide xAxis xNorm
  let nasionLpa ← p.subtract nasion lpa
  let axial ← p.dot nasionLpa xAxis
  let axialVector ← p.scale axial xAxis
  let origin ← p.add lpa axialVector
  let nasionDirection ← p.subtract nasion origin
  let yNorm ← p.norm nasionDirection
  ensure (1/1000000 < yNorm) "geometry noncollinear head fiducials"
  let yAxis ← p.divide nasionDirection yNorm
  let zAxis ← p.cross xAxis yAxis
  let rotation : Matrix := ![xAxis,yAxis,zAxis]
  let translation ← p.matvec rotation origin
  pure ⟨rotation,fun i => -translation i⟩

def identityMatrix : Matrix := fun i j => if i=j then 1 else 0

def checkRigid (p : Primitives) (transform : Transform) : Check Unit := do
  let gram ← p.matrixProductTranspose transform.rotation
  let orthogonal ← p.allclose gram identityMatrix (1/1000000)
  let determinant ← p.determinant transform.rotation
  let determinantClose ← p.determinantClose determinant
  ensure (orthogonal=true ∧ determinantClose=true) "geometry proper rigid transform"

def transformVector (p : Primitives) (transform : Transform) (position : Vector) : Check Vector := do
  let mapped ← p.matvec transform.rotation position
  p.add mapped transform.translation

def standardPositions (p : Primitives) (names : List String) : Check (List Vector) := do
  let montage ← p.loadMontage "standard_1005"
  let positions ← montagePositions p montage.positions []
  let frame := p.lower montage.frame
  let available := names.filter (fun name => decide ((positions.lookup name).isSome))
  let registered ← if available=[] then pure [] else do
    let original ← available.mapM (vectorAt positions)
    -- register_sensor_positions validates available fiducials even in head frame.
    let fiducials ← fiducials p montage.fiducials
    if frame="head" then pure original
    else do
      ensure (["lpa","rpa","nasion"] ⊆ fiducials.map Prod.fst) "standard montage has head registration fiducials"
      let transform ← fiducialTransform p fiducials
      checkRigid p transform
      original.mapM (transformVector p transform)
  ensure (available=names) "geometry all requested standard positions observed"
  registered.mapM p.float32

def explicitBidsPositions (p : Primitives) (bids : PersonalERPBids.Recording) :
    List (PersonalERPBids.Channel × Option (ℝ × ℝ × ℝ)) → List (String × Vector) → Check (List (String × Vector))
  | [],positions => .ok positions
  | (channel,position) :: tail,positions => do
    let positions ← if channel.kind="EEG" then do
      let name ← p.session.clock.bids.models.input.canonicalChannel channel.name
      let position ← match position with
        | none => .error (.sourceReject "BIDS EEG position mapping contains numeric vector")
        | some (x,y,z) => .ok (![x,y,z] : Vector)
      pure (putPosition name position positions)
      else pure positions
    explicitBidsPositions p bids tail positions

def explicitPositions (p : Primitives) (names : List String) (bids : PersonalERPBids.Recording) : Check (List Vector) := do
  ensure (bids.channels.length=bids.positions.length) "strict BIDS channel type position zip"
  let positions ← explicitBidsPositions p bids (bids.channels.zip bids.positions) []
  let selected ← names.mapM (vectorAt positions)
  let norms ← selected.mapM p.norm
  ensure (∀ norm ∈ norms, 0 < norm) "explicit nonzero observed head coordinates"
  -- Default coordinate_frame=head means register_sensor_positions copies these
  -- finite rows; there is no montage/ICP/fallback in this branch.
  selected.mapM p.float32

def positionJSON (positions : List Vector) : Value := .sequence (valuesOfList (positions.map fun position =>
  .sequence (valuesOfList ((List.ofFn position).map Value.number))))

def callbackGeometry (p : Primitives) (contract : Fields) (recording : PersonalERPSession.Recording)
    (postReference : Bool) : Check Unit := do
  let channelValue ← required contract "channel_names"
  let channels ← textList channelValue "callback contract channel string list"
  let channels ← canonicalUnique p channels
  let positions ← if postReference then do
    let positions ← standardPositions p channels
    let _ ← p.embedding positions
    pure positions
    else explicitPositions p channels recording.bids
  let expected ← required contract "channel_positions_m"
  ensure (PersonalAdmission.PyEq (positionJSON positions) expected) "callback exact final montage positions"

def recordingCallback (p : Primitives) (metadata : Fields) (recording : PersonalERPSession.Recording)
    (postReference : Bool) : Check Unit := do
  checkRecordingIdentityTimeSimulationReference p.session.clock.bids.models.input metadata
    (PersonalERPSession.callbackFields recording)
  let contract ← objectAt (fieldDefault metadata "input_contract") "callback physical contract mapping"
  callbackGeometry p contract recording postReference

theorem direct_positions_are_compared_in_requested_order (positions : List Vector) :
    positionJSON positions=.sequence (valuesOfList (positions.map fun position =>
      .sequence (valuesOfList ((List.ofFn position).map Value.number)))) := rfl

end

end N2P3.PersonalERPGeometry
