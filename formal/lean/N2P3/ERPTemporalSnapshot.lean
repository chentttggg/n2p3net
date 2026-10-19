import N2P3.ERPTemporalCache

/-!
# Literal attested sidecar and stable temporary snapshot acquisition

Sources: epochs.py:1073-1099,1167-1256. The streaming relation is expanded in
ERPTemporalCache.CopyStream. Every other leaf is one path, environment, OS,
JSON, scalar-conversion, flush or close action. No leaf returns a decoded or
validated dataset. This is the snapshot preparation stage; a successful result
still owns an OPEN private temporary snapshot, which the later NPZ context
must close. It is not a complete cache loader or fit_plan correctness claim.
-/

namespace N2P3.ERPTemporalSnapshot

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.ERPTemporalCache

noncomputable section

abbrev TraceM (World α : Type) := World → EStateM.Result Fault World α → Prop

def tracePure {World α : Type} (value : α) : TraceM World α := fun before result =>
  result=.ok value before

def traceBind {World α β : Type} (first : TraceM World α)
    (next : α → TraceM World β) : TraceM World β := fun before result =>
  ∃ middle, first before middle ∧ match middle with
    | .ok value after => next value after result
    | .error fault after => result=.error fault after

instance {World : Type} : Monad (TraceM World) where
  pure := tracePure
  bind := traceBind

def small {World α : Type} (action : EStateM Fault World α) : TraceM World α :=
  fun before result => action before=result

def checked {World α : Type} (action : Check α) : TraceM World α :=
  small fun world => match action with
    | .ok value => .ok value world
    | .error fault => .error (.check fault) world

def remapFault {World α : Type} (f : Fault → Fault) (body : TraceM World α) : TraceM World α :=
  fun before result => ∃ middle, body before middle ∧ result=
    match middle with
    | .ok value after => .ok value after
    | .error fault after => .error (f fault) after

def wrapOS (message : String) : Fault → Fault
  | .os _ => .check (.sourceReject message)
  | fault => fault

def closeFinally {World α : Type} (close : EStateM Fault World Unit)
    (body : TraceM World α) : TraceM World α := fun before result =>
  ∃ middle, body before middle ∧ match middle with
  | .ok value after => result=match close after with
    | .ok _ final => .ok value final
    | .error fault final => .error fault final
  | .error fault after => result=match close after with
    | .ok _ final => .error fault final
    | .error closingFault final => .error closingFault final

def closeOnFailure {World α : Type} (close : EStateM Fault World Unit)
    (body : TraceM World α) : TraceM World α := fun before result =>
  ∃ middle, body before middle ∧ match middle with
  | .ok value after => result=.ok value after
  | .error fault after => result=match close after with
    | .ok _ final => .error fault final
    | .error closingFault final => .error closingFault final

structure Operations (World : Type) where
  pathExists : String → EStateM Fault World Bool
  sidecarPath : String → EStateM Fault World String
  isFile : String → EStateM Fault World Bool
  readUTF8 : String → EStateM Fault World String
  decodeJSON : String → EStateM Fault World Value
  configuredDirectory : EStateM Fault World (Option String)
  pathFromString : String → EStateM Fault World String
  expandUser : String → EStateM Fault World String
  parent : String → String
  resolveStrict : String → EStateM Fault World String
  isDirectory : String → EStateM Fault World Bool
  freeBytes : String → EStateM Fault World ℕ
  fivePercentInteger : ℕ → Check ℕ
  openReadOnlyNoFollow : String → EStateM Fault World ℕ
  openTemporary : String → EStateM Fault World ℕ
  closeDescriptor : ℕ → EStateM Fault World Unit
  fdopenRead : ℕ → EStateM Fault World ℕ
  sourceStat : ℕ → EStateM Fault World FileStat
  closeSource : ℕ → EStateM Fault World Unit
  closeSnapshot : ℕ → EStateM Fault World Unit
  flushSnapshot : ℕ → EStateM Fault World Unit
  seekSnapshotStart : ℕ → EStateM Fault World Unit
  copy : CopyOperations World

def readSidecar {World : Type} (p : Operations World) (path : String) : TraceM World (Fields × Fields) := do
  let recordPath ← small (p.sidecarPath path)
  let present ← small (p.isFile recordPath)
  checked (ensure (present=true) "cache lacks attestation sidecar")
  let raw ← remapFault (fun fault => match fault with
      | .os _ | .jsonDecode => .check (.sourceReject "unreadable cache attestation")
      | _ => fault) do
    let text ← small (p.readUTF8 recordPath)
    small (p.decodeJSON text)
  let record ← checked (objectAt raw "cache record.get")
  let attestation ← match fieldDefault record "cache_attestation" with
    | .object fields => pure fields
    | _ => checked (.error (.sourceReject "cache lacks supported attestation"))
  let canonical ← remapFault (fun fault => match fault with
      | .check (.sourceReject _) => .check (.sourceReject "cache lacks supported attestation")
      | _ => fault) (checked (canonicalAttestation (.object attestation)))
  pure (record,canonical)

def snapshotDirectory {World : Type} (p : Operations World) (path : String) (bytes : ℕ) : TraceM World String := do
  let configured ← small p.configuredDirectory
  let directory ← match configured with
    | some configured =>
      if configured="" then small (p.resolveStrict (p.parent path))
      else do
        let directory ← small (p.pathFromString configured)
        let directory ← small (p.expandUser directory)
        small (p.resolveStrict directory)
    | none => small (p.resolveStrict (p.parent path))
  let isDirectory ← small (p.isDirectory directory)
  checked (ensure (isDirectory=true) "snapshot location must be directory")
  let free ← remapFault (wrapOS "cannot inspect snapshot free space") (small (p.freeBytes directory))
  let required ← checked (requiredSnapshotSpace p.fivePercentInteger bytes)
  checked (ensure (required ≤ free) "insufficient snapshot free space")
  pure directory

structure OpenSnapshot where
  handle : ℕ
  record : Fields
  attestation : Fields

structure SnapshotPacket (Record : Type) where
  handle : ℕ
  record : Record
  attestation : Fields

def openTemporaryAfterDescriptor {World : Type} (p : Operations World) (descriptor : ℕ)
    (directory : String) : TraceM World ℕ := fun before result =>
  match p.openTemporary directory before with
  | .ok handle after => result=.ok handle after
  | .error (.os _) after => result=match p.closeDescriptor descriptor after with
    | .ok _ final => .error (.check (.sourceReject "cannot create private cache snapshot")) final
    | .error fault final => .error fault final
  | .error fault after => result=.error fault after

def acquireSnapshot {World Record : Type} (p : Operations World) (path : String)
    (record : Record) (attestation : Fields) : TraceM World (SnapshotPacket Record) := do
  let bytes ← checked (positiveInt (fieldDefault attestation "byte_size") "canonical cache byte size")
  let directory ← snapshotDirectory p path bytes
  let descriptor ← remapFault (wrapOS "cannot open stable cache object") (small (p.openReadOnlyNoFollow path))
  let temporary ← openTemporaryAfterDescriptor p descriptor directory
  closeOnFailure (p.closeSnapshot temporary) do
    let copied ← remapFault (wrapOS "cannot copy stable cache snapshot") do
      let source ← small (p.fdopenRead descriptor)
      closeFinally (p.closeSource source) do
        let before ← small (p.sourceStat source)
        checked (checkBeforeCopy attestation before)
        let copied ← (fun before result => CopyStream p.copy source temporary before result)
        let after ← small (p.sourceStat source)
        pure (before,after,copied)
    checked (checkAfterCopy attestation copied.1 copied.2.1 copied.2.2.1 copied.2.2.2)
    small (p.flushSnapshot temporary)
    small (p.seekSnapshotStart temporary)
    pure ⟨temporary,record,attestation⟩

def prepareSnapshot {World : Type} (p : Operations World) (path : String) : TraceM World OpenSnapshot := do
  let present ← small (p.pathExists path)
  if present then pure ()
  else small (fun world => .error (.os "cache FileNotFoundError") world)
  let sidecar ← readSidecar p path
  let snapshot ← acquireSnapshot p path sidecar.1 sidecar.2
  pure ⟨snapshot.handle,snapshot.record,snapshot.attestation⟩

theorem permissionless_OS_wrap_preserves_non_OS_fault (message site : String) :
    wrapOS message (.other site)=.other site := rfl

theorem failed_temporary_creation_closes_original_descriptor {World : Type} (p : Operations World)
    (descriptor : ℕ) (directory : String) (before after final : World) (site : String)
    (failed : p.openTemporary directory before=.error (.os site) after)
    (closed : p.closeDescriptor descriptor after=.ok () final) :
    openTemporaryAfterDescriptor p descriptor directory before
      (.error (.check (.sourceReject "cannot create private cache snapshot")) final) := by
  simp [openTemporaryAfterDescriptor,failed,closed]

end
end N2P3.ERPTemporalSnapshot
