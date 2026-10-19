import N2P3.ERPTemporalMultiScope

/-!
# Source-temporal publication and real refits

Sources: research/execution.py:40-70 and fit_source_temporal_basis.py:51-57.
Every filesystem primitive is one OS/codec action. The seven replace attempts,
PermissionError-only backoff, context-manager close, and OSError-only ignored
cleanup are explicit. A successful replace is remembered as an event, not an
exclusive-create guarantee or a promise that external readers cannot alter it.

runPrepared starts AFTER the separate cache/row/identity preparation graph.
It is not yet the complete fit_plan entrypoint. It reads rank only there, runs
three complete fitSource bodies and four explicit source-role readers. The
three individual numerical operation environments may differ, so no hidden
cross-call eigensolver determinism is assumed. Filesystem/codec implementations
and Python/NumPy refinement remain separate proof obligations.
-/

namespace N2P3.ERPTemporalIO

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open N2P3.ERPTemporalCLI N2P3.ERPTemporalMultiScope

noncomputable section

attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

inductive Fault where
  | logic (fault : LedgerFault)
  | permission (site : String)
  | os (site : String)
  | base (site : String)
  deriving DecidableEq

def Fault.isOS : Fault → Bool
  | .permission _ | .os _ => true
  | _ => false

structure Primitives where
  World : Type
  mkdirParents : String → EStateM Fault World Unit
  temporaryName : String → EStateM Fault World String
  openUTF8 : String → EStateM Fault World ℕ
  dumpJSONNoNan : ℕ → Fields → EStateM Fault World Unit
  writeNewline : ℕ → EStateM Fault World Unit
  flush : ℕ → EStateM Fault World Unit
  fsync : ℕ → EStateM Fault World Unit
  close : ℕ → EStateM Fault World Unit
  replace : String → String → EStateM Fault World Unit
  sleepMicroseconds : ℕ → EStateM Fault World Unit
  unlink : String → EStateM Fault World Unit
  readUTF8 : String → EStateM Fault World String
  decodeJSON : String → EStateM Fault World Value

structure State (World : Type) where
  source : ERPTemporalMultiScope.State
  world : World
  lastSuccessfulPublication : Option (String × Fields) := none

abbrev Program (p : Primitives) := EStateM Fault (State p.World)

def fileStep (p : Primitives) {α : Type} (action : EStateM Fault p.World α) : Program p α := fun state =>
  match action state.world with
  | .ok value world => .ok value {state with world := world}
  | .error fault world => .error fault {state with world := world}

def sourceStep (p : Primitives) {α : Type} (action : SourceM α) : Program p α := fun state =>
  match action state.source with
  | .ok value source => .ok value {state with source := source}
  | .error fault source => .error (.logic fault) {state with source := source}

def checkStep (p : Primitives) {α : Type} (action : Check α) : Program p α :=
  sourceStep p (attempt action)

def closeAfter (p : Primitives) (handle : ℕ) {α : Type} (body : Program p α) : Program p α := fun state =>
  match body state with
  | .ok value after =>
    match fileStep p (p.close handle) after with
    | .ok _ closed => .ok value closed
    | .error fault closed => .error fault closed
  | .error fault after =>
    match fileStep p (p.close handle) after with
    | .ok _ closed => .error fault closed
    | .error closingFault closed => .error closingFault closed

def replaceAttempts (p : Primitives) (temporary target : String) : ℕ → Program p Unit
  | 0 => fun state => .error (.base "unreachable exhausted replace loop") state
  | remaining+1 => fun state =>
    match fileStep p (p.replace temporary target) state with
    | .ok _ after => .ok () after
    | .error (.permission site) after =>
      if remaining=0 then .error (.permission site) after
      else (do
        fileStep p (p.sleepMicroseconds (10000 * 2^(6-remaining)))
        replaceAttempts p temporary target remaining) after
    | .error fault after => .error fault after

def cleanupOnFailure (p : Primitives) (temporary : String) {α : Type}
    (body : Program p α) : Program p α := fun state =>
  match body state with
  | .ok value after => .ok value after
  | .error fault after =>
    match fileStep p (p.unlink temporary) after with
    | .ok _ cleaned => .error fault cleaned
    | .error cleanupFault cleaned =>
      if cleanupFault.isOS then .error fault cleaned
      else .error cleanupFault cleaned

def atomicWriteJSON (p : Primitives) (target : String) (payload : Fields) : Program p Unit := do
  fileStep p (p.mkdirParents target)
  let temporary ← fileStep p (p.temporaryName target)
  cleanupOnFailure p temporary do
    let handle ← fileStep p (p.openUTF8 temporary)
    closeAfter p handle do
      fileStep p (p.dumpJSONNoNan handle payload)
      fileStep p (p.writeNewline handle)
      fileStep p (p.flush handle)
      fileStep p (p.fsync handle)
    replaceAttempts p temporary target 7
    fun state => .ok () {state with lastSuccessfulPublication := some (target,payload)}

def runPrepared (io : Primitives) (initial replayBefore replayAfter : Operations)
    (postWriteReader : TemporalPrimitives) (plan : ERPTemporalCLI.Header)
    (scopes : List Request) (output : String) : Program io TemporalModel := do
  let rawRank ← checkStep io (required plan.fields "rank")
  let basis ← sourceStep io (fitSource initial scopes rawRank)
  sourceStep io (verifySource replayBefore basis scopes)
  atomicWriteJSON io output basis.record
  let text ← fileStep io (io.readUTF8 output)
  let raw ← fileStep io (io.decodeJSON text)
  let restored ← checkStep io (readTemporal postWriteReader raw)
  sourceStep io (verifySource replayAfter restored scopes)
  pure restored

theorem remembered_publication_survives_source_failure (p : Primitives)
    (fault : LedgerFault) (state : State p.World) :
    sourceStep p (attempt (.error fault : Check Unit)) state=
      .error (.logic fault) state := rfl

theorem file_action_cannot_silently_clear_publication (p : Primitives)
    {α : Type} (action : EStateM Fault p.World α) (state : State p.World) :
    (match fileStep p action state with
      | .ok _ after => after.lastSuccessfulPublication
      | .error _ after => after.lastSuccessfulPublication)=state.lastSuccessfulPublication := by
  cases result : action state.world <;> simp [fileStep,result]

theorem source_action_cannot_rollback_publication (p : Primitives)
    {α : Type} (action : SourceM α) (state : State p.World) :
    (match sourceStep p action state with
      | .ok _ after => after.lastSuccessfulPublication
      | .error _ after => after.lastSuccessfulPublication)=state.lastSuccessfulPublication := by
  cases result : action state.source <;> simp [sourceStep,result]

theorem retry_delays_are_exactly_six_and_sum_to_630ms :
    ((List.range 6).map (fun attempt => 10000 * 2^attempt)).sum=630000 := by decide

/- A small OS-action fixture, fixed independently of any fitted ERP answer.
It runs the actual atomicWriteJSON body above; it is not a genuine filesystem
or physical-cache experiment. Every primitive appends its own event. -/
structure FixtureWorld where
  events : List String := []
  replacements : ℕ := 0
  delays : List ℕ := []
  targetExists : Bool := true
  targetOverwritten : Bool := false
  deriving DecidableEq

def fixtureAction (event : String) (fault : Option Fault := none) : EStateM Fault FixtureWorld Unit :=
  fun world =>
    let after := {world with events := world.events ++ [event]}
    match fault with
    | none => .ok () after
    | some fault => .error fault after

def fixtureIO (succeedAt : Option ℕ := some 7) (fsyncFault : Option Fault := none)
    (cleanupFault : Option Fault := none) : Primitives where
  World := FixtureWorld
  mkdirParents := fun _ => fixtureAction "mkdir"
  temporaryName := fun _ => do fixtureAction "temporaryName"; pure "temporary"
  openUTF8 := fun _ => do fixtureAction "open"; pure 0
  dumpJSONNoNan := fun _ _ => fixtureAction "dumpJSONNoNan"
  writeNewline := fun _ => fixtureAction "newline"
  flush := fun _ => fixtureAction "flush"
  fsync := fun _ => fixtureAction "fsync" fsyncFault
  close := fun _ => fixtureAction "close"
  replace := fun _ _ world =>
    let count := world.replacements+1
    let after := {world with replacements := count,events := world.events ++ ["replace"]}
    if succeedAt.any (fun n => decide (n ≤ count)) then
      .ok () {after with targetOverwritten := world.targetExists,targetExists := true}
    else .error (.permission "replace") after
  sleepMicroseconds := fun duration world => .ok ()
    {world with delays := world.delays ++ [duration],events := world.events ++ ["sleep"]}
  unlink := fun _ => fixtureAction "unlink" cleanupFault
  readUTF8 := fun _ => do fixtureAction "read"; pure "{}"
  decodeJSON := fun _ => do fixtureAction "decode"; pure (.object .nil)

def fixtureState : State FixtureWorld := ⟨⟨fun _ => none,{}⟩,{},none⟩

def finalState {α : Type} {p : Primitives} (result : EStateM.Result Fault (State p.World) α) : State p.World :=
  match result with
  | .ok _ state | .error _ state => state

def fixtureWrite (succeedAt : Option ℕ := some 7) (fsyncFault : Option Fault := none)
    (cleanupFault : Option Fault := none) : EStateM.Result Fault (State FixtureWorld) Unit :=
  atomicWriteJSON (fixtureIO succeedAt fsyncFault cleanupFault) "output" .nil fixtureState

theorem permission_errors_retry_until_seventh_attempt :
    (finalState (p:=fixtureIO) fixtureWrite).world.replacements=7 ∧
    (finalState (p:=fixtureIO) fixtureWrite).world.delays=[10000,20000,40000,80000,160000,320000] ∧
    (finalState (p:=fixtureIO) fixtureWrite).lastSuccessfulPublication=some ("output",.nil) := by
  simp [finalState,fixtureWrite,atomicWriteJSON,fixtureIO,fixtureAction,cleanupOnFailure,
    closeAfter,fileStep,replaceAttempts,fixtureState,EStateM.bind,EStateM.pure]

theorem seventh_permission_failure_has_no_eighth_attempt :
    (finalState (p:=fixtureIO none) (fixtureWrite none)).world.replacements=7 ∧
    (finalState (p:=fixtureIO none) (fixtureWrite none)).world.delays.length=6 ∧
    (finalState (p:=fixtureIO none) (fixtureWrite none)).world.events.getLast?=some "unlink" := by
  simp [finalState,fixtureWrite,atomicWriteJSON,fixtureIO,fixtureAction,cleanupOnFailure,
    closeAfter,fileStep,replaceAttempts,fixtureState,EStateM.bind,EStateM.pure]

theorem fsync_failure_closes_then_cleans_without_publication :
    (finalState (p:=fixtureIO (some 7) (some (.os "fsync")))
      (fixtureWrite (some 7) (some (.os "fsync")))).world.events=
      ["mkdir","temporaryName","open","dumpJSONNoNan","newline","flush","fsync","close","unlink"] ∧
    (finalState (p:=fixtureIO (some 7) (some (.os "fsync")))
      (fixtureWrite (some 7) (some (.os "fsync")))).lastSuccessfulPublication=none := by
  exact ⟨rfl,rfl⟩

theorem cleanup_OS_error_preserves_original_fsync_error :
    (match fixtureWrite (some 7) (some (.os "fsync")) (some (.os "unlink")) with
      | .ok _ _ => none | .error fault _ => some fault)=some (.os "fsync") := by
  rfl

theorem cleanup_non_OS_exception_replaces_original_error :
    (match fixtureWrite (some 7) (some (.os "fsync")) (some (.base "unlink interrupt")) with
      | .ok _ _ => none | .error fault _ => some fault)=some (.base "unlink interrupt") := by
  rfl

theorem atomic_replace_does_not_recheck_output_nonexistence :
    (finalState (p:=fixtureIO (some 1)) (fixtureWrite (some 1))).world.targetOverwritten=true := by
  simp [finalState,fixtureWrite,atomicWriteJSON,fixtureIO,fixtureAction,cleanupOnFailure,
    closeAfter,fileStep,replaceAttempts,fixtureState,EStateM.bind,EStateM.pure]

end
end N2P3.ERPTemporalIO
