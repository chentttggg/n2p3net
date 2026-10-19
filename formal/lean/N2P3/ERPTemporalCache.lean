import N2P3.ERPTemporalMultiScope

/-!
# Attested-cache stream and stable snapshot stages

Source: epochs.py:1062-1099,1103-1125,1167-1256.
The byte-copy loop is an inductive relation over EACH read/hash-update/write
operation. It has no whole-copy, validated-load, decoded-dataset or fit oracle,
and does not invent a finite fuel limit in the Python loop. Finite traces are
represented; OS termination/fairness is a separate liveness obligation.

This file currently specifies the copy and completed snapshot gates. NPZ field
decoding/constructor/attested-record comparison and the whole fit_plan linkage
remain pending. In particular an NPZ stored decoded digest must not be silently
replaced by a freshly recomputed dataset digest in the final loader model.
-/

namespace N2P3.ERPTemporalCache

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData

noncomputable section

attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

inductive Fault where
  | check (fault : LedgerFault)
  | os (site : String)
  | jsonDecode
  | other (site : String)
  deriving DecidableEq

structure CopyOperations (World : Type) where
  HashState : Type
  startHash : EStateM Fault World HashState
  seekStart : ℕ → EStateM Fault World Unit
  readBlock : ℕ → ℕ → EStateM Fault World (List UInt8)
  update : HashState → List UInt8 → EStateM Fault World HashState
  write : ℕ → List UInt8 → EStateM Fault World Unit
  finish : HashState → EStateM Fault World String

/-- A completed or failed finite tail of the actual stream loop. The byte
count advances only after hash update and destination write both returned. -/
inductive CopyTail {World : Type} (p : CopyOperations World) (source destination : ℕ) :
    p.HashState → ℕ → World → EStateM.Result Fault World (String × ℕ) → Prop
  | readError (hash size before after fault)
      (read : p.readBlock source (1024*1024) before=.error fault after) :
      CopyTail p source destination hash size before (.error fault after)
  | eof (hash size before after result)
      (read : p.readBlock source (1024*1024) before=.ok [] after)
      (hex : p.finish hash after=result) :
      CopyTail p source destination hash size before
        (match result with
        | .ok digest final => .ok (digest,size) final
        | .error fault final => .error fault final)
  | updateError (hash size before readAfter after fault bytes)
      (read : p.readBlock source (1024*1024) before=.ok bytes readAfter)
      (nonempty : bytes ≠ []) (updated : p.update hash bytes readAfter=.error fault after) :
      CopyTail p source destination hash size before (.error fault after)
  | writeError (hash nextHash size before readAfter hashAfter after fault bytes)
      (read : p.readBlock source (1024*1024) before=.ok bytes readAfter)
      (nonempty : bytes ≠ []) (updated : p.update hash bytes readAfter=.ok nextHash hashAfter)
      (written : p.write destination bytes hashAfter=.error fault after) :
      CopyTail p source destination hash size before (.error fault after)
  | block (hash nextHash size before readAfter hashAfter writeAfter result bytes)
      (read : p.readBlock source (1024*1024) before=.ok bytes readAfter)
      (nonempty : bytes ≠ []) (updated : p.update hash bytes readAfter=.ok nextHash hashAfter)
      (written : p.write destination bytes hashAfter=.ok () writeAfter)
      (tail : CopyTail p source destination nextHash (size+bytes.length) writeAfter result) :
      CopyTail p source destination hash size before result

inductive CopyStream {World : Type} (p : CopyOperations World) (source destination : ℕ) :
    World → EStateM.Result Fault World (String × ℕ) → Prop
  | initializeError (before after fault) (failed : p.startHash before=.error fault after) :
      CopyStream p source destination before (.error fault after)
  | seekError (before hashAfter after hash fault)
      (started : p.startHash before=.ok hash hashAfter)
      (failed : p.seekStart source hashAfter=.error fault after) :
      CopyStream p source destination before (.error fault after)
  | run (before hashAfter seekAfter hash result)
      (started : p.startHash before=.ok hash hashAfter)
      (seeked : p.seekStart source hashAfter=.ok () seekAfter)
      (tail : CopyTail p source destination hash 0 seekAfter result) :
      CopyStream p source destination before result

structure FileStat where
  regular : Bool
  size : ℕ
  device : ℤ
  inode : ℤ
  mtimeNS : ℤ
  deriving DecidableEq

def fileIdentity (stat : FileStat) : ℤ × ℤ × ℕ × ℤ :=
  (stat.device,stat.inode,stat.size,stat.mtimeNS)

def canonicalAttestation (raw : Value) : Check Fields := do
  let fields ← objectAt raw "cache attestation mapping"
  checkCacheAttestationFields fields
  pure (Fields.ofList [
    ("schema",.text "n2p3net_epoch_cache_attestation/1"),
    ("full_contract_validated",.boolean true),("sha256",fieldDefault fields "sha256"),
    ("byte_size",fieldDefault fields "byte_size")])

def checkBeforeCopy (attestation : Fields) (before : FileStat) : Check Unit := do
  ensure (before.regular=true) "cache must be a regular file"
  ensure (fieldDefault attestation "byte_size"=.integer before.size) "cache initial byte size"

def checkAfterCopy (attestation : Fields) (before after : FileStat)
    (digest : String) (size : ℕ) : Check Unit := do
  ensure (fileIdentity after=fileIdentity before) "cache changed during snapshot copy"
  ensure (fieldDefault attestation "byte_size"=.integer size) "cache copied byte size"
  ensure (fieldDefault attestation "sha256"=.text digest) "cache copied SHA256"

/-- Five-percent reserve is the individual Python binary64 multiply/int
operation. It is not silently equated with exact integer division by 20. -/
def requiredSnapshotSpace (fivePercent : ℕ → Check ℕ) (bytes : ℕ) : Check ℕ := do
  let reserve ← fivePercent bytes
  pure (bytes + max (64*1024*1024) reserve)

/-- Source uses the stored decoded digest when present. Actual dataset
content hashing belongs exclusively to the absent-field branch. -/
def chooseDecodedDigest (stored : Option String) (compute : Check String) : Check String :=
  match stored with
  | some digest => .ok digest
  | none => compute

theorem initial_nonregular_file_refused_before_byte_size (attestation : Fields)
    (before : FileStat) (notRegular : before.regular=false) :
    checkBeforeCopy attestation before=.error (.sourceReject "cache must be a regular file") := by
  simp [checkBeforeCopy,ensure,notRegular]

theorem changed_file_identity_refused_before_size_and_hash (attestation : Fields)
    (before after : FileStat) (digest : String) (size : ℕ)
    (changed : fileIdentity after ≠ fileIdentity before) :
    checkAfterCopy attestation before after digest size=
      .error (.sourceReject "cache changed during snapshot copy") := by
  simp [checkAfterCopy,ensure,changed]

theorem stored_decoded_digest_does_not_rerun_content_hash (digest : String) (compute : Check String) :
    chooseDecodedDigest (some digest) compute=.ok digest := rfl

theorem absent_decoded_digest_runs_content_hash (compute : Check String) :
    chooseDecodedDigest none compute=compute := rfl

theorem write_failure_is_after_the_hash_update {World : Type} (p : CopyOperations World)
    (source destination size : ℕ) (hash nextHash : p.HashState)
    (before readAfter hashAfter after : World) (fault : Fault) (bytes : List UInt8)
    (read : p.readBlock source (1024*1024) before=.ok bytes readAfter) (nonempty : bytes ≠ [])
    (updated : p.update hash bytes readAfter=.ok nextHash hashAfter)
    (written : p.write destination bytes hashAfter=.error fault after) :
    CopyTail p source destination hash size before (.error fault after) :=
  .writeError hash nextHash size before readAfter hashAfter after fault bytes read nonempty updated written

end
end N2P3.ERPTemporalCache
