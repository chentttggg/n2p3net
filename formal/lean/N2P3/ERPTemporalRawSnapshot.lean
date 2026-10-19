import N2P3.ERPTemporalSnapshot
import N2P3.ERPTemporalRawFields

/-! Raw sidecar JSON, including unknown NaN/Infinity fields, survives stable
snapshot acquisition. Only the actual four attestation guards construct a
finite canonical attestation. The computed source record later compares with
the original raw sidecar; no full-record finite projection is made here. -/

namespace N2P3.ERPTemporalRawSnapshot

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open N2P3.ERPTemporalCache N2P3.ERPTemporalSnapshot N2P3.ERPTemporalJSON

noncomputable section

structure Operations (World : Type) where
  files : ERPTemporalSnapshot.Operations World
  decodeJSON : String → EStateM Fault World JValue

def readSidecar {World : Type} (p : Operations World) (path : String) : TraceM World (JFields × Fields) := do
  let recordPath ← small (p.files.sidecarPath path)
  let present ← small (p.files.isFile recordPath)
  checked (ensure (present=true) "cache lacks attestation sidecar")
  let raw ← remapFault (fun fault => match fault with
      | .os _ | .jsonDecode => .check (.sourceReject "unreadable cache attestation")
      | _ => fault) do
    let text ← small (p.files.readUTF8 recordPath)
    small (p.decodeJSON text)
  let record ← checked (ERPTemporalRawFields.objectAt raw "cache record.get")
  let attestation ← match ERPTemporalRawFields.fieldDefault record "cache_attestation" with
    | .object fields => pure fields
    | _ => checked (.error (.sourceReject "cache lacks supported attestation"))
  let canonical ← remapFault (fun fault => match fault with
      | .check (.sourceReject _) => .check (.sourceReject "cache lacks supported attestation")
      | _ => fault) (checked (ERPTemporalRawFields.cacheAttestation (.object attestation)))
  pure (record,canonical)

def prepareSnapshot {World : Type} (p : Operations World) (path : String) :
    TraceM World (SnapshotPacket JFields) := do
  let present ← small (p.files.pathExists path)
  if present then pure ()
  else small (fun world => .error (.os "cache FileNotFoundError") world)
  let sidecar ← readSidecar p path
  acquireSnapshot p.files path sidecar.1 sidecar.2

end
end N2P3.ERPTemporalRawSnapshot
