import N2P3.RawLoaderMapping

/-! RA2 ZIP metadata, complete CRC and extracted-stream identity. All info
entries (including directories/unrequested files) participate in inspection
and CRC; only nondirectories enter extracted-member lookup and byte totals.
CRC/decompression/digests are observed leaf interfaces, not native proofs. -/
namespace N2P3.RawZipLoader
open N2P3.RawManifestSnapshots N2P3.RawLoaderMapping
noncomputable section
attribute [local instance] Classical.propDecidable

structure Info where
  name : String
  directory : Bool
  size : ℕ
  encrypted : Bool
  regularType : Bool
  crcGood : Bool
  deriving DecidableEq, Repr

def memberLimit : ℕ := 64*1024^3
def totalLimit : ℕ := 256*1024^3
def memberCountLimit : ℕ := 100000

def files (infos : List Info) : List Info := infos.filter (fun info => !info.directory)
def totalBytes (infos : List Info) : ℕ := ((files infos).map Info.size).sum

def memberOK (pathOK : String → Bool) (info : Info) : Prop :=
  pathOK info.name=true ∧ info.encrypted=false ∧ info.regularType=true ∧ info.size≤memberLimit

def inspectionGate (fold : String → String) (pathOK : String → Bool) (archiveBytes : ℕ) (infos : List Info) : Prop :=
  infos≠[] ∧ infos.length≤memberCountLimit ∧
  (infos.map (fun info => fold info.name)).Nodup ∧
  (∀ info ∈ infos,memberOK pathOK info) ∧
  totalBytes infos≤totalLimit ∧ totalBytes infos≤200*max archiveBytes 1

structure Audit where
  archive : String
  memberCount : ℕ
  uncompressedBytes : ℕ
  crcVerified : Bool
  deriving DecidableEq, Repr

def completeCRC (infos : List Info) : Bool := infos.all Info.crcGood

def auditOf (archive : String) (infos : List Info) : Audit :=
  ⟨archive,infos.length,totalBytes infos,true⟩

def inspect (fold : String → String) (pathOK : String → Bool) (_archive : String) (archiveBytes : ℕ)
    (infos : List Info) : Option (List Info × ℕ × ℕ) :=
  if inspectionGate fold pathOK archiveBytes infos then some (files infos,infos.length,totalBytes infos) else none

theorem inspected_count_includes_directories_and_total_uses_only_files (archive : String) (infos : List Info) :
    (auditOf archive infos).memberCount=infos.length ∧
      (auditOf archive infos).uncompressedBytes=((files infos).map Info.size).sum := by exact ⟨rfl,rfl⟩

theorem full_inspection_is_derived_from_independent_entry_and_size_observations
    (fold : String → String) (pathOK : String → Bool) (archive : String) (archiveBytes : ℕ) (infos : List Info)
    (nonempty : infos≠[]) (count : infos.length≤memberCountLimit)
    (unique : (infos.map (fun info => fold info.name)).Nodup)
    (entries : ∀ info ∈ infos,memberOK pathOK info)
    (total : totalBytes infos≤totalLimit) (expansion : totalBytes infos≤200*max archiveBytes 1) :
    inspect fold pathOK archive archiveBytes infos=some (files infos,infos.length,totalBytes infos) := by
  exact if_pos ⟨nonempty,count,unique,entries,total,expansion⟩

theorem casefold_collision_is_checked_across_directory_and_file_entries
    (fold : String → String) (pathOK : String → Bool) (archive : String) (archiveBytes : ℕ) (infos : List Info)
    (collision : ¬(infos.map (fun info => fold info.name)).Nodup) :
    inspect fold pathOK archive archiveBytes infos=none := by
  have notReady : ¬inspectionGate fold pathOK archiveBytes infos := fun ready => collision ready.2.2.1
  simp [inspect,notReady]

theorem full_archive_CRC_checks_every_entry_even_if_not_requested (infos : List Info) :
    completeCRC infos=true ↔ ∀ info ∈ infos,info.crcGood=true := by simp [completeCRC,List.all_eq_true]

theorem one_bad_unrequested_member_also_refuses_full_CRC (infos : List Info) (bad : Info)
    (present : bad ∈ infos) (failed : bad.crcGood=false) : completeCRC infos=false := by
  have notGood : ¬completeCRC infos=true := by
    intro good
    have checked := (full_archive_CRC_checks_every_entry_even_if_not_requested infos).mp good bad present
    simp [failed] at checked
  cases reply : completeCRC infos <;> simp_all

def memberFor (name : String) (infos : List Info) : Option Info :=
  match infos.filter (fun info => decide (info.name=name) && !info.directory) with
  | [info] => some info
  | _ => none

theorem requested_member_lookup_keeps_the_exact_name_and_rejects_directories
    (name : String) (infos : List Info) (info : Info) (chosen : memberFor name infos=some info) :
    info ∈ infos ∧ info.name=name ∧ info.directory=false := by
  unfold memberFor at chosen
  cases filtered : infos.filter (fun entry => decide (entry.name=name) && !entry.directory) with
  | nil => simp [filtered] at chosen
  | cons first rest =>
    cases rest with
    | cons second tail => simp [filtered] at chosen
    | nil =>
      simp [filtered] at chosen
      subst info
      have present : first ∈ infos.filter (fun entry => decide (entry.name=name) && !entry.directory) := by rw [filtered]; simp
      simpa using (List.mem_filter.mp present)

structure Extraction where
  streamSize : ℕ
  streamDigest : String
  stored : Observation
  deriving DecidableEq, Repr

def expectedMember (target : String) (info : Info) (trace : Extraction) : File :=
  ⟨target,info.size,trace.streamDigest,trace.streamDigest,none,true⟩

def memberSnapshot (target : String) (info : Info) (trace : Extraction) : Option Snapshot :=
  if trace.streamSize=info.size ∧ streamMatches (expectedMember target info trace) trace.stored then
    some (snapshotOf (expectedMember target info trace)) else none

theorem extracted_stream_and_stored_object_determine_each_member_snapshot
    (target : String) (info : Info) (trace : Extraction)
    (size : trace.streamSize=info.size) (stored : streamMatches (expectedMember target info trace) trace.stored) :
    memberSnapshot target info trace=some ⟨target,info.size,trace.streamDigest,none,"objects/sha256/"++trace.streamDigest⟩ := by
  rw [memberSnapshot,if_pos ⟨size,stored⟩]
  rfl

theorem a_member_stream_size_mismatch_refuses_even_a_matching_digest
    (target : String) (info : Info) (trace : Extraction) (wrong : trace.streamSize≠info.size) :
    memberSnapshot target info trace=none := by simp [memberSnapshot,wrong]

def extractLoader (archive : String) (infos : List Info) (observe : Info → Extraction)
    (subject : ℕ) (target member : String) : Option Loader :=
  if completeCRC infos=false then none
  else
    (memberFor member infos).bind (fun info =>
      (memberSnapshot target info (observe info)).map (fun snapshot => ⟨subject,target,.zip archive member,snapshot⟩))

theorem full_CRC_failure_blocks_even_a_requested_good_member
    (archive : String) (infos : List Info) (observe : Info → Extraction)
    (subject : ℕ) (target member : String) (failed : completeCRC infos=false) :
    extractLoader archive infos observe subject target member=none := by simp [extractLoader,failed]

theorem fixed_requested_member_and_independent_stream_observations_construct_the_loader
    (archive : String) (infos : List Info) (observe : Info → Extraction)
    (subject : ℕ) (target member : String) (info : Info)
    (crc : completeCRC infos=true) (chosen : memberFor member infos=some info)
    (size : (observe info).streamSize=info.size)
    (stored : streamMatches (expectedMember target info (observe info)) (observe info).stored) :
    extractLoader archive infos observe subject target member=
      some ⟨subject,target,.zip archive member,
        ⟨target,info.size,(observe info).streamDigest,none,"objects/sha256/"++(observe info).streamDigest⟩⟩ := by
  simp [extractLoader,crc,chosen,extracted_stream_and_stored_object_determine_each_member_snapshot target info (observe info) size stored]

theorem aliases_of_one_extracted_member_keep_its_content_identity
    (first second : String) (info : Info) (trace : Extraction) :
    (snapshotOf (expectedMember first info trace)).digest=(snapshotOf (expectedMember second info trace)).digest ∧
    (snapshotOf (expectedMember first info trace)).size=(snapshotOf (expectedMember second info trace)).size ∧
    (snapshotOf (expectedMember first info trace)).objectPath=(snapshotOf (expectedMember second info trace)).objectPath ∧
    (snapshotOf (expectedMember first info trace)).sourcePath=first ∧
    (snapshotOf (expectedMember second info trace)).sourcePath=second := by exact ⟨rfl,rfl,rfl,rfl,rfl⟩

end
end N2P3.RawZipLoader
