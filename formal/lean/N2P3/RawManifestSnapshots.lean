import Mathlib.Data.List.Nodup
import Mathlib.Tactic.SplitIfs

/-! RA1: decoded manifest inventory and the actual per-file observation loop.
Path validation, Unicode casefold, descriptor and digest observations are
individual interfaces. No digest injectivity or whole-verifier success premise
is supplied. Input list order is the canonical manifest order; native sorting
and metadata decoding precede this interface. Publication/storage authenticity
and RA2 loader/ZIP mapping are separate obligations. -/
namespace N2P3.RawManifestSnapshots

structure File where
  path : String
  size : ℕ
  localDigest : String
  remoteDigest : String
  officialMD5 : Option String
  verified : Bool
  deriving DecidableEq, Repr

structure Observation where
  identityBefore : ℕ
  identityAfter : ℕ
  size : ℕ
  digest : String
  md5 : Option String
  deriving DecidableEq, Repr

structure Trace where
  source : Observation
  stored : Observation
  deriving DecidableEq, Repr

structure Snapshot where
  sourcePath : String
  size : ℕ
  digest : String
  md5 : Option String
  objectPath : String
  deriving DecidableEq, Repr

def optionalMD5 (expected observed : Option String) : Prop :=
  expected = none ∨ observed = expected

instance (expected observed : Option String) : Decidable (optionalMD5 expected observed) :=
  inferInstanceAs (Decidable (expected = none ∨ observed = expected))

def streamMatches (file : File) (observed : Observation) : Prop :=
  observed.identityBefore = observed.identityAfter ∧ observed.size = file.size ∧
    observed.digest = file.localDigest ∧ optionalMD5 file.officialMD5 observed.md5

instance (file : File) (observed : Observation) : Decidable (streamMatches file observed) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

def snapshotOf (file : File) : Snapshot :=
  ⟨file.path, file.size, file.localDigest, file.officialMD5,
    "objects/sha256/" ++ file.localDigest⟩

def oneFile (file : File) (trace : Trace) : Option Snapshot :=
  if streamMatches file trace.source ∧ streamMatches file trace.stored then
    some (snapshotOf file) else none

def allFiles (observe : File → Trace) : List File → Option (List Snapshot)
  | [] => some []
  | file :: rest => match oneFile file (observe file), allFiles observe rest with
    | some snapshot, some tail => some (snapshot :: tail)
    | _, _ => none

def recordMatches (pathOK digestOK : String → Bool) (md5OK : Option String → Bool)
    (file : File) : Prop :=
  pathOK file.path = true ∧ digestOK file.localDigest = true ∧
    digestOK file.remoteDigest = true ∧ md5OK file.officialMD5 = true ∧
    file.localDigest = file.remoteDigest ∧ file.verified = true

def manifestGate (fold : String → String) (pathOK digestOK : String → Bool)
    (md5OK : Option String → Bool) (expectedClass declaredClass : String)
    (files : List File) : Prop :=
  declaredClass = expectedClass ∧ files ≠ [] ∧
    (files.map (fun file => fold file.path)).Nodup ∧
    ∀ file ∈ files, recordMatches pathOK digestOK md5OK file

def bytes (files : List File) : ℕ := (files.map File.size).sum
def snapshotBytes (snapshots : List Snapshot) : ℕ := (snapshots.map Snapshot.size).sum

theorem absent_official_MD5_does_not_constrain_the_observed_MD5 (observed : Option String) :
    optionalMD5 none observed := Or.inl rfl

theorem supplied_official_MD5_requires_the_exact_optional_observation (digest : String)
    (observed : Option String) : optionalMD5 (some digest) observed ↔ observed = some digest := by
  simp [optionalMD5]

theorem a_successful_stream_observation_checks_identity_size_SHA_and_optional_MD5
    (file : File) (observed : Observation) (accepted : streamMatches file observed) :
    observed.identityBefore = observed.identityAfter ∧ observed.size = file.size ∧
      observed.digest = file.localDigest ∧ optionalMD5 file.officialMD5 observed.md5 := accepted

theorem per_file_success_has_exact_source_metadata_and_both_stream_checks
    (file : File) (trace : Trace) (snapshot : Snapshot) :
    oneFile file trace = some snapshot ↔
      streamMatches file trace.source ∧ streamMatches file trace.stored ∧ snapshot = snapshotOf file := by
  simp only [oneFile]
  split_ifs with checks
  · simp only [Option.some.injEq]
    constructor
    · intro same
      exact ⟨checks.1, checks.2, same.symm⟩
    · rintro ⟨_, _, rfl⟩
      rfl
  · simp only [false_iff]
    rintro ⟨source, stored, _⟩
    exact checks ⟨source, stored⟩

theorem independently_matching_source_and_stored_streams_construct_the_expected_snapshot
    (file : File) (trace : Trace) (source : streamMatches file trace.source)
    (stored : streamMatches file trace.stored) : oneFile file trace = some (snapshotOf file) := by
  simp [oneFile, source, stored]

theorem stored_object_recheck_is_required_even_after_a_good_source_copy
    (file : File) (trace : Trace) (stored : ¬ streamMatches file trace.stored) :
    oneFile file trace = none := by simp [oneFile, stored]

theorem all_files_success_iff_every_individual_trace_matches
    (observe : File → Trace) (files : List File) (snapshots : List Snapshot) :
    allFiles observe files = some snapshots ↔
      (∀ file ∈ files, streamMatches file (observe file).source ∧
        streamMatches file (observe file).stored) ∧ snapshots = files.map snapshotOf := by
  induction files generalizing snapshots with
  | nil => simp [allFiles]
  | cons file rest ih =>
    simp only [allFiles]
    cases first : oneFile file (observe file) with
    | none =>
      simp only [reduceCtorEq, false_iff]
      rintro ⟨all, _⟩
      have both := all file (by simp)
      have accepted := independently_matching_source_and_stored_streams_construct_the_expected_snapshot
        file (observe file) both.1 both.2
      rw [first] at accepted
      contradiction
    | some head =>
      obtain ⟨source, stored, rfl⟩ :=
        (per_file_success_has_exact_source_metadata_and_both_stream_checks file (observe file) head).mp first
      cases tail : allFiles observe rest with
      | none =>
        simp only [reduceCtorEq, false_iff]
        rintro ⟨all, _⟩
        have ready : ∀ file ∈ rest, streamMatches file (observe file).source ∧
            streamMatches file (observe file).stored := fun f member => all f (by simp [member])
        have accepted := (ih (rest.map snapshotOf)).mpr ⟨ready, rfl⟩
        rw [tail] at accepted
        contradiction
      | some result =>
        have restReady := (ih result).mp tail
        simp only [Option.some.injEq]
        constructor
        · intro same
          refine ⟨?_, ?_⟩
          · intro f member
            rcases List.mem_cons.mp member with equal | later
            · subst f; exact ⟨source, stored⟩
            · exact restReady.1 f later
          · rw [← same, List.map_cons, restReady.2]
        · rintro ⟨_, rfl⟩
          rw [List.map_cons, restReady.2]

theorem independently_verified_file_traces_preserve_every_ordered_occurrence
    (observe : File → Trace) (files : List File)
    (ready : ∀ file ∈ files, streamMatches file (observe file).source ∧
      streamMatches file (observe file).stored) :
    allFiles observe files = some (files.map snapshotOf) :=
  (all_files_success_iff_every_individual_trace_matches observe files _).mpr ⟨ready, rfl⟩

theorem successful_snapshot_loop_preserves_the_manifest_file_count
    (observe : File → Trace) (files : List File) (snapshots : List Snapshot)
    (accepted : allFiles observe files = some snapshots) : snapshots.length = files.length := by
  rw [(all_files_success_iff_every_individual_trace_matches observe files snapshots).mp accepted |>.2]
  simp

theorem successful_snapshot_loop_preserves_the_complete_source_path_list
    (observe : File → Trace) (files : List File) (snapshots : List Snapshot)
    (accepted : allFiles observe files = some snapshots) :
    snapshots.map Snapshot.sourcePath = files.map File.path := by
  rw [(all_files_success_iff_every_individual_trace_matches observe files snapshots).mp accepted |>.2]
  simp [snapshotOf]

theorem successful_snapshot_loop_preserves_the_full_byte_total_with_multiplicity
    (observe : File → Trace) (files : List File) (snapshots : List Snapshot)
    (accepted : allFiles observe files = some snapshots) : snapshotBytes snapshots = bytes files := by
  rw [(all_files_success_iff_every_individual_trace_matches observe files snapshots).mp accepted |>.2]
  simp only [snapshotBytes, bytes, List.map_map]
  rfl

theorem casefold_uniqueness_implies_literal_source_path_uniqueness
    (fold : String → String) (files : List File)
    (unique : (files.map (fun file => fold file.path)).Nodup) :
    (files.map File.path).Nodup := by
  have same : (files.map File.path).map fold = files.map (fun file => fold file.path) := by simp
  rw [← same] at unique
  exact List.Nodup.of_map fold unique

theorem successful_manifest_gate_retains_external_expected_dataset_class
    (fold : String → String) (pathOK digestOK : String → Bool) (md5OK : Option String → Bool)
    (expected declared : String) (files : List File)
    (gate : manifestGate fold pathOK digestOK md5OK expected declared files) : declared = expected := gate.1

theorem successful_manifest_gate_gives_nonempty_unique_and_verified_records
    (fold : String → String) (pathOK digestOK : String → Bool) (md5OK : Option String → Bool)
    (expected declared : String) (files : List File)
    (gate : manifestGate fold pathOK digestOK md5OK expected declared files) :
    files ≠ [] ∧ (files.map File.path).Nodup ∧
      ∀ file ∈ files, file.localDigest = file.remoteDigest ∧ file.verified = true := by
  refine ⟨gate.2.1, casefold_uniqueness_implies_literal_source_path_uniqueness fold files gate.2.2.1, ?_⟩
  intro file member
  exact ⟨(gate.2.2.2 file member).2.2.2.2.1, (gate.2.2.2 file member).2.2.2.2.2⟩

theorem source_produced_snapshot_inventory_is_complete_nonempty_and_unique
    (fold : String → String) (pathOK digestOK : String → Bool) (md5OK : Option String → Bool)
    (expected declared : String) (files : List File) (observe : File → Trace) (snapshots : List Snapshot)
    (gate : manifestGate fold pathOK digestOK md5OK expected declared files)
    (accepted : allFiles observe files = some snapshots) :
    declared = expected ∧ snapshots.length = files.length ∧ snapshots ≠ [] ∧
      snapshots.map Snapshot.sourcePath = files.map File.path ∧
      (snapshots.map Snapshot.sourcePath).Nodup ∧ snapshotBytes snapshots = bytes files := by
  have records := successful_manifest_gate_gives_nonempty_unique_and_verified_records
    fold pathOK digestOK md5OK expected declared files gate
  have count := successful_snapshot_loop_preserves_the_manifest_file_count observe files snapshots accepted
  have paths := successful_snapshot_loop_preserves_the_complete_source_path_list observe files snapshots accepted
  refine ⟨gate.1, count, ?_, paths, ?_,
    successful_snapshot_loop_preserves_the_full_byte_total_with_multiplicity observe files snapshots accepted⟩
  · intro empty
    rw [empty] at count
    exact records.1 (List.length_eq_zero_iff.mp count.symm)
  · rw [paths]
    exact records.2.1

structure Attestation where
  datasetClass : String
  files : List File
  snapshots : List Snapshot
  fileCount : ℕ
  totalBytes : ℕ
  deriving DecidableEq, Repr

def attestationOf (declared : String) (files : List File) (snapshots : List Snapshot) : Attestation :=
  ⟨declared, files, snapshots, files.length, snapshotBytes snapshots⟩

noncomputable def verifyManifest (fold : String → String) (pathOK digestOK : String → Bool)
    (md5OK : Option String → Bool) (expected declared : String) (files : List File)
    (observe : File → Trace) : Option Attestation := by
  classical
  exact if manifestGate fold pathOK digestOK md5OK expected declared files then
    (allFiles observe files).map (attestationOf declared files) else none

theorem individual_manifest_fields_and_per_file_observations_construct_the_actual_attestation
    (fold : String → String) (pathOK digestOK : String → Bool) (md5OK : Option String → Bool)
    (expected declared : String) (files : List File) (observe : File → Trace)
    (dataset : declared = expected) (nonempty : files ≠ [])
    (unique : (files.map (fun file => fold file.path)).Nodup)
    (records : ∀ file ∈ files, recordMatches pathOK digestOK md5OK file)
    (ready : ∀ file ∈ files, streamMatches file (observe file).source ∧
      streamMatches file (observe file).stored) :
    verifyManifest fold pathOK digestOK md5OK expected declared files observe =
      some (attestationOf declared files (files.map snapshotOf)) := by
  have gate : manifestGate fold pathOK digestOK md5OK expected declared files :=
    ⟨dataset, nonempty, unique, records⟩
  rw [verifyManifest, if_pos gate,
    independently_verified_file_traces_preserve_every_ordered_occurrence observe files ready]
  rfl

theorem wrong_dataset_class_refuses_before_the_snapshot_inventory_is_published
    (fold : String → String) (pathOK digestOK : String → Bool) (md5OK : Option String → Bool)
    (expected declared : String) (files : List File) (observe : File → Trace)
    (different : declared ≠ expected) :
    verifyManifest fold pathOK digestOK md5OK expected declared files observe = none := by
  simp [verifyManifest, manifestGate, different]

theorem canonical_attestation_count_and_bytes_read_the_entire_file_inventory
    (declared : String) (files : List File) :
    (attestationOf declared files (files.map snapshotOf)).fileCount = files.length ∧
      (attestationOf declared files (files.map snapshotOf)).totalBytes = bytes files := by
  constructor
  · rfl
  · simp only [attestationOf, snapshotBytes, bytes, List.map_map]
    rfl

theorem successful_snapshot_occurrence_preserves_all_corresponding_expected_fields
    (observe : File → Trace) (files : List File) (snapshots : List Snapshot)
    (accepted : allFiles observe files = some snapshots) (snapshot : Snapshot)
    (present : snapshot ∈ snapshots) :
    ∃ file ∈ files, snapshot = snapshotOf file ∧
      snapshot.sourcePath = file.path ∧ snapshot.size = file.size ∧
      snapshot.digest = file.localDigest ∧ snapshot.md5 = file.officialMD5 ∧
      snapshot.objectPath = "objects/sha256/" ++ file.localDigest := by
  rw [(all_files_success_iff_every_individual_trace_matches observe files snapshots).mp accepted |>.2] at present
  obtain ⟨file, member, same⟩ := List.mem_map.mp present
  refine ⟨file, member, same.symm, ?_⟩
  rw [← same]
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

def sampleFile (path : String) : File := ⟨path, 7, "digest", "digest", none, true⟩
def goodObservation : Observation := ⟨3, 3, 7, "digest", none⟩
def goodTrace : Trace := ⟨goodObservation, goodObservation⟩

theorem distinct_sources_can_share_one_content_addressed_object :
    (snapshotOf (sampleFile "a")).sourcePath ≠ (snapshotOf (sampleFile "b")).sourcePath ∧
      (snapshotOf (sampleFile "a")).objectPath = (snapshotOf (sampleFile "b")).objectPath := by decide

theorem duplicated_content_does_not_remove_manifest_file_occurrences_or_bytes :
    allFiles (fun _ => goodTrace) [sampleFile "a", sampleFile "b"] =
      some [snapshotOf (sampleFile "a"), snapshotOf (sampleFile "b")] ∧
      snapshotBytes [snapshotOf (sampleFile "a"), snapshotOf (sampleFile "b")] = 14 := by decide

theorem identical_file_counts_do_not_prove_snapshot_source_coverage :
    [snapshotOf (sampleFile "a")].length = [sampleFile "b"].length ∧
      [snapshotOf (sampleFile "a")].map Snapshot.sourcePath ≠ [sampleFile "b"].map File.path := by decide

theorem correct_source_copy_cannot_bypass_a_changed_stored_object :
    streamMatches (sampleFile "a") goodTrace.source ∧
      oneFile (sampleFile "a") ⟨goodObservation, { goodObservation with digest := "changed" }⟩ = none := by decide

theorem digest_equality_does_not_imply_the_expected_byte_size :
    ({goodObservation with size := 8} : Observation).digest = (sampleFile "a").localDigest ∧
      ¬ streamMatches (sampleFile "a") {goodObservation with size := 8} := by decide

theorem a_digest_function_can_identify_distinct_byte_payloads :
    (fun _ : List ℕ => "digest") [1] = (fun _ : List ℕ => "digest") [2] ∧
      ([1] : List ℕ) ≠ [2] := by decide

end N2P3.RawManifestSnapshots
