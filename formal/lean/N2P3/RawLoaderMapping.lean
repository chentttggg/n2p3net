import N2P3.RawManifestSnapshots

/-! RA2: actual direct/ZIP mapping and selected-subject support. This starts
at strictly decoded positive-subject/canonical-path records. Individual path
and Unicode casefold operations remain interfaces. The whole mapping is
checked before filtering; no final materialization success is assumed. -/
namespace N2P3.RawLoaderMapping
open N2P3.RawManifestSnapshots
noncomputable section
attribute [local instance] Classical.propDecidable

inductive Source where
  | direct (relative : String)
  | zip (archive member : String)
  deriving DecidableEq, Repr

structure Spec where
  subject : ℕ
  target : String
  source : Source
  deriving DecidableEq, Repr

structure Loader where
  subject : ℕ
  target : String
  origin : Source
  snapshot : Snapshot
  deriving DecidableEq, Repr

def sourceFieldsOK (pathOK memberOK : String → Bool) : Source → Prop
  | .direct relative => pathOK relative=true
  | .zip archive member => pathOK archive=true ∧ memberOK member=true

def fieldsOK (pathOK loaderOK memberOK : String → Bool) (spec : Spec) : Prop :=
  0<spec.subject ∧ loaderOK spec.target=true ∧ sourceFieldsOK pathOK memberOK spec.source

def selectedEntries (selected : List ℕ) (entries : List Spec) : List Spec :=
  entries.filter (fun spec => decide (spec.subject ∈ selected))

def mappingReady (fold : String → String) (pathOK loaderOK memberOK : String → Bool)
    (selected : List ℕ) (entries : List Spec) : Prop :=
  selected≠[] ∧ selected.Nodup ∧ entries≠[] ∧
  (∀ spec ∈ entries,fieldsOK pathOK loaderOK memberOK spec) ∧
  (entries.map (fun spec => fold spec.target)).Nodup ∧
  ∀ subject ∈ selected,∃ spec ∈ entries,spec.subject=subject

def selectSpecs (fold : String → String) (pathOK loaderOK memberOK : String → Bool)
    (selected : List ℕ) (entries : List Spec) : Option (List Spec) :=
  if mappingReady fold pathOK loaderOK memberOK selected entries then some (selectedEntries selected entries) else none

theorem every_selected_mapping_entry_is_an_actual_selected_subject_entry
    (selected : List ℕ) (entries : List Spec) (spec : Spec) :
    spec ∈ selectedEntries selected entries ↔ spec ∈ entries ∧ spec.subject ∈ selected := by
  simp [selectedEntries]

theorem independent_source_fields_construct_the_selected_mapping
    (fold : String → String) (pathOK loaderOK memberOK : String → Bool)
    (selected : List ℕ) (entries : List Spec)
    (selectedNonempty : selected≠[]) (selectedUnique : selected.Nodup) (entriesNonempty : entries≠[])
    (fields : ∀ spec ∈ entries,fieldsOK pathOK loaderOK memberOK spec)
    (targets : (entries.map (fun spec => fold spec.target)).Nodup)
    (coverage : ∀ subject ∈ selected,∃ spec ∈ entries,spec.subject=subject) :
    selectSpecs fold pathOK loaderOK memberOK selected entries=some (selectedEntries selected entries) := by
  exact if_pos ⟨selectedNonempty,selectedUnique,entriesNonempty,fields,targets,coverage⟩

theorem a_collision_anywhere_in_the_unfiltered_mapping_refuses_selection
    (fold : String → String) (pathOK loaderOK memberOK : String → Bool)
    (selected : List ℕ) (entries : List Spec) (collision : ¬(entries.map (fun spec => fold spec.target)).Nodup) :
    selectSpecs fold pathOK loaderOK memberOK selected entries=none := by
  have notReady : ¬mappingReady fold pathOK loaderOK memberOK selected entries := fun ready => collision ready.2.2.2.2.1
  simp [selectSpecs,notReady]

theorem absent_selected_subject_is_not_silently_omitted
    (fold : String → String) (pathOK loaderOK memberOK : String → Bool)
    (selected : List ℕ) (entries : List Spec) (subject : ℕ) (requested : subject ∈ selected)
    (missing : ∀ spec ∈ entries,spec.subject≠subject) :
    selectSpecs fold pathOK loaderOK memberOK selected entries=none := by
  have notReady : ¬mappingReady fold pathOK loaderOK memberOK selected entries := by
    intro ready
    obtain ⟨spec,member,same⟩ := ready.2.2.2.2.2 subject requested
    exact missing spec member same
  simp [selectSpecs,notReady]

theorem every_requested_subject_keeps_at_least_one_selected_entry
    (selected : List ℕ) (entries : List Spec)
    (coverage : ∀ subject ∈ selected,∃ spec ∈ entries,spec.subject=subject)
    (subject : ℕ) (member : subject ∈ selected) :
    ∃ spec ∈ selectedEntries selected entries,spec.subject=subject := by
  obtain ⟨spec,present,same⟩ := coverage subject member
  exact ⟨spec,(every_selected_mapping_entry_is_an_actual_selected_subject_entry selected entries spec).mpr
    ⟨present,by simpa [same] using member⟩,same⟩

theorem literal_loader_target_uniqueness_follows_from_the_casefold_check
    (fold : String → String) (entries : List Spec)
    (unique : (entries.map (fun spec => fold spec.target)).Nodup) : (entries.map Spec.target).Nodup := by
  have same : (entries.map Spec.target).map fold=entries.map (fun spec => fold spec.target) := by simp
  rw [←same] at unique
  exact List.Nodup.of_map fold unique

/- snapshot_for requires exactly one matching original source, then separately
reverifies that immutable object's descriptor. RA1 supplies the source-produced
inventory. This lookup does not make digest equality into source-key equality. -/
def snapshotFor (relative : String) (snapshots : List Snapshot) : Option Snapshot :=
  match snapshots.filter (fun snapshot => decide (snapshot.sourcePath=relative)) with
  | [snapshot] => some snapshot
  | _ => none

theorem successful_snapshot_lookup_retains_the_exact_original_source_key
    (relative : String) (snapshots : List Snapshot) (snapshot : Snapshot)
    (success : snapshotFor relative snapshots=some snapshot) :
    snapshot ∈ snapshots ∧ snapshot.sourcePath=relative := by
  unfold snapshotFor at success
  cases filtered : snapshots.filter (fun entry => decide (entry.sourcePath=relative)) with
  | nil => simp [filtered] at success
  | cons first rest =>
    cases rest with
    | cons second tail => simp [filtered] at success
    | nil =>
      simp [filtered] at success
      subst snapshot
      have present : first ∈ snapshots.filter (fun entry => decide (entry.sourcePath=relative)) := by rw [filtered]; simp
      simpa using (List.mem_filter.mp present)

def directLoader (manifest : List File) (snapshots : List Snapshot) (spec : Spec) : Option Loader :=
  match spec.source with
  | .zip _ _ => none
  | .direct relative =>
    if ∃ file ∈ manifest,file.path=relative then
      (snapshotFor relative snapshots).map (fun snapshot => ⟨spec.subject,spec.target,spec.source,snapshot⟩)
    else none

theorem direct_mapping_uses_the_manifest_source_and_its_snapshot_even_at_a_renamed_target
    (manifest : List File) (snapshots : List Snapshot) (subject : ℕ) (source target : String) (snapshot : Snapshot)
    (declared : ∃ file ∈ manifest,file.path=source) (chosen : snapshotFor source snapshots=some snapshot) :
    directLoader manifest snapshots ⟨subject,target,.direct source⟩=
      some ⟨subject,target,.direct source,snapshot⟩ ∧ snapshot.sourcePath=source := by
  exact ⟨by simp [directLoader,declared,chosen],
    (successful_snapshot_lookup_retains_the_exact_original_source_key source snapshots snapshot chosen).2⟩

theorem an_unmanifested_direct_source_is_refused_before_any_snapshot_lookup
    (manifest : List File) (snapshots : List Snapshot) (subject : ℕ) (source target : String)
    (missing : ¬∃ file ∈ manifest,file.path=source) :
    directLoader manifest snapshots ⟨subject,target,.direct source⟩=none := by simp [directLoader,missing]

def bySubject (entries : List Spec) (subject : ℕ) : List String :=
  (entries.filter (fun entry => decide (entry.subject=subject))).map Spec.target

def reportedBoundPaths (selected : List ℕ) (entries : List Spec) : List String :=
  selected.flatMap (bySubject entries)

theorem bound_report_membership_is_exactly_a_path_of_a_selected_subject
    (selected : List ℕ) (entries : List Spec) (path : String) :
    path ∈ reportedBoundPaths selected entries ↔
      ∃ spec ∈ entries,spec.subject ∈ selected ∧ spec.target=path := by
  simp only [reportedBoundPaths,List.mem_flatMap,bySubject,List.mem_map,List.mem_filter,decide_eq_true_eq]
  constructor
  · rintro ⟨subject,selected,entry,⟨member,same⟩,target⟩
    exact ⟨entry,member,by simpa [same] using selected,target⟩
  · rintro ⟨entry,member,selected,target⟩
    exact ⟨entry.subject,selected,entry,⟨member,rfl⟩,target⟩

theorem actual_bound_data_path_reports_the_complete_selected_target_set
    (selected : List ℕ) (entries : List Spec) (path : String) :
    path ∈ reportedBoundPaths selected entries ↔ path ∈ (selectedEntries selected entries).map Spec.target := by
  rw [bound_report_membership_is_exactly_a_path_of_a_selected_subject]
  simp only [List.mem_map,every_selected_mapping_entry_is_an_actual_selected_subject_entry]
  aesop

end
end N2P3.RawLoaderMapping
