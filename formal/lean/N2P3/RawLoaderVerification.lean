import N2P3.RawZipLoader

/-! RA2: distinguish the nonempty per-path materialization guard from the
worker's complete normcase-key set guard. Descriptor/hash/resolve operations
are independent observations. Private parser copies retain source metadata
while changing only the controlled physical root; no hash injectivity. -/
namespace N2P3.RawLoaderVerification
open N2P3.RawManifestSnapshots N2P3.RawLoaderMapping
noncomputable section
attribute [local instance] Classical.propDecidable

structure Contract where
  physical : String
  relative : String
  size : ℕ
  digest : String
  deriving DecidableEq, Repr

structure Read where
  regular : Bool
  before : ℕ
  after : ℕ
  size : ℕ
  digest : String
  deriving DecidableEq, Repr

def readMatches (contract : Contract) (observed : Read) : Prop :=
  observed.regular=true ∧ observed.before=observed.after ∧
    observed.size=contract.size ∧ observed.digest=contract.digest

def contractOf (join : String → String → String) (root : String) (loader : Loader) : Contract :=
  ⟨join root loader.target,loader.target,loader.snapshot.size,loader.snapshot.digest⟩

structure Report where
  relative : String
  contained : Bool
  physical : Bool
  observed : Read
  deriving DecidableEq, Repr

def weakEntry (contracts : List Contract) (report : Report) : Prop :=
  report.contained=true ∧ report.physical=true ∧
    ∃ contract ∈ contracts,contract.relative=report.relative ∧ readMatches contract report.observed

def weakVerify (contracts : List Contract) (reports : List Report) : Option Unit :=
  if reports≠[] ∧ ∀ report ∈ reports,weakEntry contracts report then some () else none

theorem weak_materialization_guard_checks_exactly_the_reported_nonempty_list
    (contracts : List Contract) (reports : List Report) :
    weakVerify contracts reports=some () ↔ reports≠[] ∧ ∀ report ∈ reports,weakEntry contracts report := by
  simp [weakVerify]

theorem empty_loader_report_is_refused_even_for_a_valid_mapping (contracts : List Contract) :
    weakVerify contracts []=none := by simp [weakVerify]

theorem one_reported_unmapped_or_changed_file_refuses_the_weak_guard
    (contracts : List Contract) (reports : List Report) (bad : Report)
    (present : bad ∈ reports) (failed : ¬weakEntry contracts bad) : weakVerify contracts reports=none := by
  have notReady : ¬(reports≠[] ∧ ∀ report ∈ reports,weakEntry contracts report) :=
    fun ready => failed (ready.2 bad present)
  simp [weakVerify,notReady]

def workerPathGate (normalize : String → String) (contracts : List Contract) (reported : List String) : Prop :=
  (contracts.map (fun contract => normalize contract.physical)).Nodup ∧
  (∀ path ∈ reported,normalize path ∈ contracts.map (fun contract => normalize contract.physical)) ∧
  (∀ contract ∈ contracts,normalize contract.physical ∈ reported.map normalize)

def workerVerify (normalize : String → String) (relative : String → Option String)
    (contracts : List Contract) (reported : List String) (observe : String → Read) : Option Unit :=
  if (∀ contract ∈ contracts,relative contract.physical=some contract.relative) ∧
      workerPathGate normalize contracts reported ∧
      (∀ contract ∈ contracts,readMatches contract (observe (normalize contract.physical)))
  then some () else none

theorem worker_guard_requires_every_expected_physical_key
    (normalize : String → String) (contracts : List Contract) (reported : List String)
    (gate : workerPathGate normalize contracts reported) :
    ∀ contract ∈ contracts,normalize contract.physical ∈ reported.map normalize := gate.2.2

theorem worker_guard_refuses_a_missing_expected_physical_key
    (normalize : String → String) (relative : String → Option String) (contracts : List Contract)
    (reported : List String) (observe : String → Read) (contract : Contract)
    (expected : contract ∈ contracts) (missing : normalize contract.physical ∉ reported.map normalize) :
    workerVerify normalize relative contracts reported observe=none := by
  have noGate : ¬workerPathGate normalize contracts reported := fun gate => missing (gate.2.2 contract expected)
  simp [workerVerify,noGate]

theorem worker_guard_uses_set_coverage_rather_than_report_multiplicity
    (normalize : String → String) (contracts : List Contract) (first second : List String)
    (sameKeys : ∀ key,key ∈ first.map normalize ↔ key ∈ second.map normalize) :
    workerPathGate normalize contracts first ↔ workerPathGate normalize contracts second := by
  have contains (paths : List String) :
      (∀ path ∈ paths,normalize path ∈ contracts.map (fun contract => normalize contract.physical)) ↔
      ∀ key ∈ paths.map normalize,key ∈ contracts.map (fun contract => normalize contract.physical) := by
    constructor
    · intro all key member
      obtain ⟨path,present,rfl⟩ := List.mem_map.mp member
      exact all path present
    · intro all path present
      exact all (normalize path) (List.mem_map.mpr ⟨path,present,rfl⟩)
  simp only [workerPathGate,contains]
  constructor <;> rintro ⟨unique,allowed,complete⟩
  · exact ⟨unique,fun key member => allowed key ((sameKeys key).mpr member),
      fun contract member => (sameKeys _).mp (complete contract member)⟩
  · exact ⟨unique,fun key member => allowed key ((sameKeys key).mp member),
      fun contract member => (sameKeys _).mpr (complete contract member)⟩

theorem worker_success_is_derived_from_independent_path_and_descriptor_observations
    (normalize : String → String) (relative : String → Option String) (contracts : List Contract)
    (reported : List String) (observe : String → Read)
    (paths : ∀ contract ∈ contracts,relative contract.physical=some contract.relative)
    (keys : workerPathGate normalize contracts reported)
    (reads : ∀ contract ∈ contracts,readMatches contract (observe (normalize contract.physical))) :
    workerVerify normalize relative contracts reported observe=some () := by
  exact if_pos ⟨paths,keys,reads⟩

def privateContract (destination : String) (contract : Contract) : Contract :=
  {contract with physical:=destination}

def privateCopy (destination : String) (inside physicalParents : Bool) (contract : Contract) (observed : Read) :
    Option Contract :=
  if inside=true ∧ physicalParents=true ∧ readMatches contract observed then some (privateContract destination contract) else none

theorem private_parser_copy_preserves_relative_path_size_and_digest
    (destination : String) (contract : Contract) :
    (privateContract destination contract).physical=destination ∧
      (privateContract destination contract).relative=contract.relative ∧
      (privateContract destination contract).size=contract.size ∧
      (privateContract destination contract).digest=contract.digest := by exact ⟨rfl,rfl,rfl,rfl⟩

theorem private_parser_copy_requires_the_actual_stable_source_read
    (destination : String) (contract : Contract) (observed : Read)
    (inside : Bool) (parents : Bool) (bad : ¬readMatches contract observed) :
    privateCopy destination inside parents contract observed=none := by simp [privateCopy,bad]

theorem independent_private_path_and_stream_checks_construct_the_new_contract
    (destination : String) (contract : Contract) (observed : Read)
    (read : readMatches contract observed) :
    privateCopy destination true true contract observed=some (privateContract destination contract) := by
  simp [privateCopy,read]

structure ParserFileRecord where
  sourceRelative : String
  parserRelative : String
  size : ℕ
  digest : String
  copiedFromStableDescriptor : Bool
  deriving DecidableEq, Repr

def parserFileRecord (contract : Contract) : ParserFileRecord :=
  ⟨contract.relative,"mne_data/"++contract.relative,contract.size,contract.digest,true⟩

def acceptWorkerFiles (requested returned : ℕ) (task : List Contract) (reported : List ParserFileRecord) : Option Unit :=
  if returned=requested ∧ reported=task.map parserFileRecord then some () else none

theorem worker_file_provenance_is_bound_to_its_original_ordered_task_contracts
    (subject : ℕ) (task : List Contract) :
    acceptWorkerFiles subject subject task (task.map parserFileRecord)=some () := by simp [acceptWorkerFiles]

theorem a_changed_private_digest_or_order_cannot_replace_task_provenance
    (requested returned : ℕ) (task : List Contract) (reported : List ParserFileRecord)
    (different : reported≠task.map parserFileRecord) : acceptWorkerFiles requested returned task reported=none := by
  simp [acceptWorkerFiles,different]

end
end N2P3.RawLoaderVerification
