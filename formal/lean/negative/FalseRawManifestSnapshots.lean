import N2P3.RawManifestSnapshots

namespace N2P3.NegativeRawManifestSnapshots
open N2P3.RawManifestSnapshots

-- Same number of files is insufficient for source-path coverage.
example : [snapshotOf (sampleFile "a")].map Snapshot.sourcePath = [sampleFile "b"].map File.path := by
  have different := identical_file_counts_do_not_prove_snapshot_source_coverage.2
  simp only [different]

-- Distinct source identities need not occupy distinct content objects.
example : (snapshotOf (sampleFile "a")).objectPath ≠ (snapshotOf (sampleFile "b")).objectPath := by
  rw [distinct_sources_can_share_one_content_addressed_object.2]
  simp

-- No official MD5 means this checksum observation is optional.
example : ¬ optionalMD5 none (some "unexpected") := by
  simp [optionalMD5]

-- A good source copy cannot replace verification of the stored object.
example : oneFile (sampleFile "a") ⟨goodObservation, {goodObservation with digest := "changed"}⟩ =
    some (snapshotOf (sampleFile "a")) := by
  rw [correct_source_copy_cannot_bypass_a_changed_stored_object.2]
  simp

-- Matching SHA is not a substitute for the separate byte-size guard.
example : streamMatches (sampleFile "a") {goodObservation with size := 8} := by
  simp only [digest_equality_does_not_imply_the_expected_byte_size.2]

-- A self-consistent record still needs the external requested dataset class.
example : verifyManifest id (fun _ => true) (fun _ => true) (fun _ => true)
    "expected" "other" [sampleFile "a"] (fun _ => goodTrace) =
      some (attestationOf "other" [sampleFile "a"] [snapshotOf (sampleFile "a")]) := by
  rw [wrong_dataset_class_refuses_before_the_snapshot_inventory_is_published
    id (fun _ => true) (fun _ => true) (fun _ => true) "expected" "other"
    [sampleFile "a"] (fun _ => goodTrace) (by decide)]
  simp

-- An arbitrary digest interface supplies no byte-level injectivity theorem.
example : (fun _ : List ℕ => "digest") [1] = (fun _ : List ℕ => "digest") [2] →
    ([1] : List ℕ) = [2] := by
  simp

end N2P3.NegativeRawManifestSnapshots
