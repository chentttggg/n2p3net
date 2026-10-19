import N2P3.ERPToeplitz

/-! Independent errors in the actual taper loop and its positivity domain. -/

namespace N2P3.ERPToeplitz.Negative

open Matrix

-- The source divides every lag sum by T, including the outermost lag.
example : sourceLoop rankOneCovariance 0 (0,0) (2,0)=1 := by
  rw [source_rank_one_corner]
  norm_num

-- Ordinary diagonal averaging without the taper can destroy PSD.
example : rankOneCovariance.PosSemidef → (untapered rankOneCovariance).PosSemidef := by
  intro _
  simp only [untapered_not_posSemidef]

-- A nonzero multi-channel PSD input may retain an entirely absent channel.
example : missingChannel.PosSemidef → (sourceLoop missingChannel 0).PosDef := by
  intro _
  simp only [source_missing_channel_not_posDef]

-- For raw nonsymmetric input, the second lag-zero write transposes the block.
example : sourceLoop asymmetricBlock 0 (0,0) (0,1)=2 := by
  rw [diagonal_last_write_transposes]
  norm_num

end N2P3.ERPToeplitz.Negative
