import N2P3.TrainingCrossEntropyRisk

/-!
L01 actual resolve_source_risk_weights before its separately named float32
cast. G is the actual occupied (domain,unit) support; every G has original
rows, and every actual domain has a unit. These support facts are not desired
weight/normalization premises. Native cast replies may perturb multipliers;
generic actual-reply risk and its exact difference are kept separate.
-/
namespace N2P3.HierarchicalTrainingRisk
open scoped BigOperators
open N2P3.TrainingCrossEntropyRisk
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

def unitsIn {G D : Type*} [Fintype G] (owner : G → D) (domain : D) : Finset G :=
  Finset.univ.filter (fun unit => owner unit=domain)
def unitCount {G D : Type*} [Fintype G] (owner : G → D) (domain : D) : ℕ := (unitsIn owner domain).card
def unitClassMass {I G : Type*} [Fintype I] (unitOfRow : I → G) (weight : I → ℝ) (unit : G) : ℝ :=
  fiberSum unitOfRow weight unit
def multiplier {I G D : Type*} [Fintype I] [Fintype G]
    (unitOfRow : I → G) (owner : G → D) (domainMass : D → ℝ) (weight : I → ℝ) (row : I) : ℝ :=
  denominator weight*domainMass (owner (unitOfRow row))/
    ((unitCount owner (owner (unitOfRow row)) : ℝ)*unitClassMass unitOfRow weight (unitOfRow row))

theorem occupied_unit_has_positive_original_class_mass
    {I G : Type*} [Fintype I] (unitOfRow : I → G) (weight : I → ℝ)
    (occupied : Function.Surjective unitOfRow) (positive : ∀ row,0<weight row) (unit : G) :
    0<unitClassMass unitOfRow weight unit := by
  obtain ⟨row,same⟩ := occupied unit
  unfold unitClassMass fiberSum
  apply Finset.sum_pos'
  · intro candidate _
    split_ifs <;> first | exact le_of_lt (positive candidate) | exact le_rfl
  · exact ⟨row,Finset.mem_univ _,by simp [same,positive row]⟩

theorem each_actual_unit_derives_positive_domain_unit_count
    {G D : Type*} [Fintype G] (owner : G → D) (unit : G) : 0<unitCount owner (owner unit) := by
  apply Finset.card_pos.mpr
  exact ⟨unit,by simp [unitsIn]⟩

theorem each_actual_domain_derives_positive_unit_count
    {G D : Type*} [Fintype G] (owner : G → D) (actual : Function.Surjective owner) (domain : D) :
    0<unitCount owner domain := by
  obtain ⟨unit,same⟩ := actual domain
  rw [← same]
  exact each_actual_unit_derives_positive_domain_unit_count owner unit

theorem literal_source_multiplier_derives_the_hierarchical_row_coefficient
    {I G D : Type*} [Fintype I] [Fintype G] [Nonempty I]
    (unitOfRow : I → G) (owner : G → D) (domainMass : D → ℝ) (weight : I → ℝ)
    (occupied : Function.Surjective unitOfRow) (positive : ∀ row,0<weight row) (row : I) :
    coefficient weight (multiplier unitOfRow owner domainMass weight) row=
      (domainMass (owner (unitOfRow row))/(unitCount owner (owner (unitOfRow row)) : ℝ))*
        (weight row/unitClassMass unitOfRow weight (unitOfRow row)) := by
  have total := (actual_class_denominator_is_positive_on_nonempty_batches weight positive).ne'
  have cell := (occupied_unit_has_positive_original_class_mass unitOfRow weight occupied positive (unitOfRow row)).ne'
  have units : (unitCount owner (owner (unitOfRow row)) : ℝ) ≠ 0 :=
    Nat.cast_ne_zero.mpr (Nat.ne_of_gt (each_actual_unit_derives_positive_domain_unit_count owner (unitOfRow row)))
  unfold coefficient multiplier
  field_simp

theorem actual_hierarchical_row_coefficients_are_positive
    {I G D : Type*} [Fintype I] [Fintype G] [Nonempty I]
    (unitOfRow : I → G) (owner : G → D) (domainMass : D → ℝ) (weight : I → ℝ)
    (occupied : Function.Surjective unitOfRow) (positive : ∀ row,0<weight row)
    (massPositive : ∀ domain,0<domainMass domain) (row : I) :
    0<coefficient weight (multiplier unitOfRow owner domainMass weight) row := by
  rw [literal_source_multiplier_derives_the_hierarchical_row_coefficient unitOfRow owner domainMass weight occupied positive]
  apply mul_pos
  · exact div_pos (massPositive _) (Nat.cast_pos.mpr (each_actual_unit_derives_positive_domain_unit_count owner _))
  · exact div_pos (positive row) (occupied_unit_has_positive_original_class_mass unitOfRow weight occupied positive _)

theorem every_actual_unit_gets_its_domain_mass_over_actual_unit_count
    {I G D : Type*} [Fintype I] [Fintype G] [Nonempty I]
    (unitOfRow : I → G) (owner : G → D) (domainMass : D → ℝ) (weight : I → ℝ)
    (occupied : Function.Surjective unitOfRow) (positive : ∀ row,0<weight row) (unit : G) :
    fiberSum unitOfRow (coefficient weight (multiplier unitOfRow owner domainMass weight)) unit=
      domainMass (owner unit)/(unitCount owner (owner unit) : ℝ) := by
  have cell := (occupied_unit_has_positive_original_class_mass unitOfRow weight occupied positive unit).ne'
  calc
    _ = ((domainMass (owner unit)/(unitCount owner (owner unit) : ℝ))/
        unitClassMass unitOfRow weight unit)*unitClassMass unitOfRow weight unit := by
      simp only [unitClassMass,fiberSum,Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro row _
      by_cases same : unitOfRow row=unit
      · simp only [if_pos same]
        rw [literal_source_multiplier_derives_the_hierarchical_row_coefficient unitOfRow owner domainMass weight occupied positive,same]
        simp only [unitClassMass,fiberSum]
        ring
      · simp only [if_neg same,mul_zero]
    _ = _ := by field_simp

theorem every_actual_domain_gets_its_declared_mass_without_row_pooling
    {I G D : Type*} [Fintype I] [Fintype G] [Nonempty I]
    (unitOfRow : I → G) (owner : G → D) (domainMass : D → ℝ) (weight : I → ℝ)
    (occupied : Function.Surjective unitOfRow) (actual : Function.Surjective owner)
    (positive : ∀ row,0<weight row) (domain : D) :
    fiberSum owner (fun unit =>
      fiberSum unitOfRow (coefficient weight (multiplier unitOfRow owner domainMass weight)) unit) domain=
      domainMass domain := by
  have count : (unitCount owner domain : ℝ) ≠ 0 :=
    Nat.cast_ne_zero.mpr (Nat.ne_of_gt (each_actual_domain_derives_positive_unit_count owner actual domain))
  simp_rw [every_actual_unit_gets_its_domain_mass_over_actual_unit_count unitOfRow owner domainMass weight occupied positive]
  unfold fiberSum
  have equal : (∑ unit,if owner unit=domain then domainMass (owner unit)/(unitCount owner (owner unit) : ℝ) else 0)=
      ∑ unit,if owner unit=domain then domainMass domain/(unitCount owner domain : ℝ) else 0 := by
    apply Finset.sum_congr rfl
    intro unit _
    by_cases same : owner unit=domain <;> simp [same]
  rw [equal]
  simp only [Finset.sum_ite,Finset.sum_const,nsmul_eq_mul,mul_zero,add_zero]
  change (unitCount owner domain : ℝ)*(domainMass domain/(unitCount owner domain : ℝ))=domainMass domain
  field_simp

theorem global_coefficient_mass_is_the_SUM_of_actual_declared_domain_masses
    {I G D : Type*} [Fintype I] [Fintype G] [Fintype D] [Nonempty I]
    (unitOfRow : I → G) (owner : G → D) (domainMass : D → ℝ) (weight : I → ℝ)
    (occupied : Function.Surjective unitOfRow) (actual : Function.Surjective owner) (positive : ∀ row,0<weight row) :
    (∑ row,coefficient weight (multiplier unitOfRow owner domainMass weight) row)=∑ domain,domainMass domain := by
  rw [← actual_assignment_fibers_sum_each_original_row_exactly_once unitOfRow]
  rw [← actual_assignment_fibers_sum_each_original_row_exactly_once owner]
  apply Finset.sum_congr rfl
  intro domain _
  exact every_actual_domain_gets_its_declared_mass_without_row_pooling unitOfRow owner domainMass weight occupied actual positive domain

theorem normalized_domain_inputs_derive_unit_mass_as_a_conclusion
    {I G D : Type*} [Fintype I] [Fintype G] [Fintype D] [Nonempty I]
    (unitOfRow : I → G) (owner : G → D) (domainMass : D → ℝ) (weight : I → ℝ)
    (occupied : Function.Surjective unitOfRow) (actual : Function.Surjective owner) (positive : ∀ row,0<weight row)
    (normalized : (∑ domain,domainMass domain)=1) :
    (∑ row,coefficient weight (multiplier unitOfRow owner domainMass weight) row)=1 := by
  rw [global_coefficient_mass_is_the_SUM_of_actual_declared_domain_masses unitOfRow owner domainMass weight occupied actual positive,normalized]

theorem approximate_domain_mass_inputs_keep_their_actual_mass_residual
    {I G D : Type*} [Fintype I] [Fintype G] [Fintype D] [Nonempty I]
    (unitOfRow : I → G) (owner : G → D) (domainMass : D → ℝ) (weight : I → ℝ)
    (occupied : Function.Surjective unitOfRow) (actual : Function.Surjective owner) (positive : ∀ row,0<weight row)
    (tolerance : ℝ) (residual : |(∑ domain,domainMass domain)-1|≤tolerance) :
    |(∑ row,coefficient weight (multiplier unitOfRow owner domainMass weight) row)-1|≤tolerance := by
  rw [global_coefficient_mass_is_the_SUM_of_actual_declared_domain_masses unitOfRow owner domainMass weight occupied actual positive]
  exact residual

theorem observed_multiplier_cast_replies_have_an_explicit_exact_risk_difference
    {I : Type*} [Fintype I] (weight ideal observed loss : I → ℝ) :
    risk weight observed loss-risk weight ideal loss=risk weight (fun row => observed row-ideal row) loss := by
  simp only [risk,numerator,← sub_div,← Finset.sum_sub_distrib]
  congr 1
  apply Finset.sum_congr rfl
  intro row _
  ring

end
end N2P3.HierarchicalTrainingRisk
