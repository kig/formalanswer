import Mathlib
import Aesop

-- 1. Define Shared Constants
def FranceID : Nat := 1
def ParisID : Nat := 101

-- 2. Define the Knowledge Base (The Invariant Model)
def CapitalOf (country : Nat) : Nat :=
  if country == FranceID then ParisID else 0

-- 3. Verify Logical Soundness
-- We prove that querying FranceID strictly returns ParisID
theorem capital_of_france_is_paris : CapitalOf FranceID = ParisID := by
  -- Unfold definitions to expose the underlying values
  unfold CapitalOf FranceID ParisID
  -- Aesop automatically handles the conditional logic and equality
  aesop

-- 4. Verify Arithmetic Consistency
-- We ensure our constants are distinct and strictly ordered
theorem ids_are_distinct : ParisID > FranceID := by
  unfold ParisID FranceID
  -- Omega handles linear integer arithmetic
  omega