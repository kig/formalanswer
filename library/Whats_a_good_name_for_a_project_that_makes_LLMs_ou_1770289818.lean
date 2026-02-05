import Mathlib
import Aesop

-- 1. Define the Shared Constants
def c_len : Nat := 6
def c_rel : Nat := 90
def c_mem : Nat := 80
def c_thresh : Nat := 80
def c_limit : Nat := 10

-- 2. Define the Invariant
-- The project name is valid if length is within limit AND score meets threshold.
def valid_project_name (len rel mem thresh limit : Nat) : Prop :=
  len ≤ limit ∧ (rel + mem) / 2 ≥ thresh

-- 3. The Proof
-- We prove that "Certus" satisfies these bounds.
theorem certus_verified : valid_project_name c_len c_rel c_mem c_thresh c_limit := by
  -- Expand definitions
  unfold valid_project_name c_len c_rel c_mem c_thresh c_limit
  
  -- The goal is: 6 ≤ 10 ∧ (90 + 80) / 2 ≥ 80
  
  -- Use Aesop to break the logic into subgoals
  aesop
  
  -- Subgoal 1: 6 ≤ 10
  -- Subgoal 2: 85 ≥ 80
  -- Solved by omega (Presburger arithmetic)
  all_goals omega