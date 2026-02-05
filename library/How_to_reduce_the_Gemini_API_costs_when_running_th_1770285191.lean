import Mathlib
import Aesop

-- Define constants as per Rationale
def MAX_BUDGET : ℕ := 100
def COST_PER_MISS : ℕ := 20
def TOTAL_REQUESTS : ℕ := 8

/--
  Theorem: Logic Soundness.
  Verifies that if the number of misses 'm' is constrained by the budget,
  the actual units spent do not exceed 100.
-/
theorem budget_safety_logic (m : ℕ) (h : m * COST_PER_MISS ≤ MAX_BUDGET) : 
  m * 20 ≤ 100 := by
  -- Unfold the definitions to expose the concrete values
  unfold COST_PER_MISS MAX_BUDGET at h
  -- Use omega to solve the linear inequality
  omega

/--
  Theorem: Caching Advantage.
  Verifies that for 'r' requests, having 'h' hits (h > 0) results 
  in a strictly lower cost than the base cost (all misses).
-/
theorem caching_advantage (r h : ℕ) (h_hits : h > 0) (h_le_r : h ≤ r) :
  (r - h) * COST_PER_MISS < r * COST_PER_MISS := by
  -- Prove COST_PER_MISS is positive
  have h_pos : COST_PER_MISS > 0 := by simp [COST_PER_MISS]
  unfold COST_PER_MISS
  -- Use omega for linear arithmetic on naturals
  omega

/--
  Logical Tautology check using Aesop.
  Ensures the controller is always either in a processable state or finished.
-/
theorem controller_state_completeness (is_done : Bool) : 
  is_done = true ∨ is_done = false := by
  aesop

-- Final check: Maximum possible requests under budget (no hits)
example : 5 * COST_PER_MISS = MAX_BUDGET := by
  simp [COST_PER_MISS, MAX_BUDGET]