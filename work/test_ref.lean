import Mathlib.Data.Nat.Basic
import Mathlib.Tactic

theorem simple_add (n : ℕ) : n + 0 = n := by
  rw [Nat.add_zero]
