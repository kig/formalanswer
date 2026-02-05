import Mathlib
import Aesop

-- Test linarith (Mathlib)
example (x y : Nat) (h : x > y) : x + 1 > y + 1 := by
  linarith

-- Test aesop (Aesop)
example (P Q : Prop) (h : P ∧ Q) : Q ∧ P := by
  aesop
