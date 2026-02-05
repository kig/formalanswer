import Mathlib
import Aesop

-- 1. Define the Domain
abbrev Person := Nat
abbrev Time := Nat
abbrev Meeting := Nat

-- 2. Define Availability (The "No-Fly" Zones)
def is_available (p : Person) (t : Time) : Prop :=
  match p with
  | 1 => t ≠ 1
  | 2 => t ≠ 1
  | 3 => t ≠ 2
  | _ => True 

-- 3. Define Meeting Composition (Attendees)
def attendees (m : Meeting) : List Person :=
  match m with
  | 1 => [1, 3] 
  | 2 => [2, 4] 
  | 3 => [3, 5] 
  | _ => []

-- 4. Define Validity Predicates
def meeting_valid_at (m : Meeting) (t : Time) : Prop :=
  ∀ p ∈ attendees m, is_available p t

def distinct_times (t1 t2 t3 : Time) : Prop :=
  t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3

-- 5. The Solution Candidate
def sol_m1 : Time := 3
def sol_m2 : Time := 2
def sol_m3 : Time := 1

-- 6. The Theorem: This solution satisfies all constraints
theorem schedule_is_safe : 
  (meeting_valid_at 1 sol_m1) ∧ 
  (meeting_valid_at 2 sol_m2) ∧ 
  (meeting_valid_at 3 sol_m3) ∧ 
  (distinct_times sol_m1 sol_m2 sol_m3) := by
  unfold meeting_valid_at attendees is_available distinct_times sol_m1 sol_m2 sol_m3
  aesop