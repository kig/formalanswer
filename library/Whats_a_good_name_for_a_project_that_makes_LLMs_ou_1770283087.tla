---- MODULE temp ----
EXTENDS Naturals, TLC

\* Concrete value from Rationale
MaxAttempts == 5

VARIABLES attempts, status

Init == 
    /\ attempts = 0 
    /\ status = "IDLE"

Next == 
    \/ /\ status = "IDLE"
       /\ attempts < MaxAttempts
       /\ status' = "VERIFYING"
       /\ attempts' = attempts + 1
    \/ /\ status = "VERIFYING"
       /\ \/ status' = "SUCCESS"
          \/ (attempts < MaxAttempts /\ status' = "IDLE")
          \/ (attempts = MaxAttempts /\ status' = "EXHAUSTED")
       /\ UNCHANGED attempts
    \/ /\ (status = "SUCCESS" \/ status = "EXHAUSTED")
       /\ UNCHANGED <<attempts, status>>

\* Temporal Safety Property
TypeOK == 
    /\ attempts \in 0..MaxAttempts 
    /\ status \in {"IDLE", "VERIFYING", "SUCCESS", "EXHAUSTED"}

\* The invariant that must always hold
Safety == attempts <= MaxAttempts

Spec == Init /\ [][Next]_<<attempts, status>>
====