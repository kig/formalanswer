---- MODULE CertusProtocol ----
EXTENDS Naturals, TLC

(* Concrete Constants defined as Operators *)
ValRelevance   == 90
ValMemorability == 80
ValLength      == 6
Threshold      == 80
Limit          == 10

VARIABLES 
    state,      \* {"start", "verifying", "accepted", "rejected"}
    r, m, l     \* relevance, memorability, length

(* MANDATORY: Type Invariant for the harness *)
TypeOK == 
    /\ state \in {"start", "verifying", "accepted", "rejected"}
    /\ r \in Nat
    /\ m \in Nat
    /\ l \in Nat

Init == 
    /\ state = "start"
    /\ r = 0 
    /\ m = 0 
    /\ l = 0

(* Step 1: System ingests the candidate values *)
LoadCandidate == 
    /\ state = "start"
    /\ r' = ValRelevance
    /\ m' = ValMemorability
    /\ l' = ValLength
    /\ state' = "verifying"

(* Step 2: System verifies the metric *)
CheckSafety == 
    /\ state = "verifying"
    /\ UNCHANGED <<r, m, l>>
    /\ LET score == (r + m) \div 2 IN
       IF l <= Limit /\ score >= Threshold THEN
           state' = "accepted"
       ELSE
           state' = "rejected"

(* Step 3: Terminal state *)
Done == 
    /\ state \in {"accepted", "rejected"}
    /\ UNCHANGED <<state, r, m, l>>

Next == LoadCandidate \/ CheckSafety \/ Done

(* Safety Property: An accepted name MUST satisfy the math *)
SafeAccept == 
    (state = "accepted") => 
    ((l <= Limit) /\ ((r + m) \div 2 >= Threshold))

Spec == Init /\ [][Next]_<<state, r, m, l>>
====