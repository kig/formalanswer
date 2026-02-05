---- MODULE temp ----
EXTENDS Naturals, TLC

\* Shared Constants defined as operators for standalone verification
MaxBudget == 100
CostPerMiss == 20
TotalReqs == 8

VARIABLES spent, req_count, status

vars == <<spent, req_count, status>>

Init == 
    /\ spent = 0
    /\ req_count = 0
    /\ status = "READY"

Next ==
    \/ /\ status = "READY"
       /\ req_count < TotalReqs
       /\ \/ /\ spent + CostPerMiss <= MaxBudget \* Cache Miss
             /\ spent' = spent + CostPerMiss
             /\ req_count' = req_count + 1
             /\ status' = "READY"
          \/ /\ req_count' = req_count + 1         \* Cache Hit
             /\ spent' = spent
             /\ status' = "READY"
    \/ /\ status = "READY"
       /\ (req_count = TotalReqs \/ spent + CostPerMiss > MaxBudget)
       /\ status' = "DONE"
       /\ UNCHANGED <<spent, req_count>>
    \/ /\ status = "DONE"
       /\ UNCHANGED vars

\* Invariant: Total expenditure is always within MaxBudget
Safety == spent <= MaxBudget

\* Invariant: Finite state space bound for verification
TypeOK == 
    /\ spent \in 0..MaxBudget 
    /\ req_count \in 0..TotalReqs
    /\ status \in {"READY", "DONE"}

Spec == Init /\ [][Next]_vars
====