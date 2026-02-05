---- MODULE temp ----
EXTENDS Naturals, TLC

VARIABLES count, pc

Vars == <<count, pc>>

Init == count = 0 /\ pc = "start"

Next == 
  \/ pc = "start" /\ count < 5 /\ count' = count + 1 /\ pc' = "start"
  \/ pc = "start" /\ count = 5 /\ pc' = "done" /\ UNCHANGED count
  \/ pc = "done" /\ UNCHANGED Vars

TypeOK == count \in 0..5 /\ pc \in {"start", "done"}

Spec == Init /\ [][Next]_Vars
====
