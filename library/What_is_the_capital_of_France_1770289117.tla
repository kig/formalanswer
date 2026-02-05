---- MODULE CapitalModel ----
EXTENDS Naturals, TLC

(* Concrete values from Rationale *)
FranceID == 1
ParisID == 101

VARIABLES state, result

(* States *)
Ready == "ready"
Done  == "done"

(* Type Invariant for Finite State Space *)
TypeOK == 
    /\ state \in {Ready, Done}
    /\ result \in {0, ParisID}

Init == 
    /\ state = Ready
    /\ result = 0

(* The "computation" step: retrieving the capital *)
Compute == 
    /\ state = Ready
    /\ state' = Done
    /\ result' = ParisID

Next == Compute

(* Temporal Specification *)
Spec == Init /\ [][Next]_<<state, result>>

(* Safety Property: If the computation is done, the result must be Paris *)
Correctness == (state = Done) => (result = ParisID)
====