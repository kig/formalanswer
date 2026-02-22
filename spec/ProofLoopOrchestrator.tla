----------------------- MODULE ProofLoopOrchestrator -----------------------
EXTENDS Naturals, TLC, FiniteSets

CONSTANTS MaxIterations

VARIABLES 
    state,          
    iteration,      
    verifiersPassed, 
    answerAccepted  

Verifiers == {"lean", "tla", "python"}
Vars == <<state, iteration, verifiersPassed, answerAccepted>>

Init ==
    /\ state = "IDLE"
    /\ iteration = 0
    /\ verifiersPassed = {}
    /\ answerAccepted = FALSE

Propose ==
    /\ (state = "IDLE" \/ state = "REPAIRING")
    /\ iteration < MaxIterations
    /\ state' = "PROPOSING"
    /\ iteration' = iteration + 1
    /\ UNCHANGED <<verifiersPassed, answerAccepted>>

Verify ==
    \/ /\ state = "PROPOSING"
       /\ state' = "VERIFYING"
       /\ UNCHANGED <<iteration, verifiersPassed, answerAccepted>>
    \/ /\ state = "VERIFYING"
       /\ \/ /\ verifiersPassed' = Verifiers
             /\ state' = "SUCCESS"
             /\ answerAccepted' = TRUE
          \/ /\ verifiersPassed' \in (SUBSET Verifiers \ {Verifiers})
             /\ state' = "REPAIRING"
             /\ UNCHANGED answerAccepted
       /\ UNCHANGED iteration

Terminate ==
    /\ iteration = MaxIterations
    /\ state /= "SUCCESS"
    /\ state' = "FAILURE"
    /\ UNCHANGED <<iteration, verifiersPassed, answerAccepted>>

Next == Propose \/ Verify \/ Terminate \/ (state \in {"SUCCESS", "FAILURE"} /\ UNCHANGED Vars)

Spec == Init /\ [][Next]_Vars /\ WF_Vars(Next)

Safety_VerificationSoundness == answerAccepted = TRUE => verifiersPassed = Verifiers
Safety_BudgetRespected == iteration <= MaxIterations
Liveness_Termination == <>(state = "SUCCESS" \/ state = "FAILURE")

=============================================================================
