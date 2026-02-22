----------------------- MODULE ParallelRunner -----------------------
EXTENDS Naturals, TLC, Sequences, FiniteSets

CONSTANTS NumWorkers, NumTasks

VARIABLES 
    tasks,          
    results,        
    completed,      
    workerState     

Indices == 0..NumTasks-1

Vars == <<tasks, results, completed, workerState>>

Init ==
    /\ tasks = [i \in Indices |-> "code"]
    /\ results = [i \in Indices |-> "none"]
    /\ completed = {}
    /\ workerState = [i \in Indices |-> "idle"]

Submit(i) ==
    /\ workerState[i] = "idle"
    /\ workerState' = [workerState EXCEPT ![i] = "running"]
    /\ UNCHANGED <<tasks, results, completed>>

Finish(i) ==
    /\ workerState[i] = "running"
    /\ workerState' = [workerState EXCEPT ![i] = "finished"]
    /\ results' = [results EXCEPT ![i] = "done"]
    /\ completed' = completed \cup {i}
    /\ UNCHANGED tasks

Next == 
    \/ \E i \in Indices : Submit(i)
    \/ \E i \in Indices : Finish(i)
    \/ /\ completed = Indices 
       /\ UNCHANGED Vars

Spec == Init /\ [][Next]_Vars /\ WF_Vars(Next)

Safety_IndexAlignment == \A i \in completed : results[i] = "done"
Liveness_AllFinished == <>(completed = Indices)

=============================================================================