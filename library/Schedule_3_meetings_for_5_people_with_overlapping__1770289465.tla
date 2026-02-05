---- MODULE Scheduler ----
EXTENDS Naturals, FiniteSets, TLC

\* CONCRETE CONSTANT DEFINITIONS (Replaces CONFIG file)
Meetings == {1, 2, 3}
Times    == {1, 2, 3}
People   == {1, 2, 3, 4, 5}

VARIABLES 
    schedule,   \* Function [Meetings -> Times \cup {0}]
    pc          \* Program counter

\* Data Definitions
M1_Attendees == {1, 3}
M2_Attendees == {2, 4}
M3_Attendees == {3, 5}

GetAttendees(m) == 
    CASE m = 1 -> M1_Attendees
      [] m = 2 -> M2_Attendees
      [] m = 3 -> M3_Attendees

\* Availability Logic (No-Fly Zones)
\* P1, P2 busy at T1. P3 busy at T2.
IsAvailable(p, t) == 
    /\ (p = 1 => t # 1)
    /\ (p = 2 => t # 1)
    /\ (p = 3 => t # 2)

\* Check if all attendees of meeting m are free at time t
MeetingCanOccur(m, t) ==
    \A p \in GetAttendees(m) : IsAvailable(p, t)

Init == 
    /\ schedule = [m \in Meetings |-> 0]
    /\ pc = "schedule"

\* Action: Assign meeting m to time t
Assign(m, t) ==
    /\ schedule[m] = 0                  \* Not yet scheduled
    /\ \A other \in Meetings : schedule[other] # t \* Time slot is free (Bijective constraint)
    /\ MeetingCanOccur(m, t)            \* Attendees are available
    /\ schedule' = [schedule EXCEPT ![m] = t]
    /\ pc' = IF \A k \in Meetings : schedule'[k] # 0 THEN "done" ELSE "schedule"

Next == 
    \/ \E m \in Meetings, t \in Times : Assign(m, t)
    \/ (pc = "done" /\ UNCHANGED <<schedule, pc>>)

\* Invariants
TypeOK == 
    /\ schedule \in [Meetings -> Times \cup {0}]
    /\ pc \in {"schedule", "done"}

\* Safety: At "done", the schedule must be valid
ValidSchedule == 
    (pc = "done") => 
        /\ \A m1, m2 \in Meetings : (m1 # m2) => schedule[m1] # schedule[m2]
        /\ \A m \in Meetings : MeetingCanOccur(m, schedule[m])

Spec == Init /\ [][Next]_<<schedule, pc>>
====