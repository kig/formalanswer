---- MODULE VCFunding ----
EXTENDS Naturals, Integers, Sequences

\* DEFINITIONS
ProjectTypes == {"FormalAnswer", "AI_Toolkit", "Softwiki", "AppTok"}
Stages       == {"Pitch", "Diligence", "TermSheet", "Reject"}

\* THRESHOLDS
MinMoatScore == 6
MinTAM       == 20   \* Billions
HypeThreshold == 8

VARIABLES 
    candidate,         \* Current project being evaluated
    stage,             \* Current stage in the funnel
    metrics            \* Record: [moat, tam, hype]

Vars == <<candidate, stage, metrics>>

\* Attributes logic: Maps project names to their scores
Attributes(p) ==
    CASE p = "FormalAnswer" -> [moat |-> 9, tam |-> 50, hype |-> 8]
      [] p = "AppTok"       -> [moat |-> 6, tam |-> 100, hype |-> 9]
      [] p = "Softwiki"     -> [moat |-> 3, tam |-> 10, hype |-> 4]
      [] p = "AI_Toolkit"   -> [moat |-> 2, tam |-> 5, hype |-> 3]
      [] OTHER              -> [moat |-> 0, tam |-> 0, hype |-> 0]

\* The Type Correctness Invariant (Required for Verification)
TypeOK ==
    /\ candidate \in ProjectTypes
    /\ stage \in Stages
    /\ metrics \in [moat : Int, tam : Int, hype : Int]

\* Initial State
Init ==
    /\ candidate \in ProjectTypes
    /\ stage = "Pitch"
    /\ metrics = Attributes(candidate)

\* State Transition: Pitch Phase
\* Filter based on TAM or Hype
EvaluatePitch ==
    /\ stage = "Pitch"
    /\ IF (metrics.tam >= MinTAM) \/ (metrics.hype >= HypeThreshold)
       THEN stage' = "Diligence"
       ELSE stage' = "Reject"
    /\ UNCHANGED <<candidate, metrics>>

\* State Transition: Diligence Phase
\* Filter based on Moat (Defensibility)
DoDiligence ==
    /\ stage = "Diligence"
    /\ IF metrics.moat >= MinMoatScore
       THEN stage' = "TermSheet"
       ELSE stage' = "Reject"
    /\ UNCHANGED <<candidate, metrics>>

Next == EvaluatePitch \/ DoDiligence

Spec == Init /\ [][Next]_Vars

\* Logic Safety: A Term Sheet implies the project has a sufficient Moat.
FundingInvariant == 
    (stage = "TermSheet") => (metrics.moat >= MinMoatScore)
====