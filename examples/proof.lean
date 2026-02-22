import Mathlib
import Aesop

structure Project where
  name : String
  moat : Real
  tam : Real
  hype : Real

def formalAnswer : Project := { name := "FormalAnswer", moat := 0.9, tam := 0.7, hype := 0.8 }
def aiToolkit    : Project := { name := "AI Toolkit",   moat := 0.2, tam := 0.4, hype := 0.3 }
def softwiki     : Project := { name := "Softwiki",     moat := 0.3, tam := 0.3, hype := 0.4 }
def appTok       : Project := { name := "AppTok",       moat := 0.6, tam := 0.9, hype := 0.9 }

/-- The VC Scoring Function: Moat is weighted highest (0.5) -/
def vcScore (p : Project) : Real :=
  0.5 * p.moat + 0.3 * p.tam + 0.2 * p.hype

theorem formal_answer_dominates_softwiki : vcScore formalAnswer > vcScore softwiki := by
  dsimp [vcScore, formalAnswer, softwiki]
  norm_num

theorem apptok_dominates_toolkit : vcScore appTok > vcScore aiToolkit := by
  dsimp [vcScore, appTok, aiToolkit]
  norm_num

/-- Prove FormalAnswer and AppTok are the top two contenders -/
theorem top_tier_projects : 
  vcScore formalAnswer > 0.8 ∧ vcScore appTok > 0.7 := by
  dsimp [vcScore, formalAnswer, appTok]
  norm_num