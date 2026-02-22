import jax.numpy as jnp
from jax import random, vmap

# Shared Constants
NUM_SIMULATIONS = 10000
KEY = random.PRNGKey(42)

# Projects: [Moat, TAM, Hype]
# 0: ProofLoop, 1: AI Toolkit, 2: Softwiki, 3: AppTok
projects = jnp.array([
    [0.9, 0.7, 0.8],
    [0.2, 0.4, 0.3],
    [0.3, 0.3, 0.4],
    [0.6, 0.9, 0.9]
])

def simulate_vc_round(key):
    k1, k2, k3 = random.split(key, 3)
    
    # VC Preferences (Randomized weights)
    # Mean weights: Moat=0.4, TAM=0.3, Hype=0.3
    w_moat = 0.4 + 0.1 * random.normal(k1)
    w_tam  = 0.3 + 0.1 * random.normal(k2)
    w_hype = 0.3 + 0.1 * random.normal(k3)
    
    # Execution Noise
    noise = 0.1 * random.normal(k1, shape=projects.shape)
    p_adj = jnp.clip(projects + noise, 0.0, 1.0)
    
    scores = (w_moat * p_adj[:, 0] + 
              w_tam  * p_adj[:, 1] + 
              w_hype * p_adj[:, 2])
    
    return jnp.argmax(scores)

outcomes = vmap(simulate_vc_round)(random.split(KEY, NUM_SIMULATIONS))
probs = jnp.bincount(outcomes, length=4) / NUM_SIMULATIONS

print("Funding Probabilities (N=10,000):")
print(f"ProofLoop: {probs[0]*100:.1f}%")
print(f"AI Toolkit:   {probs[1]*100:.1f}%")
print(f"Softwiki:     {probs[2]*100:.1f}%")
print(f"AppTok:       {probs[3]*100:.1f}%")