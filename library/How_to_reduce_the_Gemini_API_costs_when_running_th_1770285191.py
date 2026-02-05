from z3 import *

# Variables
hits = Int('hits')
misses = Int('misses')
total_cost = Int('total_cost')

# Constants
MAX_BUDGET = 100
COST_PER_MISS = 20
TOTAL_REQS = 8

s = Solver()

# Constraints
s.add(hits >= 0, misses >= 0)
s.add(hits + misses == TOTAL_REQS)
s.add(total_cost == (misses * COST_PER_MISS))
s.add(total_cost <= MAX_BUDGET)

# Optimize: We want to find the minimum number of hits needed 
# to ensure the project is successful (completes all 8 requests)
results = []
while s.check() == sat:
    m = s.model()
    results.append((m[hits], m[misses], m[total_cost]))
    # Exclude this solution to find more
    s.add(Not(And(hits == m[hits], misses == m[misses])))

print("Valid Execution Profiles (Hits needed to stay within 100 units for 8 reqs):")
for h, m, c in results:
    print(f"Hits: {h}, Misses: {m}, Cost: {c}")