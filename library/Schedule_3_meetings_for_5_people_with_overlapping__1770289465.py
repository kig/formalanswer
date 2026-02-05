from z3 import *

m1, m2, m3 = Ints('m1 m2 m3')
s = Solver()

# Domain
s.add(And(m1 >= 1, m1 <= 3))
s.add(And(m2 >= 1, m2 <= 3))
s.add(And(m3 >= 1, m3 <= 3))
s.add(Distinct(m1, m2, m3))

# Availability
# M1(P1, P3): P1!=1, P3!=2
s.add(m1 != 1, m1 != 2)
# M2(P2, P4): P2!=1
s.add(m2 != 1)
# M3(P3, P5): P3!=2
s.add(m3 != 2)

if s.check() == sat:
    print(s.model())