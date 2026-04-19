# Phase 1: method comparison — the core results table (~100 experiments)
python sweep.py --phase method

# Phase 2: rank ablation — H3 in your proposal (~6 experiments)
python sweep.py --phase rank

# Phase 3: placement ablation — H2 in your proposal (~4 experiments)
python sweep.py --phase placement