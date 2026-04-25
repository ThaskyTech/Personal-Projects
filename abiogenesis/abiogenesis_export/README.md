# Abiogenesis Simulator

A symbolic-string molecular evolution simulator built for scale (millions of
molecules) with a live matplotlib dashboard.

## Concept

Molecules are **binary strings** — e.g. `1011 0010 1100 0001`.  
The simulation models three layers of chemistry:

```
┌──────────────────────────────────────────────────────────────┐
│  Your rule engine          (deterministic chemistry)          │
│  "when this pattern appears, transform it to this"            │
├──────────────────────────────────────────────────────────────┤
│  Stochastic layer          (noise beyond your rules)          │
│  mutations · ligation · cleavage · template copying · cosmic  │
├──────────────────────────────────────────────────────────────┤
│  Population dynamics       (death · cap · resource pressure)  │
└──────────────────────────────────────────────────────────────┘
```

Products of reactions re-enter the pool, so complex molecules can
spontaneously accumulate — including self-complementary sequences
(the precursor to replication).

---

## Setup

```bash
pip install numpy matplotlib numba
```

`numba` is optional but gives ~10× speed on the hot loops.

---

## Run

```bash
cd abiogenesis
python simulation.py
```

A dashboard window will open with 6 live panels updating every 5 ticks.
PNGs are saved to `output/` every 25 ticks and a final summary on exit.

---

## File structure

```
abiogenesis/
├── molecules.py     – MoleculePool  (the population as numpy arrays)
├── reactions.py     – RuleEngine + StochasticLayer
├── metrics.py       – MetricsEngine + Snapshot + emergence detection
├── visualiser.py    – Dashboard (matplotlib live plots)
├── simulation.py    – SimConfig + Simulation + __main__
└── README.md
```

---

## Customising your chemistry

### Add a rule

In `simulation.py`, after `sim = Simulation(cfg)`:

```python
from reactions import Reaction

sim.rule_engine.add(Reaction(
    motif   = [1, 1, 0, 0, 1, 1],   # pattern to match
    product = [0, 0, 1, 1, 0, 0],   # what it becomes
    prob    = 0.08,                  # probability per match per tick
    label   = "my_enzyme",
))
```

### Tune the stochastic layer

```python
sim.stochastic.mutation_rate   = 5e-4   # bit flips — increase for volatile early Earth
sim.stochastic.ligation_rate   = 1e-5   # fusion of two molecules
sim.stochastic.cleavage_rate   = 2e-5   # splitting into halves
sim.stochastic.copy_rate       = 1e-4   # template/complement copying (proto-replication)
sim.stochastic.cosmic_rate     = 1e-6   # completely novel sequences injected
```

### Scale up

```python
cfg = SimConfig(
    population = 1_000_000,
    mol_length = 32,        # longer strings → more complex patterns
    ticks      = 1000,
    max_pop    = 5_000_000,
)
```

With `numba` installed, 1M × 32-bit × 1000 ticks runs in ~15 min on a
modern laptop.  Without numba, expect ~3× slower.

---

## What the dashboard shows

| Panel | What it tells you |
|---|---|
| Population | Are new molecules accumulating? Plateaus = equilibrium found |
| Entropy + GC | Bit diversity — high entropy = no dominant pattern yet |
| Complexity + Autocatalytic | Rising LZ complexity = longer, more structured molecules forming. Autocatalytic > 0 = complement pairs present (proto-replication!) |
| Novelty + Dominance | Novelty drop + dominance spike = a "winning" sequence taking over |
| Rule fires | Which of your rules is doing the most work |
| Stochastic events | How much the random layer is contributing |
| Event log | Timestamped emergence events |

---

## Emergence events

The metrics engine automatically logs when:
- Shannon entropy crosses 0.85 (the pool is highly diverse)
- Autocatalytic fraction exceeds 5% (complement pairing widespread)
- LZ complexity crosses 0.70 (structurally complex molecules dominant)
- A single sequence exceeds 5% of the population (a "species" forming)

These are the moments worth paying attention to.

---

## Biological parallels

| Simulator concept | Real biology |
|---|---|
| Bit flip mutation | Point mutation (A→G, C→T) |
| Ligation | RNA ligation, peptide bond formation |
| Cleavage | Ribozyme-catalysed strand cleavage |
| Template copy | RNA replication via complementary base pairing |
| Cosmic ray | UV-induced novel bond / random chemical injection |
| Dominant sequence > 5% | A proto-species emerging |
| Autocatalytic pair | The RNA World — strands that template their own replication |


NB: However this project was a failure since the exact simulation was not achieved, but I will continue to host it on this repository for anyone who is willing to improve and better it. Have fun with it :)
