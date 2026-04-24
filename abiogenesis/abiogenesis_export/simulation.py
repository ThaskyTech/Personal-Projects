"""
simulation.py
-------------
The top-level simulation loop.

Ties together:
  MoleculePool    – the population
  RuleEngine      – your deterministic chemistry
  StochasticLayer – the random noise layer
  MetricsEngine   – measurement + emergence detection
  Dashboard       – live plots

Usage (quick start):
  python simulation.py

Usage (custom):
  from simulation import Simulation, SimConfig
  from reactions import Reaction, RuleEngine, StochasticLayer, starter_reactions

  cfg = SimConfig(population=500_000, mol_length=16, ticks=500)
  sim = Simulation(cfg)
  sim.rule_engine = starter_reactions()
  sim.stochastic.mutation_rate = 1e-3   # crank up the chaos
  sim.run()
"""

from __future__ import annotations
import time, sys, os
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from molecules import MoleculePool, DEFAULT_LEN, DEFAULT_POP
from reactions import RuleEngine, StochasticLayer, starter_reactions
from metrics   import MetricsEngine
from visualiser import Dashboard


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    population   : int   = 500_000    # starting molecule count (scale down for testing)
    mol_length   : int   = DEFAULT_LEN # bits per molecule
    ticks        : int   = 300         # total simulation ticks
    death_rate   : float = 1e-4        # per-molecule per-tick death probability
    max_pop      : int   = 2_000_000   # hard cap — triggers culling above this
    plot_every   : int   = 5           # redraw dashboard every N ticks
    save_every   : int   = 50          # save PNG every N ticks (0 = never)
    seed         : int | None = None   # RNG seed for reproducibility
    output_dir   : str   = "output"
    verbose      : bool  = True

    # Motifs to watch in the event log (name → bit pattern)
    watch_motifs : dict[str, list[int]] = field(default_factory=lambda: {
        "palindrome_4" : [1,0,0,1],
        "all_ones_8"   : [1,1,1,1,1,1,1,1],
        "alternating"  : [1,0,1,0,1,0,1,0],
    })


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

class Simulation:

    def __init__(self, cfg: SimConfig = SimConfig()):
        self.cfg       = cfg
        self.rng       = np.random.default_rng(cfg.seed)
        self.pool      = MoleculePool(cfg.population, cfg.mol_length, self.rng)
        self.rule_engine  = starter_reactions()
        self.stochastic   = StochasticLayer()
        self.metrics_eng  = MetricsEngine(watch_motifs=cfg.watch_motifs)
        self.dashboard    = Dashboard(output_dir=cfg.output_dir,
                                      save_every=cfg.save_every,
                                      show=True)
        self._rule_label_history : list[dict[str, int]] = []
        self._stoch_history      : list[dict[str, int]] = []
        self.tick = 0

    # ── environment controls ──────────────────────────────────────────────────

    def _apply_death(self):
        """
        Stochastic death: each molecule has a small chance of being removed
        each tick.  This keeps population size roughly stable even as
        new molecules are created.
        """
        active_idx = np.where(self.pool.active)[0]
        if len(active_idx) == 0:
            return
        die_mask_local = self.rng.random(len(active_idx)) < self.cfg.death_rate
        die_idx        = active_idx[die_mask_local]
        if len(die_idx) > 0:
            mask = np.zeros(self.pool._cap, dtype=bool)
            mask[die_idx] = True
            self.pool.kill(mask)

    def _enforce_population_cap(self):
        """
        If population explodes past the cap, cull the oldest molecules.
        This models resource competition / carrying capacity.
        """
        if self.pool.count <= self.cfg.max_pop:
            return
        excess     = self.pool.count - self.cfg.max_pop
        active_idx = np.where(self.pool.active)[0]
        ages       = self.pool.age[active_idx]
        # kill the oldest excess molecules
        kill_local = np.argsort(ages)[-excess:]
        kill_global = active_idx[kill_local]
        mask = np.zeros(self.pool._cap, dtype=bool)
        mask[kill_global] = True
        self.pool.kill(mask)

    # ── main tick ─────────────────────────────────────────────────────────────

    def _tick(self):
        # 1. age everyone
        self.pool.step_age()

        # 2. deterministic rules
        rule_fires = self.rule_engine.tick(self.pool, self.rng)

        # 3. stochastic layer
        stoch_events = self.stochastic.tick(self.pool, self.rng)

        # 4. death
        self._apply_death()

        # 5. population cap
        self._enforce_population_cap()

        # 6. record for plotting
        self._rule_label_history.append(rule_fires)
        self._stoch_history.append(stoch_events)

    # ── run loop ──────────────────────────────────────────────────────────────

    def run(self):
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"  Abiogenesis Simulator")
        print(f"  population={cfg.population:,}  length={cfg.mol_length}  ticks={cfg.ticks}")
        print(f"  rules: {[r.label for r in self.rule_engine.rules]}")
        print(f"{'='*60}\n")

        for t in range(1, cfg.ticks + 1):
            self.tick = t
            t0 = time.perf_counter()
            self._tick()
            elapsed = time.perf_counter() - t0

            # build extended arrays for visualiser
            live_seqs  = self.pool.live_seqs()
            rule_fires = self._rule_label_history[-1]
            stoch_ev   = self._stoch_history[-1]
            snap       = self.metrics_eng.snapshot(t, live_seqs,
                                                    rule_fires, stoch_ev)

            # inject stochastic / rule time-series into arrays dict
            # (so the visualiser can access them)
            self._patch_arrays()

            if cfg.verbose:
                self._print_tick(t, snap, elapsed)

            for ev in snap.events:
                print(f"\n  *** {ev}")

            if t % cfg.plot_every == 0:
                self.dashboard.update(self.metrics_eng, t)

        # final save
        final_path = Path(cfg.output_dir) / "final.png"
        self.dashboard.final_save(final_path)
        print(f"\n[sim] done.  Output saved to {cfg.output_dir}/")
        self._print_summary()

        # keep window open
        try:
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.show()
        except Exception:
            pass

    def _patch_arrays(self):
        """
        Inject per-tick rule-fire and stochastic series into the metrics
        engine's history so the visualiser can pick them up.
        """
        arrays = self.metrics_eng.as_arrays()
        if not arrays:
            return
        n = len(self.metrics_eng.history)

        # rule labels
        rule_labels = [r.label for r in self.rule_engine.rules]
        arrays["_rule_labels"] = rule_labels
        for lbl in rule_labels:
            arrays[f"rule:{lbl}"] = [h.get(lbl, 0)
                                      for h in self._rule_label_history[-n:]]

        # stochastic keys
        stoch_keys = ["mutations", "ligations", "cleavages",
                      "template_copies", "cosmic_rays"]
        for key in stoch_keys:
            arrays[f"stoch:{key}"] = [h.get(key, 0)
                                       for h in self._stoch_history[-n:]]

    def _print_tick(self, t, snap, elapsed):
        bar = "▓" * int(snap.entropy * 20) + "░" * (20 - int(snap.entropy * 20))
        print(f"  tick {t:>4}  "
              f"pop={snap.population:>8,}  "
              f"H={snap.entropy:.3f} [{bar}]  "
              f"LZ={snap.mean_complexity:.3f}  "
              f"auto={snap.autocatalytic:.4f}  "
              f"({elapsed*1000:.1f}ms)")

    def _print_summary(self):
        h   = self.metrics_eng.history
        print(f"\n{'─'*60}")
        print(f"  SUMMARY  ({len(h)} snapshots)")
        print(f"  Peak population : {max(s.population for s in h):,}")
        print(f"  Max entropy     : {max(s.entropy for s in h):.4f}")
        print(f"  Max complexity  : {max(s.mean_complexity for s in h):.4f}")
        print(f"  Max autocatal.  : {max(s.autocatalytic for s in h):.4f}")
        total_events = sum(len(s.events) for s in h)
        print(f"  Emergence events: {total_events}")
        print(f"{'─'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Quick-start configuration ─────────────────────────────────────────────
    # Adjust population/ticks to taste.
    # On a modern laptop with numpy only: 500k mols × 100 ticks ≈ 2–4 minutes.
    # With numba installed (pip install numba): 10x faster.
    # For genuine million-mol runs uncomment the larger values below.

    cfg = SimConfig(
        population = 100_000,       # ← change to 1_000_000 for large-scale
        mol_length = 16,
        ticks      = 200,           # ← change to 1000 for deep runs
        death_rate = 5e-4,
        max_pop    = 500_000,
        plot_every = 5,
        save_every = 25,
        seed       = 42,
        verbose    = True,
    )

    sim = Simulation(cfg)

    # ── Customise your chemistry ──────────────────────────────────────────────
    # Add or replace rules here.  Each Reaction takes:
    #   motif   – bit pattern that must be present
    #   product – what it transforms into
    #   prob    – probability per match per tick
    #   label   – shown in plots and event log
    #
    # Example: add a "killer" motif that converts 00001111 → 11110000
    # from reactions import Reaction
    # sim.rule_engine.add(Reaction([0,0,0,0,1,1,1,1],
    #                               [1,1,1,1,0,0,0,0],
    #                               prob=0.10,
    #                               label="inverter"))

    # ── Customise the stochastic layer ───────────────────────────────────────
    # These control the "beyond-your-rules" randomness.
    # Crank up mutation_rate to create a volatile early-earth environment.
    # Crank up copy_rate to promote rapid replication.
    sim.stochastic.mutation_rate   = 2e-4    # default: 1e-4
    sim.stochastic.copy_rate       = 5e-5    # default: 2e-5
    sim.stochastic.cosmic_rate     = 5e-7    # default: 1e-7

    sim.run()
