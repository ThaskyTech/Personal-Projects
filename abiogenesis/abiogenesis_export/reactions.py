"""
reactions.py
------------
Two-layer reaction system:

  Layer 1 – Rule Engine
      User-defined reactions encoded as motif → product transformations.
      Evaluated each tick via numpy vectorised substring matching.

  Layer 2 – Stochastic Layer
      Random events that operate *outside* the rule table, generating
      genuinely novel structures the rules did not anticipate:
        • point mutations     (bit flip)
        • ligation            (two molecules fuse into one longer one)
        • cleavage            (one molecule splits into two shorter ones)
        • template copying    (complementary strand synthesis — replication!)
        • cosmic-ray event    (low-probability random sequence injection)

Both layers are implemented as pure-numpy operations so Numba can
JIT-compile the hot inner loops when available.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from molecules import MoleculePool


# ─────────────────────────────────────────────────────────────────────────────
# Rule Engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Reaction:
    """
    A single user-defined reaction rule.

    motif     : list[int]  – the bit pattern that must appear in a molecule
    product   : list[int]  – replaces the motif when reaction fires
    prob      : float      – probability of firing per tick when motif present
    label     : str        – human-readable name shown in the event log
    """
    motif   : list[int]
    product : list[int]
    prob    : float = 0.05
    label   : str   = "unnamed"

    def __post_init__(self):
        self._motif   = np.array(self.motif,   dtype=np.uint8)
        self._product = np.array(self.product, dtype=np.uint8)
        self._mlen    = len(self._motif)
        self._plen    = len(self._product)

    def apply(self, pool: MoleculePool, rng: np.random.Generator) -> int:
        """
        Vectorised application over the whole active population.
        Returns the number of reactions that fired this tick.
        """
        seqs = pool.seqs
        active_idx = np.where(pool.active)[0]
        if len(active_idx) == 0:
            return 0

        L, mlen, plen = pool.L, self._mlen, self._plen
        if mlen > L:
            return 0

        fired = 0
        # slide motif across every possible position
        for start in range(L - mlen + 1):
            chunk = seqs[active_idx, start:start + mlen]     # (N, mlen)
            match  = np.all(chunk == self._motif, axis=1)    # (N,)
            candidates = active_idx[match]
            if len(candidates) == 0:
                continue
            trigger = rng.random(len(candidates)) < self.prob
            fire_idx = candidates[trigger]
            if len(fire_idx) == 0:
                continue
            # apply replacement in-place, padding/truncating as needed
            end = start + plen
            if end <= L:
                seqs[fire_idx, start:end] = self._product
            else:
                # product longer than space: truncate to fit
                seqs[fire_idx, start:L] = self._product[:L - start]
            fired += len(fire_idx)

        return fired


class RuleEngine:
    """Holds all user-defined reactions and applies them each tick."""

    def __init__(self):
        self.rules: list[Reaction] = []

    def add(self, reaction: Reaction):
        self.rules.append(reaction)
        return self

    def tick(self, pool: MoleculePool, rng: np.random.Generator) -> dict[str, int]:
        counts = {}
        for r in self.rules:
            counts[r.label] = r.apply(pool, rng)
        return counts


# ─────────────────────────────────────────────────────────────────────────────
# Stochastic Layer  (the "beyond-your-rules" chaos)
# ─────────────────────────────────────────────────────────────────────────────

class StochasticLayer:
    """
    Random events that fire independently of the rule table.
    Each event type has its own rate parameter (events per molecule per tick).

    Parameters
    ----------
    mutation_rate    : prob of a single bit flip per molecule per tick
    ligation_rate    : prob that two random molecules fuse into one
    cleavage_rate    : prob that a molecule splits into two halves
    copy_rate        : prob that a molecule copies itself (template replication)
    cosmic_rate      : prob of a completely new random molecule appearing
    """

    def __init__(self,
                 mutation_rate : float = 1e-4,
                 ligation_rate : float = 5e-6,
                 cleavage_rate : float = 1e-5,
                 copy_rate     : float = 2e-5,
                 cosmic_rate   : float = 1e-7):
        self.mutation_rate  = mutation_rate
        self.ligation_rate  = ligation_rate
        self.cleavage_rate  = cleavage_rate
        self.copy_rate      = copy_rate
        self.cosmic_rate    = cosmic_rate

    # ── individual stochastic events ─────────────────────────────────────────

    def _mutations(self, pool: MoleculePool, rng: np.random.Generator) -> int:
        """Point mutations: random bit flips across the whole population."""
        seqs = pool.seqs
        active = pool.active
        # probability matrix: flip each bit independently
        flip_mask = rng.random(seqs.shape) < self.mutation_rate
        flip_mask[~active] = False          # don't touch dead slots
        seqs[flip_mask] ^= 1                # XOR flip
        return int(flip_mask.sum())

    def _ligation(self, pool: MoleculePool, rng: np.random.Generator) -> int:
        """
        Ligation: pick random pairs of active molecules.
        Fuse them: take the left half of A + right half of B → new molecule
        (both originals survive; the ligation product is inserted separately).
        """
        active_idx = np.where(pool.active)[0]
        n = len(active_idx)
        if n < 2:
            return 0
        n_events = max(1, int(n * self.ligation_rate))
        # pick random pairs (with replacement — duplicates are fine)
        ia = rng.choice(active_idx, n_events, replace=True)
        ib = rng.choice(active_idx, n_events, replace=True)
        same = ia == ib
        ib[same] = (ib[same] + 1) % n
        ib = active_idx[ib % len(active_idx)]   # re-map to valid idx

        L = pool.L
        half = L // 2
        new_seqs = np.empty((n_events, L), dtype=np.uint8)
        new_seqs[:, :half] = pool.seqs[ia, :half]
        new_seqs[:, half:] = pool.seqs[ib, half:]
        pool.add(new_seqs)
        return n_events

    def _cleavage(self, pool: MoleculePool, rng: np.random.Generator) -> int:
        """
        Cleavage: split a molecule into two halves.
        Both children get the first or second half padded with zeros.
        """
        active_idx = np.where(pool.active)[0]
        n = len(active_idx)
        n_events = max(1, int(n * self.cleavage_rate))
        chosen = rng.choice(active_idx, n_events, replace=False
                            if n_events <= n else True)
        L = pool.L
        half = L // 2
        left  = np.zeros((n_events, L), dtype=np.uint8)
        right = np.zeros((n_events, L), dtype=np.uint8)
        left[:, :half]  = pool.seqs[chosen, :half]
        right[:, :half] = pool.seqs[chosen, half:]
        # kill originals, insert two children
        pool.kill(np.isin(np.arange(pool._cap), chosen))
        pool.add(left)
        pool.add(right)
        return n_events

    def _template_copy(self, pool: MoleculePool, rng: np.random.Generator) -> int:
        """
        Template copying: produce the bitwise complement of a molecule.
        This is the simplest model of replication — the complement acts as
        a template strand, mirroring what RNA does with nucleotide pairing.
        """
        active_idx = np.where(pool.active)[0]
        n = len(active_idx)
        n_events = max(1, int(n * self.copy_rate))
        chosen = rng.choice(active_idx, n_events, replace=True)
        complements = 1 - pool.seqs[chosen]     # bitwise NOT for binary
        pool.add(complements)
        return n_events

    def _cosmic_rays(self, pool: MoleculePool, rng: np.random.Generator) -> int:
        """
        Cosmic-ray events: spontaneously inject completely novel molecules.
        Rate is per *total population* not per molecule, so it stays rare.
        """
        n_events = max(1, int(pool.count * self.cosmic_rate))
        new_seqs = rng.integers(0, 2, size=(n_events, pool.L), dtype=np.uint8)
        pool.add(new_seqs)
        return n_events

    # ── main tick ─────────────────────────────────────────────────────────────

    def tick(self, pool: MoleculePool, rng: np.random.Generator) -> dict[str, int]:
        return {
            "mutations"     : self._mutations(pool, rng),
            "ligations"     : self._ligation(pool, rng),
            "cleavages"     : self._cleavage(pool, rng),
            "template_copies": self._template_copy(pool, rng),
            "cosmic_rays"   : self._cosmic_rays(pool, rng),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pre-built reaction library  (examples you can use or modify)
# ─────────────────────────────────────────────────────────────────────────────

def starter_reactions() -> RuleEngine:
    """
    A small set of biologically inspired rules to get you started.

    Rule 1 – "Activator motif"
        If a molecule contains the pattern 1111 (four ones in a row),
        flip the bit immediately after it to 0 — simulating a repressor.

    Rule 2 – "Stabiliser motif"
        Alternating 1010 is a stable secondary structure.
        When found, the surrounding bits are nudged toward stability (→ 0101).

    Rule 3 – "Polymerase seed"
        The motif 110011 triggers ligation-friendly rewriting.
        Models a proto-enzyme that helps join monomers.
    """
    engine = RuleEngine()
    engine.add(Reaction([1,1,1,1],   [1,1,1,0],   prob=0.08, label="repressor"))
    engine.add(Reaction([1,0,1,0],   [0,1,0,1],   prob=0.03, label="stabiliser"))
    engine.add(Reaction([1,1,0,0,1,1], [0,1,1,0,1,0], prob=0.05, label="polymerase_seed"))
    return engine
