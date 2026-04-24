"""
molecules.py
------------
Core data structures for the symbolic-string molecule pool.

Molecules are fixed-length binary strings packed as uint8 numpy arrays.
The entire population lives in a single 2D array:  pool[i, :] = molecule i.
This layout lets numpy vectorise across millions of molecules cheaply.

String length is configurable.  Reactions operate on substrings (motifs).
"""

import numpy as np

# ── constants ────────────────────────────────────────────────────────────────
ALPHABET      = np.uint8([0, 1])          # binary alphabet
DEFAULT_LEN   = 16                         # bits per molecule
DEFAULT_POP   = 1_000_000                  # starting population size
MAX_POLYMER   = 64                         # cap on polymer length after ligation


# ── molecule pool ─────────────────────────────────────────────────────────────
class MoleculePool:
    """
    Holds the full population as a packed uint8 array of shape (N, L).

    Attributes
    ----------
    seqs   : ndarray (N, L)   – the sequences themselves
    energy : ndarray (N,)     – per-molecule free energy proxy (0.0–1.0)
    age    : ndarray (N,)     – ticks since molecule was created
    active : ndarray (N,)     – bool mask (False = slot is vacant / dead)
    L      : int              – sequence length
    """

    def __init__(self, population: int = DEFAULT_POP, length: int = DEFAULT_LEN,
                 rng: np.random.Generator | None = None):
        self.L     = length
        self.rng   = rng or np.random.default_rng()
        self._cap  = population                  # total allocated slots

        # initialise with random binary sequences
        self.seqs   = self.rng.integers(0, 2, size=(population, length), dtype=np.uint8)
        self.energy = self.rng.random(population).astype(np.float32)
        self.age    = np.zeros(population, dtype=np.int32)
        self.active = np.ones(population, dtype=bool)

    # ── bookkeeping ──────────────────────────────────────────────────────────
    @property
    def count(self) -> int:
        return int(self.active.sum())

    def step_age(self):
        self.age[self.active] += 1

    def kill(self, mask: np.ndarray):
        """Deactivate molecules where mask is True."""
        self.active[mask] = False

    def add(self, new_seqs: np.ndarray, new_energies: np.ndarray | None = None):
        """
        Insert new molecules by recycling vacant slots (or appending).
        new_seqs: (M, L) uint8 array.
        Returns the indices of the newly filled slots.
        """
        M = len(new_seqs)
        vacant = np.where(~self.active)[0]

        if len(vacant) >= M:
            slots = vacant[:M]
        else:
            # grow arrays if needed
            extra = M - len(vacant)
            pad   = np.zeros((extra, self.L), dtype=np.uint8)
            self.seqs   = np.vstack([self.seqs, pad])
            self.energy = np.concatenate([self.energy, np.zeros(extra, np.float32)])
            self.age    = np.concatenate([self.age,    np.zeros(extra, np.int32)])
            self.active = np.concatenate([self.active, np.zeros(extra, bool)])
            slots = np.concatenate([vacant, np.arange(self._cap, self._cap + extra)])
            self._cap  += extra

        self.seqs[slots]   = new_seqs
        self.age[slots]    = 0
        self.active[slots] = True
        if new_energies is not None:
            self.energy[slots] = new_energies
        else:
            self.energy[slots] = self.rng.random(M).astype(np.float32)

        return slots

    # ── convenience ──────────────────────────────────────────────────────────
    def live_seqs(self) -> np.ndarray:
        """Return a view of active sequences only (copy)."""
        return self.seqs[self.active]

    def seq_to_str(self, idx: int) -> str:
        return "".join(self.seqs[idx].astype(str))

    def __repr__(self):
        return f"<MoleculePool n={self.count}/{self._cap} L={self.L}>"
