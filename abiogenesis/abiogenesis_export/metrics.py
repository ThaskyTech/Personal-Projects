"""
metrics.py
----------
What to measure — and how to detect emergence.

Computed each tick over the active population:

  Shannon entropy   – information diversity of the pool
  GC content        – fraction of 1-bits (analogous to GC in real DNA)
  Mean complexity   – Lempel-Ziv complexity proxy per molecule
  Motif frequency   – count of interesting patterns (user-defined)
  Autocatalysis     – proportion of molecules that match their own complement
  Dominance         – frequency of the single most-common sequence
  Novelty rate      – fraction of sequences not seen in the previous tick

Emergence events:
  An "emergence event" is logged when a metric crosses a threshold
  for the first time, or when a novel dominant sequence appears.
"""

from __future__ import annotations
import numpy as np
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Snapshot:
    tick           : int
    population     : int
    entropy        : float          # Shannon entropy over 1-bit frequencies
    gc_content     : float          # mean fraction of 1-bits
    mean_complexity: float          # LZ complexity proxy (0–1)
    autocatalytic  : float          # fraction complement-matches own type
    dominance      : float          # frequency of most-common sequence
    novelty_rate   : float          # fraction of sequences new this tick
    motif_counts   : dict[str, int] = field(default_factory=dict)
    rule_fires     : dict[str, int] = field(default_factory=dict)
    stochastic     : dict[str, int] = field(default_factory=dict)
    events         : list[str]      = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Complexity estimator  (LZ-proxy — fast, no compression library needed)
# ─────────────────────────────────────────────────────────────────────────────

def _lz_complexity(seq: np.ndarray) -> float:
    """
    Lempel-Ziv complexity of a 1D uint8 binary sequence.
    Returns a float in [0, 1] normalised by the theoretical maximum.
    """
    n = len(seq)
    i, c, l = 0, 1, 1
    while i + l <= n:
        # try to find seq[i:i+l] in seq[0:i+l-1]
        pattern = seq[i:i+l]
        found = False
        for j in range(i):
            if j + l <= n and np.array_equal(seq[j:j+l], pattern):
                found = True
                break
        if found:
            l += 1
        else:
            c += 1
            i += l
            l = 1
    # normalise
    if n > 1:
        return c / (n / np.log2(n + 1e-9))
    return 0.0


def batch_lz_complexity(seqs: np.ndarray, sample: int = 2000) -> float:
    """
    Estimate mean LZ complexity on a random sample of the population.
    Full LZ over millions is expensive — sampling gives a good proxy.
    """
    n = len(seqs)
    if n == 0:
        return 0.0
    idx = np.random.choice(n, min(sample, n), replace=False)
    return float(np.mean([_lz_complexity(seqs[i]) for i in idx]))


# ─────────────────────────────────────────────────────────────────────────────
# Metrics engine
# ─────────────────────────────────────────────────────────────────────────────

class MetricsEngine:
    """
    Computes a Snapshot each tick and tracks emergence events.

    Parameters
    ----------
    watch_motifs  : dict[str, list[int]]  – named motifs to count each tick
    lz_sample     : int                   – molecules sampled for complexity
    """

    def __init__(self,
                 watch_motifs: dict[str, list[int]] | None = None,
                 lz_sample: int = 2000):
        self.watch_motifs = watch_motifs or {}
        self.lz_sample    = lz_sample
        self._prev_seqs_set: set[bytes] | None = None
        self._seen_events: set[str]            = set()
        self.history: list[Snapshot]           = []
        self._emergence_thresholds = {
            "entropy"         : 0.85,   # high diversity sustained
            "autocatalytic"   : 0.05,   # 5% autocatalytic is significant
            "mean_complexity" : 0.70,   # high structural complexity
        }

    # ── snapshot builder ─────────────────────────────────────────────────────

    def snapshot(self,
                 tick        : int,
                 pool_seqs   : np.ndarray,          # active sequences only
                 rule_fires  : dict[str, int],
                 stochastic  : dict[str, int]) -> Snapshot:

        n = len(pool_seqs)
        events: list[str] = []

        if n == 0:
            snap = Snapshot(tick=tick, population=0, entropy=0.0,
                            gc_content=0.0, mean_complexity=0.0,
                            autocatalytic=0.0, dominance=0.0, novelty_rate=0.0,
                            rule_fires=rule_fires, stochastic=stochastic)
            self.history.append(snap)
            return snap

        # ── entropy (Shannon over bit-position frequencies) ──────────────────
        p1   = pool_seqs.mean(axis=0)                          # (L,) – prob of 1 per position
        p0   = 1 - p1
        eps  = 1e-9
        H    = -np.sum(p1 * np.log2(p1 + eps) + p0 * np.log2(p0 + eps))
        H   /= pool_seqs.shape[1]                              # normalise per bit
        entropy = float(np.clip(H, 0, 1))

        # ── GC content ───────────────────────────────────────────────────────
        gc = float(pool_seqs.mean())

        # ── complexity (sampled LZ) ───────────────────────────────────────────
        complexity = batch_lz_complexity(pool_seqs, self.lz_sample)

        # ── autocatalysis proxy ──────────────────────────────────────────────
        # Fraction of molecules whose complement also exists in the pool.
        # Implemented via fast hash check on a sample.
        sample_n    = min(5000, n)
        sample_idx  = np.random.choice(n, sample_n, replace=False)
        sample_seqs = pool_seqs[sample_idx]
        comps       = 1 - sample_seqs                          # bitwise complement
        # hash both sets
        pool_set    = {bytes(r) for r in pool_seqs[np.random.choice(n, min(20_000, n), replace=False)]}
        auto_count  = sum(1 for r in comps if bytes(r) in pool_set)
        autocatalytic = auto_count / sample_n

        # ── dominance ────────────────────────────────────────────────────────
        # Frequency of the most common sequence (on a sample)
        sample_bytes = [bytes(r) for r in pool_seqs[np.random.choice(n, min(10_000, n), replace=False)]]
        freq = Counter(sample_bytes)
        top_count = freq.most_common(1)[0][1] if freq else 0
        dominance = top_count / min(10_000, n)

        # ── novelty rate ─────────────────────────────────────────────────────
        current_set = {bytes(r) for r in pool_seqs[np.random.choice(n, min(10_000, n), replace=False)]}
        if self._prev_seqs_set is not None:
            new_seqs = current_set - self._prev_seqs_set
            novelty  = len(new_seqs) / max(len(current_set), 1)
        else:
            novelty = 1.0
        self._prev_seqs_set = current_set

        # ── motif frequencies ─────────────────────────────────────────────────
        motif_counts = {}
        for name, motif in self.watch_motifs.items():
            m = np.array(motif, dtype=np.uint8)
            mlen = len(m)
            L    = pool_seqs.shape[1]
            count = 0
            for start in range(L - mlen + 1):
                chunk = pool_seqs[:, start:start + mlen]
                count += int(np.all(chunk == m, axis=1).sum())
            motif_counts[name] = count

        # ── emergence detection ───────────────────────────────────────────────
        metrics_map = {"entropy": entropy, "autocatalytic": autocatalytic,
                       "mean_complexity": complexity}
        for name, val in metrics_map.items():
            thresh = self._emergence_thresholds[name]
            key    = f"{name}>{thresh:.2f}"
            if val > thresh and key not in self._seen_events:
                events.append(f"🔬 EMERGENCE: {name} crossed {thresh:.2f} → {val:.3f}")
                self._seen_events.add(key)

        # Dominant sequence change
        top_seq = freq.most_common(1)[0][0] if freq else None
        if top_seq and dominance > 0.05:
            key = f"dominant:{top_seq.hex()}"
            if key not in self._seen_events:
                events.append(f"🧬 NEW DOMINANT sequence at {dominance*100:.1f}% prevalence")
                self._seen_events.add(key)

        snap = Snapshot(
            tick=tick, population=n, entropy=entropy, gc_content=gc,
            mean_complexity=complexity, autocatalytic=autocatalytic,
            dominance=dominance, novelty_rate=novelty,
            motif_counts=motif_counts, rule_fires=rule_fires,
            stochastic=stochastic, events=events
        )
        self.history.append(snap)
        return snap

    # ── history arrays for plotting ───────────────────────────────────────────

    def as_arrays(self):
        """Return dict of numpy arrays suitable for matplotlib."""
        if not self.history:
            return {}
        ticks = np.array([s.tick        for s in self.history])
        return {
            "tick"            : ticks,
            "population"      : np.array([s.population       for s in self.history]),
            "entropy"         : np.array([s.entropy           for s in self.history]),
            "gc_content"      : np.array([s.gc_content        for s in self.history]),
            "mean_complexity" : np.array([s.mean_complexity   for s in self.history]),
            "autocatalytic"   : np.array([s.autocatalytic     for s in self.history]),
            "dominance"       : np.array([s.dominance         for s in self.history]),
            "novelty_rate"    : np.array([s.novelty_rate      for s in self.history]),
        }
