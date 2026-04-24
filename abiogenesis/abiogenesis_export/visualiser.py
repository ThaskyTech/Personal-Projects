"""
visualiser.py
-------------
Live matplotlib dashboard that updates every N ticks.

Two modes:
  interactive  – matplotlib in interactive mode, window updates while sim runs
  checkpoint   – saves PNG snapshots (good for headless / long runs)

Dashboard panels:
  1. Population over time
  2. Shannon entropy + GC content
  3. Mean complexity + autocatalytic fraction
  4. Novelty rate + dominance
  5. Rule fire counts (stacked area)
  6. Stochastic event counts (stacked area)
  7. Event log  (emergence events as vertical lines)
"""

from __future__ import annotations
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from metrics import MetricsEngine


# Use non-interactive backend when no display is available
try:
    matplotlib.use("TkAgg")
    import tkinter
    tkinter.Tk().destroy()
    _INTERACTIVE = True
except Exception:
    matplotlib.use("Agg")
    _INTERACTIVE = False


DARK_BG   = "#0f0f14"
PANEL_BG  = "#16161e"
LINE_COLS = ["#7ec8e3", "#f7c59f", "#a3d977", "#e07be0",
             "#f4e04d", "#ff6b6b", "#69d2e7", "#a8ff78"]
GRID_COL  = "#2a2a38"
TEXT_COL  = "#c8c8d8"


def _style():
    plt.rcParams.update({
        "figure.facecolor"   : DARK_BG,
        "axes.facecolor"     : PANEL_BG,
        "axes.edgecolor"     : GRID_COL,
        "axes.labelcolor"    : TEXT_COL,
        "xtick.color"        : TEXT_COL,
        "ytick.color"        : TEXT_COL,
        "text.color"         : TEXT_COL,
        "grid.color"         : GRID_COL,
        "grid.linestyle"     : "--",
        "grid.alpha"         : 0.5,
        "legend.facecolor"   : PANEL_BG,
        "legend.edgecolor"   : GRID_COL,
        "font.family"        : "monospace",
        "font.size"          : 9,
        "lines.linewidth"    : 1.5,
    })


class Dashboard:
    """
    Matplotlib dashboard for the abiogenesis simulation.

    Parameters
    ----------
    output_dir  : where to save PNG checkpoints (None = interactive only)
    save_every  : save a PNG every N ticks (0 = never)
    show        : whether to open an interactive window
    """

    def __init__(self,
                 output_dir : str | Path | None = "output",
                 save_every : int  = 50,
                 show       : bool = True):
        _style()
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_every = save_every
        self.show       = show and _INTERACTIVE

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fig = plt.figure(figsize=(16, 10), dpi=100)
        self.fig.patch.set_facecolor(DARK_BG)
        self._build_layout()

        if self.show:
            plt.ion()
            plt.show(block=False)

    def _build_layout(self):
        gs = gridspec.GridSpec(3, 3, figure=self.fig,
                               hspace=0.45, wspace=0.35,
                               left=0.07, right=0.97,
                               top=0.93, bottom=0.06)
        self.ax_pop   = self.fig.add_subplot(gs[0, 0])
        self.ax_ent   = self.fig.add_subplot(gs[0, 1])
        self.ax_comp  = self.fig.add_subplot(gs[0, 2])
        self.ax_nov   = self.fig.add_subplot(gs[1, 0])
        self.ax_rules = self.fig.add_subplot(gs[1, 1])
        self.ax_stoch = self.fig.add_subplot(gs[1, 2])
        self.ax_log   = self.fig.add_subplot(gs[2, :])

        for ax in self.fig.axes:
            ax.grid(True)
            ax.set_facecolor(PANEL_BG)

        self.fig.suptitle("🧬  Abiogenesis Simulator", fontsize=13,
                          color=TEXT_COL, y=0.98)

    def update(self, engine: MetricsEngine, tick: int):
        arrays = engine.as_arrays()
        if not arrays or len(arrays["tick"]) < 2:
            return

        t = arrays["tick"]

        self._plot_line(self.ax_pop,  t, arrays["population"],
                        "Population", "molecules", ["#7ec8e3"])
        self._plot_lines(self.ax_ent, t,
                         [arrays["entropy"], arrays["gc_content"]],
                         ["entropy", "GC content"],
                         "Diversity", "value (0–1)")
        self._plot_lines(self.ax_comp, t,
                         [arrays["mean_complexity"], arrays["autocatalytic"]],
                         ["LZ complexity", "autocatalytic"],
                         "Complexity & Self-replication", "value (0–1)")
        self._plot_lines(self.ax_nov, t,
                         [arrays["novelty_rate"], arrays["dominance"]],
                         ["novelty rate", "dominance"],
                         "Novelty & Dominance", "fraction")

        self._plot_rule_stacks(arrays, t)
        self._plot_event_log(engine, t)

        # draw emergence event lines on all panels
        for snap in engine.history:
            for ev in snap.events:
                for ax in [self.ax_pop, self.ax_ent, self.ax_comp, self.ax_nov]:
                    ax.axvline(snap.tick, color="#f4e04d", alpha=0.4,
                               linewidth=0.8, linestyle=":")

        self.fig.canvas.draw_idle()
        if self.show:
            plt.pause(0.001)

        if self.save_every and tick % self.save_every == 0 and self.output_dir:
            self._save(tick)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _plot_line(self, ax, t, y, title, ylabel, colors):
        ax.clear()
        ax.set_facecolor(PANEL_BG)
        ax.grid(True, color=GRID_COL, linestyle="--", alpha=0.5)
        ax.plot(t, y, color=colors[0])
        ax.fill_between(t, y, alpha=0.15, color=colors[0])
        ax.set_title(title, fontsize=9, color=TEXT_COL)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel("tick", fontsize=8)

    def _plot_lines(self, ax, t, ys, labels, title, ylabel):
        ax.clear()
        ax.set_facecolor(PANEL_BG)
        ax.grid(True, color=GRID_COL, linestyle="--", alpha=0.5)
        for i, (y, lbl) in enumerate(zip(ys, labels)):
            c = LINE_COLS[i % len(LINE_COLS)]
            ax.plot(t, y, label=lbl, color=c)
        ax.set_title(title, fontsize=9, color=TEXT_COL)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel("tick", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")

    def _plot_rule_stacks(self, arrays, t):
        ax = self.ax_rules
        ax.clear()
        ax.set_facecolor(PANEL_BG)
        ax.grid(True, color=GRID_COL, linestyle="--", alpha=0.5)

        # collect rule fire series from history
        from metrics import MetricsEngine
        history = None
        # fall back to raw arrays — rule_fires is in Snapshot objects
        # we grab it from the arrays dict if present
        # (stored separately below)
        if "_rule_labels" in arrays:
            labels = arrays["_rule_labels"]
            bottom = np.zeros(len(t))
            for i, lbl in enumerate(labels):
                vals = np.array(arrays.get(f"rule:{lbl}", [0]*len(t)))
                ax.fill_between(t, bottom, bottom + vals,
                                alpha=0.6, label=lbl,
                                color=LINE_COLS[i % len(LINE_COLS)])
                bottom += vals
        ax.set_title("Rule fires / tick", fontsize=9, color=TEXT_COL)
        ax.set_xlabel("tick", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")

        # stochastic side
        ax2 = self.ax_stoch
        ax2.clear()
        ax2.set_facecolor(PANEL_BG)
        ax2.grid(True, color=GRID_COL, linestyle="--", alpha=0.5)
        stoch_keys = ["mutations", "ligations", "cleavages",
                      "template_copies", "cosmic_rays"]
        bottom2 = np.zeros(len(t))
        for i, key in enumerate(stoch_keys):
            if f"stoch:{key}" in arrays:
                vals = np.array(arrays[f"stoch:{key}"])
                ax2.fill_between(t, bottom2, bottom2 + vals,
                                 alpha=0.6, label=key,
                                 color=LINE_COLS[(i+3) % len(LINE_COLS)])
                bottom2 += vals
        ax2.set_title("Stochastic events / tick", fontsize=9, color=TEXT_COL)
        ax2.set_xlabel("tick", fontsize=8)
        ax2.legend(fontsize=7, loc="upper left")

    def _plot_event_log(self, engine: MetricsEngine, t):
        ax = self.ax_log
        ax.clear()
        ax.set_facecolor(PANEL_BG)
        ax.axis("off")
        ax.set_title("Emergence event log", fontsize=9, color=TEXT_COL)

        lines = []
        for snap in engine.history[-30:]:
            for ev in snap.events:
                lines.append(f"  tick {snap.tick:>6}  │  {ev}")
        if not lines:
            lines = ["  (no emergence events yet)"]
        text = "\n".join(lines[-15:])
        ax.text(0.01, 0.95, text, transform=ax.transAxes,
                fontsize=8, verticalalignment="top",
                color=TEXT_COL, family="monospace")

    def _save(self, tick: int):
        path = self.output_dir / f"tick_{tick:06d}.png"
        self.fig.savefig(str(path), dpi=100, bbox_inches="tight",
                         facecolor=DARK_BG)

    def final_save(self, path: str | Path = "output/final.png"):
        self.fig.savefig(str(path), dpi=120, bbox_inches="tight",
                         facecolor=DARK_BG)
        print(f"[visualiser] saved → {path}")
