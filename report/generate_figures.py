"""One-off script generating the report's figures from real, currently-cached project data.

Not part of the shipped package — run manually when the report needs regenerating.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from app.logic import compute_fold_timeline, cumulative_equity
from ml_backtester.config import Config
from ml_backtester.experiments.run_comparison import build_dataset, run_walk_forward
from ml_backtester.experiments.run_deflation_study import compute_deflated_sharpe, run_config_sweep
from ml_backtester.validation.splitters import PurgedEmbargoedSplitter, WalkForwardSplitter

FIG_DIR = "report/figures"
plt.rcParams.update({"font.size": 9, "figure.dpi": 200})

config = Config()
dataset = build_dataset(config)

naive_splitter = WalkForwardSplitter(config.train_window_days, config.test_window_days, config.embargo_days)
corrected_splitter = PurgedEmbargoedSplitter(
    config.train_window_days, config.test_window_days, config.embargo_days
)
naive = run_walk_forward(dataset, naive_splitter, config)
corrected = run_walk_forward(dataset, corrected_splitter, config)

print("naive metrics:", naive.metrics)
print("corrected metrics:", corrected.metrics)
print("naive folds:", len(list(naive_splitter.split(dataset, dataset["t1"]))))
print("corrected folds:", len(list(corrected_splitter.split(dataset, dataset["t1"]))))

# --- Figure 1: cumulative equity curves ---
fig, ax = plt.subplots(figsize=(3.4, 2.4))
cumulative_equity(naive.returns).plot(ax=ax, label="Naive (no purge/embargo)", color="#E45756")
cumulative_equity(corrected.returns).plot(ax=ax, label="Corrected (purged/embargoed)", color="#4C78A8")
ax.set_ylabel("Cumulative equity (start = 1.0)")
ax.set_xlabel("Date")
ax.legend(fontsize=7, loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/equity_curves.png")
plt.close(fig)

# --- Figure 2: fold timeline (first 6 folds per variant) ---
timeline = pd.concat(
    [
        compute_fold_timeline(dataset, naive_splitter, "naive"),
        compute_fold_timeline(dataset, corrected_splitter, "corrected"),
    ],
    ignore_index=True,
)
colors = {"train": "#4C78A8", "purged": "#E45756", "test": "#54A24B", "embargo": "#B279A2"}

fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.0), sharex=True)
for ax, variant in zip(axes, ["naive", "corrected"]):
    sub = timeline[(timeline["variant"] == variant) & (timeline["fold"] < 6)]
    for _, row in sub.iterrows():
        width = (row["end"] - row["start"]).days + 1
        ax.broken_barh(
            [(mdates.date2num(row["start"]), width)],
            (row["fold"] - 0.4, 0.8),
            facecolors=colors[row["segment"]],
        )
    ax.set_ylabel(f"{variant}\nfold")
    ax.set_yticks(range(6))
    ax.invert_yaxis()
    ax.xaxis_date()

axes[-1].set_xlabel("Date")
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate(rotation=45)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
fig.legend(handles, colors.keys(), loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fold_timeline.png", bbox_inches="tight")
plt.close(fig)

# --- Figure 3: hyperparameter sweep bar chart ---
trials = run_config_sweep(config)
best, deflated = compute_deflated_sharpe(trials)
print("best:", best.hyperparameters, best.annualized_sharpe)
print("deflated sharpe:", deflated)

labels = [str(t.hyperparameters) for t in trials]
sharpes = [t.annualized_sharpe for t in trials]
bar_colors = ["#F2A900" if t is best else "#4C78A8" for t in trials]

fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.barh(range(len(trials)), sharpes, color=bar_colors)
ax.set_yticks(range(len(trials)))
ax.set_yticklabels([lbl.replace("'", "").replace("{", "").replace("}", "") for lbl in labels], fontsize=6)
ax.set_xlabel("Annualized Sharpe")
ax.axvline(0, color="black", linewidth=0.6)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/sweep_bars.png")
plt.close(fig)

print("figures written to", FIG_DIR)