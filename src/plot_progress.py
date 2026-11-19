"""Progress chart E0 -> final (Sprint 3 report). Run from the project root:

    .venv/bin/python src/plot_progress.py

Reads experiments/experiment_log.csv, writes reports/figures/progress_sprint3.png.
Experiments in run order; best-so-far step line over the KEEP experiments; E0/E1 are far below the zoomed
y-range and are annotated instead of plotted (one axis, no break).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, BLUE_L, GRAY, INK, INK2, GRID, BG = "#2a78d6", "#86b6ef", "#898781", "#1f1f1e", "#52514e", "#e1e0d9", "#fcfcfb"
YMIN, YMAX = 0.7455, 0.7660

d = pd.read_csv("experiments/experiment_log.csv")
d["x"] = range(len(d))
d["kept"] = d["verdict"].str.startswith(("KEEP", "BASELINE"))
best, run = [], 0.0
for _, r in d.iterrows():
    if r["kept"]:
        run = max(run, r["mean_auc"])
    best.append(run)
d["best"] = best

fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=BG)
ax.set_facecolor(BG)
s3 = d.loc[d["exp_id"] == "E14", "x"].iloc[0]
ax.axvspan(s3 - 0.5, d["x"].max() + 0.5, color="#f1f0eb", zorder=0)
ax.text(s3 - 0.3, YMIN + 0.0003, "Sprint 3", fontsize=9, color=INK2)
ax.text(s3 - 0.7, YMIN + 0.0003, "Sprint 1–2", fontsize=9, color=INK2, ha="right")

vis = d[d["mean_auc"] >= YMIN]
ax.step(vis["x"], vis["best"], where="post", color=BLUE, lw=2, zorder=2, label="best so far (KEEP)")
k, nk = vis[vis["kept"]], vis[~vis["kept"]]
ax.scatter(nk["x"], nk["mean_auc"], s=46, facecolor=BG, edgecolor=GRAY, lw=1.5, zorder=3,
           label="DISCARD / component")
ax.scatter(k["x"], k["mean_auc"], s=60, color=BLUE, edgecolor=BG, lw=2, zorder=4, label="KEEP")
labels = {"E2": "E2 all raw features", "E4": "E4 credit ratios", "E14": "E14 Optuna tuning",
          "E19": "E19 5 seeds", "E21": "E21 blend LGB+XGB"}
offsets = {"E19": (-6, -26, "right"), "E21": (-2, 10, "center")}   # avoid collisions in the Sprint-3 cluster
for _, r in k.iterrows():
    if r["exp_id"] in labels:
        dx, dy, ha = offsets.get(r["exp_id"], (-6, 9, "right"))
        ax.annotate(f"{labels[r['exp_id']]}\n{r['mean_auc']:.5f}", (r["x"], r["mean_auc"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK, ha=ha)
ax.text(-0.4, YMIN + 0.0004, "below range:\nE0 RF baseline 0.67020\nE1 LightGBM 0.67464", fontsize=8, color=INK2)

ax.set_xticks(d["x"]); ax.set_xticklabels(d["exp_id"], fontsize=8, color=INK2)
ax.set_ylim(YMIN, YMAX); ax.set_xlim(-0.6, d["x"].max() + 0.6)
ax.set_ylabel("mean 5-fold CV AUC", color=INK2)
ax.tick_params(axis="y", labelsize=8, colors=INK2)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]:
    ax.spines[sp].set_color(GRID)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("Experiment progress E0 → E21 (frozen 5-fold CV, experiments in run order)", loc="left",
             fontsize=11, color=INK)
fig.tight_layout()
fig.savefig("reports/figures/progress_sprint3.png", dpi=130)
print("wrote reports/figures/progress_sprint3.png")
