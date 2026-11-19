"""Sprint-3 experiment runner (docs/sprints/sprint-3.md). Run from the project root, e.g.

    .venv/bin/python src/run_experiment3.py --exp E14 --parent E10 --params-json outputs/optuna/lgb_best_params.json \
        --hypothesis "..." --confirm-always
    .venv/bin/python src/run_experiment3.py --exp E15 --parent E14 --lr 0.02 --hypothesis "..."
    .venv/bin/python src/run_experiment3.py --exp E17 --parent E16 --flags ext_source_aggs --hypothesis "..."
    .venv/bin/python src/run_experiment3.py --exp E19 --parent E18 --seeds 42,43,44,45,46 --hypothesis "..."
    .venv/bin/python src/run_experiment3.py --exp E20 --parent E19 --model xgb --xgb-params-json ... --component

An experiment config = {model, lgb_params, xgb_params, flags, seeds}. Exactly ONE item may differ from the
parent: a feature flag, the LightGBM parameter set (a full set from Optuna, or only the learning rate), the
seed list, or the model family. Decision rule = ORIGINAL validation-setup §3 incl. CONFIRM (restored in
Sprint 3): CONFIRM re-runs parent and child on fold seeds 1, 2, 3 (outputs/folds_seed{s}.npy) and keeps the
child only if its mean over all 4 fold sets (20 folds) is higher AND it wins on >= 3 of the 4 fold sets.
Parent fold AUCs on the main folds are recomputed from its saved OOF at full precision.
"""
import argparse
import contextlib
import csv
import datetime as dt
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import SEED, compare, make_folds, run_cv_lgb, run_cv_xgb  # noqa: E402
from features import FEATURE_FLAGS_DEFAULT, make_features  # noqa: E402
from utils import log_experiment, save_artifacts, seed_everything  # noqa: E402

LOG = Path("experiments/experiment_log.csv")
ALT_FOLD_SEEDS = (1, 2, 3)
_FEATURE_CACHE = {}


def load_config(exp):
    p = json.load(open(f"outputs/params/{exp}.json"))["params"]
    return {
        "model": p.get("model", "rf"),
        "lgb_params": p.get("lgb_params"),
        "xgb_params": p.get("xgb_params"),
        "flags": {**FEATURE_FLAGS_DEFAULT, **p.get("flags", {})},
        "seeds": p.get("seeds", [SEED]),
    }


def features_for(flags, train, test):
    key = json.dumps(flags, sort_keys=True)
    if key not in _FEATURE_CACHE:
        _FEATURE_CACHE[key] = make_features(train, test, flags)
    return _FEATURE_CACHE[key]


def run_config(cfg, train, test, y, folds, importance_out=None, quiet=False):
    """Run one config on one fold set. Returns oof, test_pred, fold_aucs, best_iters."""
    X, Xt, names, cats = features_for(cfg["flags"], train, test)
    ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with ctx:
        if cfg["model"] in ("lgb", "lgb_seedavg"):
            oofs, tests, iters = [], [], []
            for s in cfg["seeds"]:
                o, t, _, it = run_cv_lgb(X, y, Xt, cfg["lgb_params"], folds, seed=s, cat_features=cats or "auto",
                                         importance_out=importance_out)
                oofs.append(o); tests.append(t); iters.append(it)
            oof, test_pred = np.mean(oofs, axis=0), np.mean(tests, axis=0)   # fixed formula: plain average
            best_iters = iters[0] if len(iters) == 1 else iters
        elif cfg["model"] == "xgb":
            oof, test_pred, _, best_iters = run_cv_xgb(X, y, Xt, cfg["xgb_params"], folds, seed=cfg["seeds"][0],
                                                       importance_out=importance_out)
        else:
            raise ValueError(cfg["model"])
    fold_aucs = [roc_auc_score(y[folds == k], oof[folds == k]) for k in range(folds.max() + 1)]
    return oof, test_pred, fold_aucs, best_iters, names


def alt_folds(y, s):
    path = Path(f"outputs/folds_seed{s}.npy")
    if path.exists():
        f = np.load(path)
        assert (f == make_folds(y, seed=s, path=None)).all()
        return f
    return make_folds(y, seed=s, path=str(path))      # never touches outputs/folds.npy


def confirm(parent_cfg, child_cfg, train, test, y, main_parent, main_child):
    """Alternative-fold check: returns dict with per-fold-set means, 20-fold means, wins."""
    res = {"main": (float(np.mean(main_parent)), float(np.mean(main_child)))}
    for s in ALT_FOLD_SEEDS:
        f = alt_folds(y.values, s)
        _, _, pa, _, _ = run_config(parent_cfg, train, test, y, f, quiet=True)
        _, _, ch, _, _ = run_config(child_cfg, train, test, y, f, quiet=True)
        res[f"seed{s}"] = (float(np.mean(pa)), float(np.mean(ch)))
        print(f"  fold seed {s}: parent {np.mean(pa):.5f} | child {np.mean(ch):.5f} | Δ {np.mean(ch)-np.mean(pa):+.5f}")
    p_all = np.mean([v[0] for v in res.values()]); c_all = np.mean([v[1] for v in res.values()])
    wins = sum(v[1] > v[0] for v in res.values())
    alt_gain = np.mean([res[f"seed{s}"][1] - res[f"seed{s}"][0] for s in ALT_FOLD_SEEDS])
    return {"sets": res, "parent_20": p_all, "child_20": c_all, "wins_sets": wins,
            "main_gain": res["main"][1] - res["main"][0], "alt_gain": float(alt_gain),
            "passed": bool(c_all > p_all and wins >= 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--hypothesis", required=True)
    ap.add_argument("--change", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--flags", default=None, help="one feature flag to switch on")
    ap.add_argument("--params-json", default=None, help="full LightGBM param set (Optuna best)")
    ap.add_argument("--lr", type=float, default=None, help="change only the LightGBM learning rate")
    ap.add_argument("--seeds", default=None, help="comma-separated LightGBM seeds (seed averaging)")
    ap.add_argument("--model", default=None, choices=["xgb"], help="model family change")
    ap.add_argument("--xgb-params-json", default=None)
    ap.add_argument("--component", action="store_true", help="blend component: verdict COMPONENT unless it beats parent")
    ap.add_argument("--confirm-always", action="store_true", help="always run the alternative-fold check (T1)")
    a = ap.parse_args()
    seed_everything(SEED)
    t0 = time.time()
    with open(LOG, newline="") as f:
        if any(r["exp_id"] == a.exp for r in csv.DictReader(f)):
            sys.exit(f"{a.exp} already in the log - experiments are never overwritten")

    parent = load_config(a.parent)
    child = json.loads(json.dumps(parent))
    if a.flags:
        assert not child["flags"][a.flags], f"{a.flags} already on"
        child["flags"][a.flags] = True
    if a.params_json:
        child["lgb_params"] = json.load(open(a.params_json))["params"]
    if a.lr is not None:
        child["lgb_params"] = {**child["lgb_params"], "learning_rate": a.lr}
    if a.seeds:
        child["seeds"] = [int(s) for s in a.seeds.split(",")]
        child["model"] = "lgb_seedavg" if len(child["seeds"]) > 1 else "lgb"
    if a.model == "xgb":
        child["model"] = "xgb"
        child["xgb_params"] = json.load(open(a.xgb_params_json))["params"]
        child["seeds"] = [SEED]
    n_changes = (sum(child["flags"][k] != parent["flags"][k] for k in child["flags"])
                 + int(child["lgb_params"] != parent["lgb_params"] and child["model"] != "xgb")
                 + int(child["seeds"] != parent["seeds"] and child["model"] != "xgb")
                 + int((child["model"] == "xgb") != (parent["model"] == "xgb")))
    if n_changes != 1:
        sys.exit(f"REFUSED: {n_changes} things differ from {a.parent} (exactly 1 allowed)")

    train = pd.read_csv("input/train.csv"); test = pd.read_csv("input/test.csv")
    y = train["TARGET"]
    folds = np.load("outputs/folds.npy")
    p_oof = np.load(f"outputs/oof/{a.parent}_oof.npy")
    p_folds = [roc_auc_score(y[folds == k], p_oof[folds == k]) for k in range(5)]

    imps = []
    oof, test_pred, fold_aucs, best_iters, names = run_config(child, train, test, y, folds, importance_out=imps)
    t_main = time.time() - t0
    print(f"Fold AUCs: {np.round(fold_aucs, 5)} | mean {np.mean(fold_aucs):.5f} | OOF {roc_auc_score(y, oof):.5f}")

    verdict = compare(fold_aucs, p_folds)
    d_mean = float(np.mean(fold_aucs) - np.mean(p_folds))
    wins = int((np.asarray(fold_aucs) > np.asarray(p_folds)).sum())
    conf, notes = None, [a.notes] if a.notes else []
    if verdict.startswith("CONFIRM") or a.confirm_always:
        print("Alternative-fold check (fold seeds 1, 2, 3):")
        conf = confirm(parent, child, train, test, y, p_folds, fold_aucs)
        notes.append("alt folds: " + "; ".join(f"{k} {v[0]:.5f}->{v[1]:.5f}" for k, v in conf["sets"].items())
                     + f"; 20-fold {conf['parent_20']:.5f}->{conf['child_20']:.5f}; set wins {conf['wins_sets']}/4")
        if verdict.startswith("CONFIRM"):
            verdict = "KEEP (CONFIRMED)" if conf["passed"] else "DISCARD (CONFIRM failed)"
    if a.component and not verdict.startswith("KEEP"):
        verdict = f"COMPONENT (rule: {verdict})"

    guards = []
    if max(fold_aucs) > 0.80:
        guards.append("fold AUC > 0.80")
    if conf and conf["main_gain"] > 0 and conf["alt_gain"] < 0.3 * conf["main_gain"]:
        guards.append(f"tuning gain does not reproduce: main {conf['main_gain']:+.5f} vs alt {conf['alt_gain']:+.5f}")
    flat_iters = np.ravel(best_iters)
    if flat_iters.max() >= 10000 or flat_iters.min() < 50:
        guards.append(f"best_iteration out of range {flat_iters.min()}..{flat_iters.max()}")

    if imps:
        imp = pd.concat(imps, axis=1).fillna(0)
        Path("outputs/importance").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"gain_mean": imp.mean(axis=1), "gain_std": imp.std(axis=1)}).sort_values(
            "gain_mean", ascending=False).to_csv(f"outputs/importance/{a.exp}.csv")
    save_artifacts(a.exp, oof, test_pred, child, names, list(flat_iters), out_dir="outputs")
    elapsed = time.time() - t0
    change = a.change or (a.flags or (f"lr {parent['lgb_params']['learning_rate']} -> {a.lr}" if a.lr is not None else
                          ("params -> Optuna best" if a.params_json else
                           (f"seeds {child['seeds']}" if a.seeds else f"model {parent['model']} -> {child['model']}"))))
    log_experiment({
        "exp_id": a.exp, "date": dt.date.today().isoformat(), "parent_id": a.parent,
        "hypothesis": a.hypothesis, "change": change, "model": child["model"], "n_features": len(names),
        **{f"fold_{k+1}": round(v, 5) for k, v in enumerate(fold_aucs)},
        "mean_auc": round(float(np.mean(fold_aucs)), 5), "std_auc": round(float(np.std(fold_aucs)), 5),
        "oof_auc": round(roc_auc_score(y, oof), 5), "delta_vs_parent": f"{d_mean:+.5f}",
        "folds_improved": f"{wins}/5", "verdict": verdict, "public_lb": "",
        "notes": "; ".join(notes + [f"main run {t_main:.0f}s, total {elapsed:.0f}s",
                                    f"best_iters {flat_iters.min()}-{flat_iters.max()}"] + guards),
    }, log_path=LOG)
    if conf:
        json.dump(conf, open(f"outputs/params/{a.exp}_confirm.json", "w"), indent=2)
    print(f"SUMMARY {a.exp} | parent {a.parent} | Δmean {d_mean:+.5f} | {wins}/5 | {verdict} | "
          f"mean {np.mean(fold_aucs):.5f} | folds {np.round(fold_aucs, 5).tolist()} | iters {flat_iters.min()}-{flat_iters.max()} "
          f"| main {t_main:.0f}s total {elapsed:.0f}s")
    if guards:
        print("SANITY GUARD TRIGGERED:", guards)
        sys.exit(2)


if __name__ == "__main__":
    main()
