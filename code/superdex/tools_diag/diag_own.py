"""f v4 / calib v4 / identity contact F1 on the resync pairs of each student's own IG eval rollouts (eval seeds 42-44)."""
import sys, glob, json, numpy as np, torch
sys.path.insert(0, "tools")
from train_f import FNet, f1_scores, THRESH
torch.set_num_threads(2)
FPATH, OUT = (sys.argv[1:3] + [None, None])[:2]
FPATH = FPATH or "runs/f_v4/f_b100_s0.pt"; OUT = OUT or "runs/diag/own_state_f1.json"
ck = torch.load(FPATH, map_location="cpu")
net = FNet(ck["din"], np.zeros(ck["din"], np.float32), np.ones(ck["din"], np.float32)); net.load_state_dict(ck["state_dict"]); net.eval()
cal = np.load("runs/calib/calib_v4.npz"); taus = cal["taus"]; lag = int(cal["lag"])
k = np.arange(32) != 11
# B+tactile eval records (777024) predate the qd recorder, so its control is the held-out pairBT seed 103 (not in f's training data).
SETS = {"f_student": [("evalFB_seed%d_783809" % s, ["runs/pairs_own/evalFB_seed%d_783809_0-64.npz" % s]) for s in (42, 43, 44)],
        "btactile": [("pairBT_seed103_777113", ["runs/pairs_v4/pairBT_seed103_777113_0-32.npz", "runs/pairs_v4/pairBT_seed103_777113_32-64.npz"])],
        "calib_student": [("evalTD_seed%d_783810" % s, ["runs/pairs_own/evalTD_seed%d_783810_0-64.npz" % s]) for s in (42, 43, 44)]}
res = {}
for name, recs in SETS.items():
    per_seed = {}; allP = {"identity": [], "calib": [], "f": []}; allY = []
    for rec, files in recs:
        s = rec; r = dict(np.load("isaac/runs/record_%s.npz" % rec)); ps = [dict(np.load(f)) for f in files]
        p = {kk: np.concatenate([q[kk] for q in ps]) for kk in ("episodes", "lengths", "resync_contact", "resync_pen_pos_err")}
        c = np.log1p(r["contact_norm"] / THRESH); a = np.clip(r["action"], -1, 1); cn = r["contact_norm"]
        X = []; I = []; C = []
        for e, L in zip(p["episodes"], p["lengths"]):
            for t in range(int(L)):
                h1 = c[e, t-1] if t >= 1 else 0*c[e, t]; h2 = c[e, t-2] if t >= 2 else 0*c[e, t]; qp = r["q"][e, t-1] if t >= 1 else r["q"][e, t]
                X.append(np.concatenate([c[e, t], h1, h2, r["q"][e, t], qp, r["target"][e, t], a[e, t]]))
                I.append(cn[e, t] >= THRESH); C.append((cn[e, t-lag] if t >= lag else 0*cn[e, t]) >= taus)
        Y = p["resync_contact"] >= THRESH; assert len(Y) == len(X)
        with torch.no_grad(): P = (net(torch.tensor(np.array(X), dtype=torch.float32)) > 0).numpy()
        preds = {"identity": np.array(I), "calib": np.array(C), "f": P}
        per_seed[s] = {n: round(f1_scores(v, Y)["micro_f1_ex_pen"], 4) for n, v in preds.items()}
        per_seed[s]["rows"] = len(Y); per_seed[s]["pen_err_med_cm"] = round(float(np.median(p["resync_pen_pos_err"]) * 100), 3)
        for n in allP: allP[n].append(preds[n])
        allY.append(Y)
    Y = np.concatenate(allY); out = {"rows": int(len(Y)), "on_sdx": round(float(Y[:, k].mean()), 4), "per_seed": per_seed}
    for n in allP:
        P = np.concatenate(allP[n]); sc = f1_scores(P, Y)
        out[n] = {"F1": round(sc["micro_f1_ex_pen"], 4), "prec": round(sc["precision_ex_pen"], 4), "rec": round(sc["recall_ex_pen"], 4),
                  "on": round(float(P[:, k].mean()), 4), "F1_seed_sd": round(float(np.std([per_seed[s][n] for s in per_seed], ddof=1)), 4) if len(per_seed) > 1 else None}
    res[name] = out; print(name, json.dumps({n: out[n] for n in ("rows", "on_sdx", "identity", "calib", "f")}), flush=True)
Ff = res["f_student"]["f"]["F1"]; Fo = min(res["btactile"]["f"]["F1"], res["calib_student"]["f"]["F1"])
res["gap_Fo_minus_Ff"] = round(Fo - Ff, 4); print("gap", res["gap_Fo_minus_Ff"])
res["f_path"] = FPATH
json.dump(res, open(OUT, "w"), indent=1)
