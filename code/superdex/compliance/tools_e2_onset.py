"""E2b: contact onset timing error on held-out resync pairs (dataset v5), identity / calib v4 / f v4 / f v5.
Onset = 0->1 transition of a channel's binary contact within an episode (t >= 1; the pen-total channel 11 excluded).
Each true (SDX) onset is matched to the nearest predicted onset of the same channel within +-W steps (W=5, 250 ms,
fixed before looking at results). Reports match rate, mean |dt| and mean signed dt (pred - true, >0 = late) over
matched onsets in ms, and extra predicted onsets per true onset.
usage: e2_onset.py <dataset.npz> <out.json>"""
import json
import sys
import numpy as np
import torch
sys.path.insert(0, "tools")
from train_f import FNet, features, THRESH, KEEP  # noqa: E402
from calib_thresh_delay import lagged  # noqa: E402

W, DT_MS = 5, 50.0
F_MODELS = {"f_v4": "<PROJECT_ROOT>/penspin-superdex-penlen-20261031/runs/f_v4/f_b100_s0.pt",
            "f_v5": "<PROJECT_ROOT>/penspin-superdex-penlen-20261031/runs/f_v5/f_b100_s0.pt"}
CALIB = "<PROJECT_ROOT>/penspin-superdex-penlen-20261031/runs/calib/calib_v4.npz"


def onsets(b):
    return np.nonzero(b[1:] & ~b[:-1])[0] + 1


def main(ds, out):
    torch.set_num_threads(4)
    d = dict(np.load(ds)); y = d["resync_contact"] >= THRESH
    cal = np.load(CALIB); C = lagged(d, int(cal["lag"])) >= cal["taus"]
    preds = {"identity": d["ig_contact"] >= THRESH, "calib_v4": C}
    for name, path in F_MODELS.items():
        ck = torch.load(path, map_location="cpu")
        net = FNet(ck["din"], np.zeros(ck["din"], np.float32), np.ones(ck["din"], np.float32)); net.load_state_dict(ck["state_dict"]); net.eval()
        P = np.zeros_like(y)
        for idx in np.array_split(np.arange(len(y)), 20):
            with torch.no_grad():
                P[idx] = (net(torch.as_tensor(features(d, idx), dtype=torch.float32)) > 0).numpy()
        preds[name] = P
    res = {"window_steps": W, "dt_ms": DT_MS}
    for split in ("val", "shift", "shift_p", "shift_f"):
        rows = np.nonzero(d["split"] == split)[0]
        eps = {}
        for i in rows:
            eps.setdefault(int(d["episode"][i]), []).append(i)
        r = {}
        for name, P in preds.items():
            dts, n_true, n_pred, n_match = [], 0, 0, 0
            for e, ii in eps.items():
                ii = np.array(ii)[np.argsort(d["t"][ii])]
                assert np.all(np.diff(d["t"][ii]) == 1), (split, e)
                for c in KEEP:
                    to, po = onsets(y[ii, c]), onsets(P[ii, c])
                    n_true += len(to); n_pred += len(po); used = set()
                    for t in to:
                        if len(po) == 0:
                            continue
                        j = int(np.argmin(np.abs(po - t)))
                        if abs(po[j] - t) <= W and j not in used:
                            used.add(j); dts.append(po[j] - t); n_match += 1
            dts = np.array(dts, float)
            r[name] = {"true_onsets": n_true, "pred_onsets": n_pred, "match_rate": round(n_match / max(1, n_true), 4),
                       "abs_dt_ms": round(float(np.abs(dts).mean() * DT_MS), 1) if len(dts) else None,
                       "signed_dt_ms": round(float(dts.mean() * DT_MS), 1) if len(dts) else None,
                       "extra_pred_per_true": round((n_pred - n_match) / max(1, n_true), 4)}
        res[split] = r
        print(split, json.dumps(r), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
