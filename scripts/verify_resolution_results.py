"""Recompute the recorded grid-transfer scores from every test-error curve."""
import csv
from pathlib import Path
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1] / "results/resolution_transfer"
    with (root / "runs.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        stem = f"run_{chr(65+int(row['seed']))}_N{row['grid']}_{row['policy']}"
        with np.load(root / "metrics" / (stem + ".npz"), allow_pickle=False) as z:
            error = z["nrmse"]
        assert error.shape == (30, 200)
        curve = error.mean(axis=0)
        scores = dict(gmean_1_100=np.exp(np.log(curve[:100]+1e-12).mean()),
                      gmean_101_200=np.exp(np.log(curve[100:]+1e-12).mean()),
                      nrmse_200=curve[-1], one_step_nrmse=curve[0])
        for key, value in scores.items():
            np.testing.assert_allclose(value, float(row[key]), rtol=1e-10)
        assert int((~np.isfinite(error)).any(axis=1).sum()) == int(row["failed_ics"])
    assert len(rows) == 25
    print("Verified all 25 groups (five runs, 30 initial states, 200 steps).")


if __name__ == "__main__":
    main()
