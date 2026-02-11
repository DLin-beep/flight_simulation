



from __future__ import annotations



import argparse

import json

from pathlib import Path



import pandas as pd



from src.srcr import ShiftRobustConformalDelayModel





def main() -> None:

    p = argparse.ArgumentParser(description="Evaluate shift-robust calibration on a delay CSV.")

    p.add_argument("--csv", required=True, help="Path to delay CSV (must include origin/dest/delay and date).")

    p.add_argument("--alpha", type=float, default=0.1)

    p.add_argument("--min_group", type=int, default=30)

    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--out", default="srcr_shift_metrics.json")

    args = p.parse_args()



    df = pd.read_csv(args.csv)

    model = ShiftRobustConformalDelayModel(alpha=args.alpha, min_group=args.min_group, seed=args.seed).fit(df)

    metrics = model.evaluate_shift(df)



    Path(args.out).write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))





if __name__ == "__main__":

    main()

