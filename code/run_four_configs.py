
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
import json

CONFIGS = [
    ("Full", []),  # 全开
    ("NoViewAttn", ["--disable_view_attention"]),  # 关视图注意力
    ("NoNodeAttn", ["--disable_node_attention"]),  # 关节点注意力
    ("AllOff", ["--disable_view_attention", "--disable_node_attention"]),  # 全关
]

PATTERN = re.compile(r"^(ACCURACY|PRECISION|RECALL|F1|AUC):\s*([0-9.]+)\s*$", re.MULTILINE)

def run_one(python_exec, script, data_dir, base_args, extra_flags):
    cmd = [python_exec, script, "--data_dir", data_dir] + base_args + extra_flags
    print("\n==> Running:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    # Parse metrics
    block_start = proc.stdout.rfind("=== Best Test Results ===")
    metrics = {}
    if block_start != -1:
        m = PATTERN.findall(proc.stdout[block_start:])
        metrics = {k.lower(): float(v) for (k, v) in m}
    else:
        print("WARNING: Could not find 'Best Test Results' block.")
    return metrics, proc.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable, help="Python executable to use")
    ap.add_argument("--script", default="CryptoSSGNN0808_fast.py", help="Training script filename")
    ap.add_argument("--data_dir", default="dataset", help="Dataset directory")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()

    base_args = ["--epochs", str(args.epochs),
                 "--eval_every", str(args.eval_every),
                 "--alpha", str(args.alpha),
                 "--seed", str(args.seed)]
    if args.cuda:
        base_args.append("--cuda")

    results = []
    for name, flags in CONFIGS:
        metrics, rc = run_one(args.python, args.script, args.data_dir, base_args, flags)
        row = {"config": name, **metrics, "returncode": rc}
        results.append(row)

    # Save JSON & CSV
    out_dir = Path("ablation_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "four_config_results.json"
    csv_path = out_dir / "four_config_results.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Write CSV
    cols = ["config", "accuracy", "precision", "recall", "f1", "auc", "returncode"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    # Pretty print table
    try:
        import pandas as pd
        df = pd.DataFrame(results).set_index("config")
        # Sort by F1 desc if present
        if "f1" in df.columns:
            df = df.sort_values("f1", ascending=False)
        print("\n==== Summary (sorted by F1) ====\n")
        print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    except Exception as e:
        print("Summary table requires pandas. Install with: pip install pandas")
        print("Error:", e)

    print(f"\nSaved results to:\n- {json_path}\n- {csv_path}")

if __name__ == "__main__":
    main()
