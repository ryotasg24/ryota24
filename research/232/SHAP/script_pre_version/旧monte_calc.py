#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 使い方
# $ python monte_calc.py Monte_k32_p100_dsSHAP_PointNeXt_h5


import os
import csv
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Compute average φ̂ and average 95% CI half-width from shap_phi_ci_rel.csv"
    )
    parser.add_argument(
        "base_dir",
        help="Result folder under /workspace/PointNeXt/result (e.g. Monte_k32_p100_dsSHAP_PointNeXt_h5)"
    )
    parser.add_argument(
        "--root",
        default="/workspace/PointNeXt/result",
        help="Root path to the result folders"
    )
    parser.add_argument(
        "--ds",
        default="50",
        help="Downsample folder name (default: 50)"
    )
    args = parser.parse_args()

    csv_path = os.path.join(args.root, args.base_dir, args.ds, "shap_phi_ci_rel.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} が見つかりません")

    phi_means = []
    ci_hw     = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phi_means.append(float(row["phi_mean"]))
            ci_hw.append(float(row["CI95_HW"]))

    if len(phi_means) == 0:
        print("データがありません")
        return

    avg_phi = sum(phi_means) / len(phi_means)
    avg_hw  = sum(ci_hw)     / len(ci_hw)
    avg_rel = avg_hw / abs(avg_phi) * 100

    print(f"全サンプル平均\t{avg_phi:.3f} \t±{avg_hw:.3f}")
    print(f"相対誤差幅\t±{avg_rel:.1f}%")

if __name__ == "__main__":
    main()
