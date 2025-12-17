#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
script_Analysis/comfirm_scoreScale.py

目的:
  SD-CS のスコア
    score_i(t, N) = Hybrid[i] + λ_cov(N, t, gap_cov(t, N)) · d_i(t)
  において、
    - Hybrid のスケール
    - λ_cov と λ_cov · d_norm の大きさ
  を 1 サンプルでデバッグ・可視化するためのスクリプト。

前提:
  - SD-CS / AMA 等で作成した MID キャッシュ (.npz) を読み込む。
    この .npz には以下の配列が含まれている想定:
      - "region_per_pt": (num_points,) Region-SHAP を per-point 化したもの
      - "point_adj"    : (num_points,) L1 調整後の Point-SHAP
      - "pc"           : (num_points,3) 元の点群座標
      - "label"        : (1,) クラスラベル (任意)
  - Hybrid は
      Hybrid = alpha * region_per_pt + (1 - alpha) * point_adj
    として構成する。
  - alpha は
      - 固定値 (--alpha_fixed) を指定する
      - または heuristic Gate で推定する
    の 2 通りをサポート。
  - MLP Gate も使いたい場合は、必要に応じて本スクリプトに実装を追加すること。

使い方(例):
  python script_Analysis/comfirm_scoreScale.py --mid_npz /workspace/PointNeXt/result/_shap_cache_mid/p10000_kr32_kp64_ckpt-modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT_ckpt_best_scheduling/ply_data_test0_00016_r32_p64.npz  --ds_points 900 --max_steps_to_pr
int 30

  ※ alpha_fixed を省略した場合は heuristic Gate を用いる。
"""


import argparse
import os
import sys
import math
import numpy as np


# =========================================================
# coverage_target(N) テーブル (SD-CS と同じもの)
# =========================================================

COVERAGE_TARGET_TABLE = {
    100: 0.927338,
    200: 0.949036,
    300: 0.958705,
    400: 0.964581,
    500: 0.913272,
    600: 0.919857,
    700: 0.925192,
    800: 0.930477,
    900: 0.935687,
    1000: 0.940567,
}


def get_coverage_target(N: int) -> float:
    """
    coverage_target(N) を返す。
    N がテーブルにない場合は、近傍のキーで線形補間／最近傍近似する。
    """
    if N in COVERAGE_TARGET_TABLE:
        return float(COVERAGE_TARGET_TABLE[N])

    ks = sorted(COVERAGE_TARGET_TABLE.keys())
    if N <= ks[0]:
        return float(COVERAGE_TARGET_TABLE[ks[0]])
    if N >= ks[-1]:
        return float(COVERAGE_TARGET_TABLE[ks[-1]])

    for i in range(len(ks) - 1):
        k0, k1 = ks[i], ks[i + 1]
        if k0 <= N <= k1:
            v0 = COVERAGE_TARGET_TABLE[k0]
            v1 = COVERAGE_TARGET_TABLE[k1]
            ratio = (float(N) - k0) / float(k1 - k0)
            return float(v0 + (v1 - v0) * ratio)

    # フォールバック
    return float(COVERAGE_TARGET_TABLE[ks[-1]])


# =========================================================
# GlobalCoverageState (SD-CS と同じ挙動)
# =========================================================

class GlobalCoverageState:
    """
    全点群 P に対する directed Hausdorff 半径ベースの被覆率を
    インクリメンタルに管理するクラス。

    points: (N,3) 元点群
    """

    def __init__(self, points: np.ndarray, eps: float = 1e-12):
        self.points = points.astype(np.float32, copy=False)
        self.N = self.points.shape[0]
        self.eps = float(eps)

        if self.N > 0:
            bbox_min = np.min(self.points, axis=0)
            bbox_max = np.max(self.points, axis=0)
            self.diag = float(np.linalg.norm(bbox_max - bbox_min) + self.eps)
        else:
            self.diag = self.eps

        # 各点の「S からの最近傍距離」
        self.dist_to_S = np.full(self.N, np.inf, dtype=np.float32)
        self.has_selected = False

    def update_with_point(self, idx: int):
        """
        新たに選択した点 idx を S に追加し、dist_to_S を更新する。
        """
        if self.N == 0:
            return
        idx = int(idx)
        y = self.points[idx]
        diff = self.points - y
        d_new = np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)

        if not self.has_selected:
            self.dist_to_S = d_new
        else:
            self.dist_to_S = np.minimum(self.dist_to_S, d_new)

        self.has_selected = True

    def get_coverage(self) -> float:
        """
        coverage = 1 - (directed HD 半径 / bounding box 対角長)
        """
        if not self.has_selected or self.diag <= 0.0 or self.N == 0:
            return 0.0

        r_cov = float(np.max(self.dist_to_S))
        r_norm = r_cov / self.diag
        score = 1.0 - r_norm

        if not np.isfinite(score):
            return 0.0

        return max(0.0, min(1.0, score))

    def get_normalized_distances(self) -> np.ndarray:
        """
        各点の「S からの最近傍距離 / diag」を返す。
        S が空の間は 0 ベクトル。
        """
        if not self.has_selected or self.diag <= 0.0 or self.N == 0:
            return np.zeros(self.N, dtype=np.float32)

        d_norm = self.dist_to_S / self.diag
        d_norm = np.clip(d_norm, 0.0, 1.0).astype(np.float32)
        return d_norm


# =========================================================
# λ_cov の計算 (SD-CS と同じロジック)
# =========================================================

def compute_lambda_cov(N: int,
                       step_idx: int,
                       coverage: float,
                       coverage_target: float,
                       lambda_scale: float = 1.0) -> float:
    """
    λ_cov(N, t, gap) を計算する。
      t = step_idx + 1
      gap = max(0, coverage_target - coverage)

    ・Λ_max(N):
        N<=200      : 1.0
        200 < N<=400: 0.8
        400 < N<=600: 0.5
        600 < N<=800: 0.3
        800 < N     : 0.2

    ・f_t(t/N) = max(0, 1 - t/N)

    ・g_gap(gap):
        gap >= 0.05 で 1.0 に飽和、それ未満では gap/0.05 で線形スケール。

    lambda_scale で全体スケールを調整する。
    """
    if N <= 0:
        return 0.0

    # Λ_max(N)
    if N <= 200:
        Lmax = 1.0
    elif N <= 400:
        Lmax = 0.8
    elif N <= 600:
        Lmax = 0.5
    elif N <= 800:
        Lmax = 0.3
    else:
        Lmax = 0.2

    Lmax *= float(lambda_scale)

    t = step_idx + 1
    s = float(t) / float(N)
    f_t = max(0.0, 1.0 - s)

    gap = max(0.0, coverage_target - coverage)
    gap_scale = 0.05
    if gap >= gap_scale:
        g_gap = 1.0
    else:
        g_gap = gap / gap_scale if gap_scale > 0 else 0.0

    lam = float(Lmax * f_t * g_gap)
    return lam


# =========================================================
# heuristic Gate による alpha 推定
# =========================================================

def heuristic_alpha(N: int,
                    var_reg: float,
                    var_pnt: float,
                    Nmid: int = 300,
                    k: float = 4.0) -> float:
    """
    SD-CS / AMA の heuristic Gate と同等のロジックで alpha を決める。

    logit = termN + 0.5 * log(var_reg / var_pnt)
    alpha = sigmoid(logit)

    termN = -k * ((N - Nmid) / Nmid)
    """
    Nmid = max(1, int(Nmid))
    k = float(k)

    termN = -k * ((float(N) - float(Nmid)) / float(Nmid))
    logr = math.log((var_reg + 1e-9) / (var_pnt + 1e-9))
    logit = termN + 0.5 * logr
    alpha = 1.0 / (1.0 + math.exp(-logit))
    return float(alpha)


# =========================================================
# メイン処理: 1 サンプルについて SD-CS のスコア挙動をデバッグ
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug SD-CS score balance between Hybrid and FPS distance term for one sample."
    )
    parser.add_argument("--mid_npz", type=str, required=True,
                        help="MID キャッシュ .npz のパス (region_per_pt, point_adj, pc を含む)")
    parser.add_argument("--ds_points", type=int, required=True,
                        help="SD-CS で取りたい点数 N (例: 100, 200, ...)")
    parser.add_argument("--lambda_scale", type=float, default=2.0,
                        help="Coverage 距離成分 λ_cov の全体スケール (SD-CS と同じ意味)")
    parser.add_argument("--alpha_fixed", type=float, default=None,
                        help="固定 alpha を使う場合に指定 (例: 0.5)。"
                             "指定がなければ heuristic Gate で決定。")
    parser.add_argument("--ama_Nmid", type=int, default=300,
                        help="heuristic Gate 用の Nmid")
    parser.add_argument("--ama_k", type=float, default=4.0,
                        help="heuristic Gate 用の傾き k")
    parser.add_argument("--max_steps_to_print", type=int, default=20,
                        help="何ステップ目まで詳細ログを表示するか")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.mid_npz):
        print(f"[ERROR] mid_npz not found: {args.mid_npz}")
        sys.exit(1)

    print(f"[INFO] Loading MID cache from: {args.mid_npz}")
    z = np.load(args.mid_npz)

    if "region_per_pt" not in z or "point_adj" not in z or "pc" not in z:
        print("[ERROR] .npz に 'region_per_pt', 'point_adj', 'pc' のいずれかが欠けています。")
        print("       SD-CS / AMA の MID キャッシュ (.npz) を指定してください。")
        sys.exit(1)

    region_per_pt = z["region_per_pt"].astype(np.float32).reshape(-1)
    point_adj = z["point_adj"].astype(np.float32).reshape(-1)
    pc = z["pc"].astype(np.float32)

    if pc.shape[0] != region_per_pt.shape[0]:
        print("[WARN] pc.shape[0] と region_per_pt の長さが一致していません。")
        print(f"       pc.shape = {pc.shape}, region_per_pt.shape = {region_per_pt.shape}")
        print("       ひとまず min(len) に合わせて切り詰めます。")
        n = min(pc.shape[0], region_per_pt.shape[0])
        pc = pc[:n]
        region_per_pt = region_per_pt[:n]
        point_adj = point_adj[:n]

    num_points = pc.shape[0]
    N_target = int(min(max(args.ds_points, 1), num_points))

    print("==============================================")
    print("[INFO] Sample loaded")
    print(f"  num_points        : {num_points}")
    print(f"  requested ds_points (N): {args.ds_points} -> actual N_target: {N_target}")
    print(f"  lambda_scale      : {args.lambda_scale}")
    print("==============================================")

    # alpha の決定
    var_reg = float(np.var(region_per_pt))
    var_pnt = float(np.var(point_adj))

    if args.alpha_fixed is not None:
        alpha = float(args.alpha_fixed)
        print(f"[INFO] Using fixed alpha = {alpha}")
    else:
        alpha = heuristic_alpha(
            N=N_target,
            var_reg=var_reg,
            var_pnt=var_pnt,
            Nmid=args.ama_Nmid,
            k=args.ama_k
        )
        print("[INFO] Using heuristic alpha")
        print(f"  var_reg = {var_reg:.6e}, var_pnt = {var_pnt:.6e}")
        print(f"  ama_Nmid = {args.ama_Nmid}, ama_k = {args.ama_k}")
        print(f"  -> alpha = {alpha:.6f}")

    # Hybrid スコア
    hybrid_scores = alpha * region_per_pt + (1.0 - alpha) * point_adj
    hs_min = float(hybrid_scores.min())
    hs_max = float(hybrid_scores.max())
    hs_std = float(hybrid_scores.std())
    hs_mean = float(hybrid_scores.mean())

    print("==============================================")
    print("[DEBUG_INIT] Hybrid score statistics")
    print(f"  hybrid_min  = {hs_min:.6e}")
    print(f"  hybrid_max  = {hs_max:.6e}")
    print(f"  hybrid_mean = {hs_mean:.6e}")
    print(f"  hybrid_std  = {hs_std:.6e}")
    print("==============================================")

    # coverage_target
    cov_target = get_coverage_target(N_target)
    print(f"[INFO] coverage_target(N={N_target}) = {cov_target:.6f}")

    # Coverage 状態
    cov_state = GlobalCoverageState(pc)

    selected_flag = np.zeros(num_points, dtype=bool)
    selected_indices = []

    print("==============================================")
    print("[INFO] Start SD-CS-like selection debug")
    print("  (step, chosen_idx, hybrid(chosen), d_norm(chosen), "
          "lambda, lambda*d_norm(chosen), coverage_after, "
          "lambda*d_norm.max)")
    print("==============================================")

    for step in range(N_target):
        if step == 0:
            # 初回は Hybrid のみで選択
            lam = 0.0
            coverage_before = cov_state.get_coverage()
            d_norm = np.zeros(num_points, dtype=np.float32)
            scores = hybrid_scores.copy()
            scores[selected_flag] = -np.inf

            if not np.isfinite(scores).any():
                print("[ERROR] All scores are -inf at step 0.")
                break

            chosen_idx = int(np.argmax(scores))
            selected_indices.append(chosen_idx)
            selected_flag[chosen_idx] = True
            cov_state.update_with_point(chosen_idx)
            coverage_after = cov_state.get_coverage()

            if step < args.max_steps_to_print:
                print(f"[STEP {step:03d}] "
                      f"idx={chosen_idx:04d}, "
                      f"hybrid={float(hybrid_scores[chosen_idx]): .6e}, "
                      f"d_norm={0.0: .6e}, "
                      f"lambda={lam: .6e}, "
                      f"lambda*d_norm={0.0: .6e}, "
                      f"coverage_after={coverage_after: .6f}, "
                      f"lambda*d_norm_max={0.0: .6e}")
            continue

        # 2 回目以降は coverage と距離を考慮
        coverage_before = cov_state.get_coverage()
        d_norm = cov_state.get_normalized_distances()
        lam = compute_lambda_cov(
            N=N_target,
            step_idx=step,
            coverage=coverage_before,
            coverage_target=cov_target,
            lambda_scale=args.lambda_scale
        )

        scores = hybrid_scores + lam * d_norm
        scores[selected_flag] = -np.inf

        if not np.isfinite(scores).any():
            # fallback: Hybrid のみ
            scores = hybrid_scores.copy()
            scores[selected_flag] = -np.inf

        if not np.isfinite(scores).any():
            print(f"[ERROR] All scores are -inf at step {step}. Stopping.")
            break

        chosen_idx = int(np.argmax(scores))
        selected_indices.append(chosen_idx)
        selected_flag[chosen_idx] = True
        cov_state.update_with_point(chosen_idx)
        coverage_after = cov_state.get_coverage()

        # debug 出力
        if step < args.max_steps_to_print:
            d_chosen = float(d_norm[chosen_idx])
            term_fps_chosen = lam * d_chosen
            term_fps_max = lam * float(d_norm.max())
            print(f"[STEP {step:03d}] "
                  f"idx={chosen_idx:04d}, "
                  f"hybrid={float(hybrid_scores[chosen_idx]): .6e}, "
                  f"d_norm={d_chosen: .6e}, "
                  f"lambda={lam: .6e}, "
                  f"lambda*d_norm={term_fps_chosen: .6e}, "
                  f"coverage_after={coverage_after: .6f}, "
                  f"lambda*d_norm_max={term_fps_max: .6e}")

    print("==============================================")
    print("[INFO] Selection finished.")
    print(f"  N_target       = {N_target}")
    print(f"  selected_count = {len(selected_indices)}")
    final_cov = cov_state.get_coverage()
    print(f"  final_coverage = {final_cov:.6f}")
    print("==============================================")


if __name__ == "__main__":
    main()
