

CUDA_VISIBLE_DEVICES=0 python monte_region_SHAP_DS_for_PointNeXt.py --pattern 10000 --division 32 && CUDA_VISIBLE_DEVICES=0 python monte_region_SHAP_DS_for_PointNeXt.py --pattern 10000 --division 64

[SD-GLM（L1正規化+weight-Estimator）]
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 && \
CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --ds_points 300 --ama --ama_mode heuristic --gate_dump_train --gate_dump_dir result/_gate_train_dump --cache_mode save --pattern 10000 --division_region 32 --division_point 64 && \
CUDA_VISIBLE_DEVICES=0 python train_gate_AMAmlp.py --division_region 32 --division_point 64 --max_files 10 --max_samples_per_file 3000 --alpha_grid 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --beta_grid 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --dump_dir result/_gate_train_dump/p10000_kr32_kp64_heuristic --out_npz result/_gate_train_dump/gate_mlp_train_data_n300_500_800_1000.npz --n_list 300 500 800 1000 && \
CUDA_VISIBLE_DEVICES=0 python fit_gate_AMAmlp.py --train_npz result/_gate_train_dump/gate_mlp_train_data_n300_500_800_1000.npz --n_list 300 500 800 1000 && \
for ds_points in 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --ds_points $ds_points --ama --ama_mode mlp --mask_chunk 512 --min_mask_chunk 16 --gate_ckpt result/_gate_train_dump/gate_mlp_n300_500_800_1000.pth --gate_scaler result/_gate_train_dump/gate_scaler_n300_500_800_1000.npz --pattern 10000 --division_region 32 --division_point 64; done && \
CUDA_VISIBLE_DEVICES=0 python train_gate_AMAmlp.py --division_region 32 --division_point 64 --pattern 10000 --max_files 10 --max_samples_per_file 3000 --alpha_grid 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --beta_grid 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --dump_dir result/_gate_train_dump/p10000_kr32_kp64_heuristic --out_npz result/_gate_train_dump/gate_mlp_train_data_n300_400_500_600_700_800_900_1000.npz --n_list 300 400 500 600 700 800 900 1000 && \
CUDA_VISIBLE_DEVICES=0 python fit_gate_AMAmlp.py --train_npz result/_gate_train_dump/gate_mlp_train_data_n300_400_500_600_700_800_900_1000.npz --n_list 300 400 500 600 700 800 900 1000 && \
for ds_points in 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --ds_points $ds_points --ama --ama_mode mlp --mask_chunk 512 --min_mask_chunk 16 --gate_ckpt result/_gate_train_dump/gate_mlp_n300_400_500_600_700_800_900_1000.pth --gate_scaler result/_gate_train_dump/gate_scaler_n300_400_500_600_700_800_900_1000.npz --pattern 10000 --division_region 32 --division_point 64; done

[{100,200}点_SD-GLM（L1正規化+weight-Estimator）]
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 && \
CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --ds_points 300 --ama --ama_mode heuristic --gate_dump_train --gate_dump_dir result/_gate_train_dump --cache_mode load --pattern 10000 --division_region 32 --division_point 64 && \
CUDA_VISIBLE_DEVICES=0 python train_gate_AMAmlp.py --division_region 32 --division_point 64 --max_files 10 --max_samples_per_file 3000 --alpha_grid 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --beta_grid 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --dump_dir result/_gate_train_dump/p10000_kr32_kp64_heuristic --out_npz result/_gate_train_dump/gate_mlp_train_data_n100_200.npz --n_list 100 200 && \
CUDA_VISIBLE_DEVICES=0 python fit_gate_AMAmlp.py --train_npz result/_gate_train_dump/gate_mlp_train_data_n100_200.npz --n_list 100 200 && \
for ds_points in 100 200; do CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --ds_points $ds_points --ama --ama_mode mlp --mask_chunk 512 --min_mask_chunk 16 --gate_ckpt result/_gate_train_dump/gate_mlp_n100_200.pth --gate_scaler result/_gate_train_dump/gate_scaler_n100_200.npz --pattern 10000 --division_region 32 --division_point 64 --cache_mode save; done

[モジュール別検証（L1のみ）（weight-Estimatorなし）]
for ds_points in 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python L1_noWeight_SHAP_for_PointNeXt.py --ds_points $ds_points --pattern 10000 --division_region 32 --division_point 64; done

[モジュール別検証（weight-Estimatorのみ）（L1なし）]
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 && \
CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --ds_points 300 --ama --ama_mode heuristic --gate_dump_train --gate_dump_dir result/_gate_train_dump --cache_mode save --pattern 10000 --division_region 32 --division_point 64 --no_l1 --force_mid_build && \
CUDA_VISIBLE_DEVICES=0 python train_gate_AMAmlp.py --division_region 32 --division_point 64 --max_files 10 --max_samples_per_file 3000 --alpha_grid 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --beta_grid 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --dump_dir result/_gate_train_dump/p10000_kr32_kp64_heuristic_nol1 --out_npz result/_gate_train_dump/gate_mlp_train_data_n300_500_800_1000.npz --n_list 300 500 800 1000 && \
CUDA_VISIBLE_DEVICES=0 python fit_gate_AMAmlp.py --train_npz result/_gate_train_dump/gate_mlp_train_data_n300_500_800_1000.npz --n_list 300 500 800 1000 && \
for ds_points in 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --ds_points $ds_points --ama --ama_mode mlp --mask_chunk 512 --min_mask_chunk 16 --gate_ckpt   result/_gate_train_dump/gate_mlp_n300_500_800_1000.pth --gate_scaler result/_gate_train_dump/gate_scaler_n300_500_800_1000.npz --pattern 10000 --division_region 32 --division_point 64 --no_l1 --cache_mode save; done && \
CUDA_VISIBLE_DEVICES=0 python train_gate_AMAmlp.py --division_region 32 --division_point 64 --max_files 10 --max_samples_per_file 3000 --alpha_grid 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --beta_grid 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --dump_dir result/_gate_train_dump/p10000_kr32_kp64_heuristic_nol1 --out_npz result/_gate_train_dump/gate_mlp_train_data_n300_400_500_600_700_800_900_1000.npz --n_list 300 400 500 600 700 800 900 1000 && \
CUDA_VISIBLE_DEVICES=0 python fit_gate_AMAmlp.py --train_npz result/_gate_train_dump/gate_mlp_train_data_n300_400_500_600_700_800_900_1000.npz --n_list 300 400 500 600 700 800 900 1000 && \
for ds_points in 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --ds_points $ds_points --ama --ama_mode mlp --mask_chunk 512 --min_mask_chunk 16 --gate_ckpt   result/_gate_train_dump/gate_mlp_n300_400_500_600_700_800_900_1000.pth --gate_scaler result/_gate_train_dump/gate_scaler_n300_400_500_600_700_800_900_1000.npz --pattern 10000 --division_region 32 --division_point 64 --no_l1 --cache_mode save; done

[重み付け係数(α,β)のplotグラフ作成]
CUDA_VISIBLE_DEVICES=0 python train_gate_AMAmlp.py \
  --division_region 32 --division_point 64 \
  --max_files 10 --max_samples_per_file 3000 \
  --alpha_grid 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --beta_grid 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \
  --dump_dir result/_gate_train_dump/p10000_kr32_kp64_heuristic \
  --out_npz result/_gate_train_dump/gate_mlp_train_data_n300_500_800_1000.npz \
  --n_list 300 500 800 1000 \
  --plot_alpha_beta --plot_n_min 300 --plot_n_max 1000 &&\
CUDA_VISIBLE_DEVICES=0 python train_gate_AMAmlp.py \
  --division_region 32 --division_point 64 \
  --max_files 10 --max_samples_per_file 3000 \
  --alpha_grid 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --beta_grid 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \
  --dump_dir result/_gate_train_dump/p10000_kr32_kp64_heuristic \
  --out_npz result/_gate_train_dump/gate_mlp_train_data_n300_400_500_600_700_800_900_1000.npz \
  --n_list 300 400 500 600 700 800 900 1000 \
  --plot_alpha_beta --plot_n_min 300 --plot_n_max 1000


[二段階サンプリング（FPS:500点→SD-GLM:300,400点）]
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 &&\
CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --dataset /workspace/PointNeXt/result/FPS_CPU/500/modelnet40_ply_hdf5_2048 --ds_points 300 --ama --ama_mode heuristic --gate_dump_train --gate_dump_dir result/_gate_train_dump --cache_mode save --pattern 10000 --division_region 32 --division_point 64 --run_tag stage2_fps500 --force_mid_build &&\
CUDA_VISIBLE_DEVICES=0 python train_gate_AMAmlp.py --division_region 32 --division_point 64 --pattern 10000 --max_files 10 --max_samples_per_file 3000 --alpha_grid 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --beta_grid 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --dump_dir result/_gate_train_dump/p10000_kr32_kp64_heuristic_stage2_fps500 --out_npz result/_gate_train_dump/stage2_fps500_gate_mlp_train_data_n100_200_300_400.npz --n_list 100 200 300 400 &&\
CUDA_VISIBLE_DEVICES=0 python fit_gate_AMAmlp.py --train_npz result/_gate_train_dump/stage2_fps500_gate_mlp_train_data_n100_200_300_400.npz --n_list 100 200 300 400 --out_pth result/_gate_train_dump/stage2_fps500_gate_mlp.pth --out_scaler result/_gate_train_dump/stage2_fps500_gate_scaler.npz &&\
for ds_points in 100 200 300 400; do CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --dataset /workspace/PointNeXt/result/FPS_CPU/500/modelnet40_ply_hdf5_2048 --ds_points $ds_points --ama --ama_mode mlp --mask_chunk 512 --min_mask_chunk 16 --gate_ckpt result/_gate_train_dump/stage2_fps500_gate_mlp_n100_200_300_400.pth --gate_scaler result/_gate_train_dump/stage2_fps500_gate_scaler_n100_200_300_400.npz --pattern 10000 --division_region 32 --division_point 64 --run_tag stage2_fps500; done



[二段階サンプリング]
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 &&\
CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --dataset /workspace/PointNeXt/result/FPS_CPU/800/modelnet40_ply_hdf5_2048 --ds_points 300 --ama --ama_mode heuristic --gate_dump_train --gate_dump_dir result/_gate_train_dump --cache_mode save --pattern 10000 --division_region 32 --division_point 64 --run_tag stage2_fps800 --force_mid_build &&\
CUDA_VISIBLE_DEVICES=0 python train_gate_AMAmlp.py --division_region 32 --division_point 64 --pattern 10000 --max_files 10 --max_samples_per_file 3000 --alpha_grid 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --beta_grid 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --dump_dir result/_gate_train_dump/p10000_kr32_kp64_heuristic_stage2_fps800 --out_npz result/_gate_train_dump/stage2_fps800_gate_mlp_train_data_n100_200_300_400_500_600_700.npz --n_list 100 200 300 400 500 600 700 &&\
CUDA_VISIBLE_DEVICES=0 python fit_gate_AMAmlp.py --train_npz result/_gate_train_dump/stage2_fps800_gate_mlp_train_data_n100_200_300_400_500_600_700.npz --n_list 100 200 300 400 500 600 700 --out_pth result/_gate_train_dump/stage2_fps800_gate_mlp.pth --out_scaler result/_gate_train_dump/stage2_fps800_gate_scaler.npz &&\
for ds_points in 100 200 300 400 500 600 700; do CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --dataset /workspace/PointNeXt/result/FPS_CPU/800/modelnet40_ply_hdf5_2048 --ds_points $ds_points --ama --ama_mode mlp --mask_chunk 512 --min_mask_chunk 16 --gate_ckpt result/_gate_train_dump/stage2_fps800_gate_mlp_n100_200_300_400_500_600_700.pth --gate_scaler result/_gate_train_dump/stage2_fps800_gate_scaler_n100_200_300_400_500_600_700.npz --pattern 10000 --division_region 32 --division_point 64 --run_tag stage2_fps800 --cache_mode save; done &&\
CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --dataset /workspace/PointNeXt/result/AMA-mlp-gate_mlp_n300_400_500_600_700_800_900_1000_kr32_kp64_p10000_dsSHAP_PointNeXt_h5/800/modelnet40_ply_hdf5_2048 --ds_points 300 --ama --ama_mode heuristic --gate_dump_train --gate_dump_dir result/_gate_train_dump --cache_mode save --pattern 10000 --division_region 32 --division_point 64 --run_tag stage2_sdglm800 --force_mid_build &&\
CUDA_VISIBLE_DEVICES=0 python train_gate_AMAmlp.py --division_region 32 --division_point 64 --pattern 10000 --max_files 10 --max_samples_per_file 3000 --alpha_grid 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --beta_grid 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --dump_dir result/_gate_train_dump/p10000_kr32_kp64_heuristic_stage2_sdglm800 --out_npz result/_gate_train_dump/stage2_sdglm800_gate_mlp_train_data_n100_200_300_400_500_600_700.npz --n_list 100 200 300 400 500 600 700 &&\
CUDA_VISIBLE_DEVICES=0 python fit_gate_AMAmlp.py --train_npz result/_gate_train_dump/stage2_sdglm800_gate_mlp_train_data_n100_200_300_400_500_600_700.npz --n_list 100 200 300 400 500 600 700 --out_pth result/_gate_train_dump/stage2_sdglm800_gate_mlp.pth --out_scaler result/_gate_train_dump/stage2_sdglm800_gate_scaler.npz &&\
for ds_points in 100 200 300 400 500 600 700; do CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --dataset /workspace/PointNeXt/result/AMA-mlp-gate_mlp_n300_400_500_600_700_800_900_1000_kr32_kp64_p10000_dsSHAP_PointNeXt_h5/800/modelnet40_ply_hdf5_2048 --ds_points $ds_points --ama --ama_mode mlp --mask_chunk 512 --min_mask_chunk 16 --gate_ckpt result/_gate_train_dump/stage2_sdglm800_gate_mlp_n100_200_300_400_500_600_700.pth --gate_scaler result/_gate_train_dump/stage2_sdglm800_gate_scaler_n100_200_300_400_500_600_700.npz --pattern 10000 --division_region 32 --division_point 64 --run_tag stage2_sdglm800 --cache_mode save; done &&\
CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --dataset /workspace/PointNeXt/result/AMA-mlp-gate_mlp_n300_400_500_600_700_800_900_1000_kr32_kp64_p10000_dsSHAP_PointNeXt_h5/500/modelnet40_ply_hdf5_2048 --ds_points 300 --ama --ama_mode heuristic --gate_dump_train --gate_dump_dir result/_gate_train_dump --cache_mode save --pattern 10000 --division_region 32 --division_point 64 --run_tag stage2_sdglm500 --force_mid_build &&\
CUDA_VISIBLE_DEVICES=0 python train_gate_AMAmlp.py --division_region 32 --division_point 64 --pattern 10000 --max_files 10 --max_samples_per_file 3000 --alpha_grid 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --beta_grid 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --dump_dir result/_gate_train_dump/p10000_kr32_kp64_heuristic_stage2_sdglm500 --out_npz result/_gate_train_dump/stage2_sdglm500_gate_mlp_train_data_n100_200_300_400.npz --n_list 100 200 300 400 &&\
CUDA_VISIBLE_DEVICES=0 python fit_gate_AMAmlp.py --train_npz result/_gate_train_dump/stage2_sdglm500_gate_mlp_train_data_n100_200_300_400.npz --n_list 100 200 300 400 --out_pth result/_gate_train_dump/stage2_sdglm500_gate_mlp.pth --out_scaler result/_gate_train_dump/stage2_sdglm500_gate_scaler.npz &&\
for ds_points in 100 200 300 400; do CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --dataset /workspace/PointNeXt/result/AMA-mlp-gate_mlp_n300_400_500_600_700_800_900_1000_kr32_kp64_p10000_dsSHAP_PointNeXt_h5/500/modelnet40_ply_hdf5_2048 --ds_points $ds_points --ama --ama_mode mlp --mask_chunk 512 --min_mask_chunk 16 --gate_ckpt result/_gate_train_dump/stage2_sdglm500_gate_mlp_n100_200_300_400.pth --gate_scaler result/_gate_train_dump/stage2_sdglm500_gate_scaler_n100_200_300_400.npz --pattern 10000 --division_region 32 --division_point 64 --run_tag stage2_sdglm500 --cache_mode save; done


[ヒストグラムRegion]
[SD-GLM]
for N in 100 200 300 400 500 600 700 800 900 1000; do   CUDA_VISIBLE_DEVICES=0 python animate_Region_StackHistogram.py --mid_cache_dir /workspace/PointNeXt/result/_shap_cache_mid/p10000_kr32_kp64_ckpt-modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT_ckpt_best --dataset /workspace/PointNeXt/data/ModelNet40Ply2048/modelnet40_ply_hdf5_2048 --division_region 32 --division_point 64 --sample_index 0 --ama --ama_mode mlp --gate_ckpt result/_gate_train_dump/gate_mlp_n100_200_300_400_500_600_700_800_900_1000.pth --gate_scaler result/_gate_train_dump/gate_scaler_n100_200_300_400_500_600_700_800_900_1000.npz --out_gif script_Analysis/Histogram/hist_result/region_stack_sample0_N${N}.gif --fps 10 --target_N ${N} --step 5; done
[RR]
for N in 100 200 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python script_Analysis/Histogram/RR_animation_StackHistogram.py --mid_cache_dir /workspace/PointNeXt/result/_shap_cache_mid/p10000_kr32_kp64_ckpt-modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT_ckpt_best --dataset /workspace/PointNeXt/data/ModelNet40Ply2048/modelnet40_ply_hdf5_2048 --division_region 32 --division_point 64 --sample_index 0 --target_N $N --step 5 --fps 10; done

[αのみSD-GLM]
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 && CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --ds_points 300 --ama --ama_mode heuristic --gate_dump_train --gate_dump_dir result/_gate_train_dump --cache_mode load --pattern 10000 --division_region 32 --division_point 64 && CUDA_VISIBLE_DEVICES=0 python train_gate_AMAmlp.py --division_region 32 --division_point 64 --max_files 10 --max_samples_per_file 3000 --alpha_grid 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.8 0.85 0.9 0.95 1.00 --beta_grid 1.0 --dump_dir result/_gate_train_dump/p10000_kr32_kp64_heuristic --out_npz result/_gate_train_dump/gate_mlp_train_data_noBeta.npz --n_list 100 200 300 400 500 600 700 800 900 1000 && CUDA_VISIBLE_DEVICES=0 python fit_gate_AMAmlp.py --train_npz result/_gate_train_dump/gate_mlp_train_data_noBeta.npz --n_list 100 200 300 400 500 600 700 800 900 1000 --out_pth result/_gate_train_dump/gate_mlp_noBeta.pth --out_scaler result/_gate_train_dump/gate_scaler_noBeta.npz --no_auto_suffix && for ds_points in 100 200 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --ds_points $ds_points --ama --ama_mode mlp --mask_chunk 512 --min_mask_chunk 16 --gate_ckpt result/_gate_train_dump/gate_mlp_noBeta.pth --gate_scaler result/_gate_train_dump/gate_scaler_noBeta.npz --pattern 10000 --division_region 32 --division_point 64 --cache_mode save; done

[Region-SHAP順位に基づくラウンドロビンPoint-SHAP選択による均衡型点群ダウンサンプリング]
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 &&\
for ds_points in 100 200 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python RoundRobin_SHAP_for_PointNeXt.py --ds_points ${ds_points} --pattern 10000 --division_region 32 --division_point 64 --cache_mode load --no_l1 --mask_chunk 512 --min_mask_chunk 16; done


[タスクスケジューリング型点群ダウンサンプリング]
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 && \
for ds_points in 100 200 300 400 500 600 700 800 900 1000; do \
  CUDA_VISIBLE_DEVICES=0 python scheduling_SHAP_for_PointNeXt.py \
    --ds_points $ds_points \
    --ama --ama_mode mlp \
    --gate_ckpt   result/_gate_train_dump/gate_mlp_noBeta.pth \
    --gate_scaler result/_gate_train_dump/gate_scaler_noBeta.npz \
    --pattern 10000 \
    --division_region 32 --division_point 64 \
    --mask_chunk 512 --min_mask_chunk 16 \
    --cache_mode save \
    --run_tag scheduling; \
done


[SD-CS:SHAPDownsamplingwithCoverageScheduling]
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 && \
for ds_points in 100 200 300 400 500 600 700 800 900 1000; do \
  CUDA_VISIBLE_DEVICES=0 python SD-CS_for_PointNeXt.py \
    --ds_points ${ds_points} \
    --ama --ama_mode mlp \
    --gate_ckpt   result/_gate_train_dump/gate_mlp_noBeta.pth \
    --gate_scaler result/_gate_train_dump/gate_scaler_noBeta.npz \
    --pattern 10000 \
    --division_region 32 \
    --division_point 64 \
    --mask_chunk 512 \
    --min_mask_chunk 16 \
    --cache_mode auto; \
done


[低N域でFPS傾向強化_SD-CS]
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 && \
for ds_points in 100 200 300 400 500 600 700 800 900 1000; do \
  CUDA_VISIBLE_DEVICES=0 python Ndepend_SD-CS.py \
    --ds_points ${ds_points} \
    --ama --ama_mode mlp \
    --gate_ckpt result/_gate_train_dump/gate_mlp_noBeta.pth \
    --gate_scaler result/_gate_train_dump/gate_scaler_noBeta.npz \
    --pattern 10000 \
    --division_region 32 \
    --division_point 64 \
    --mask_chunk 512 \
    --min_mask_chunk 16 \
    --cache_mode auto; \
done

[SDCS_Hy重み係数_モジュールON/OFF]
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 && \
for ds_points in 200 300 400 500 600 700 800 900 1000; do \
  CUDA_VISIBLE_DEVICES=0 python switch_SD-CS.py \
    --ds_points ${ds_points} \
    --ama --ama_mode mlp \
    --gate_ckpt result/_gate_train_dump/gate_mlp_noBeta.pth \
    --gate_scaler result/_gate_train_dump/gate_scaler_noBeta.npz \
    --pattern 10000 \
    --division_region 32 \
    --division_point 64 \
    --mask_chunk 512 \
    --min_mask_chunk 16 \
    --cache_mode auto \
    --run_tag allOn;
done &&\
for ds_points in 200 300 400 500 600 700 800 900 1000; do \
  CUDA_VISIBLE_DEVICES=0 python switch_SD-CS.py \
    --ds_points ${ds_points} \
    --ama --ama_mode mlp \
    --gate_ckpt result/_gate_train_dump/gate_mlp_noBeta.pth \
    --gate_scaler result/_gate_train_dump/gate_scaler_noBeta.npz \
    --pattern 10000 \
    --division_region 32 \
    --division_point 64 \
    --mask_chunk 512 \
    --min_mask_chunk 16 \
    --cache_mode auto \
    --disable_cov \
    --run_tag noCov;
done && \
for ds_points in 100 200 300 400 500 600 700 800 900 1000; do \
  CUDA_VISIBLE_DEVICES=0 python switch_SD-CS.py \
    --ds_points ${ds_points} \
    --ama --ama_mode mlp \
    --gate_ckpt result/_gate_train_dump/gate_mlp_noBeta.pth \
    --gate_scaler result/_gate_train_dump/gate_scaler_noBeta.npz \
    --pattern 10000 \
    --division_region 32 \
    --division_point 64 \
    --mask_chunk 512 \
    --min_mask_chunk 16 \
    --cache_mode auto \
    --disable_fps \
    --run_tag noFPS;
done

