
### Coding languages
Python, JavaSprict, Java, C
### keywords
pointcloud, deepleaning, downsampling, compression, SHAP, shapley-value



# SHAP Downsampling
## Usage

### 1. Preparation: Training the Gate Model

First, run the following commands to generate training data, process it, and fit the Gate model. This is a prerequisite for the evaluation steps.

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 && \
CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py \
    --ds_points 300 \
    --ama \
    --ama_mode heuristic \
    --gate_dump_train \
    --gate_dump_dir result/_gate_train_dump \
    --cache_mode load \
    --pattern 10000 \
    --division_region 32 \
    --division_point 64 && \
CUDA_VISIBLE_DEVICES=0 python train_gate_AMAmlp.py \
    --division_region 32 \
    --division_point 64 \
    --max_files 10 \
    --max_samples_per_file 3000 \
    --alpha_grid 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.8 0.85 0.9 0.95 1.00 \
    --beta_grid 1.0 \
    --dump_dir result/_gate_train_dump/p10000_kr32_kp64_heuristic \
    --out_npz result/_gate_train_dump/gate_mlp_train_data_noBeta.npz \
    --n_list 100 200 300 400 500 600 700 800 900 1000 && \
CUDA_VISIBLE_DEVICES=0 python fit_gate_AMAmlp.py \
    --train_npz result/_gate_train_dump/gate_mlp_train_data_noBeta.npz \
    --n_list 100 200 300 400 500 600 700 800 900 1000 \
    --out_pth result/_gate_train_dump/gate_mlp_noBeta.pth \
    --out_scaler result/_gate_train_dump/gate_scaler_noBeta.npz \
    --no_auto_suffix
```

### 2. Evaluation Loops
After the preparation is complete, you can run the evaluation loops for different methods.

**SHAP Downsampling (AMA-MLP)**

```bash
for ds_points in 100 200 300 400 500 600 700 800 900 1000; do \
    CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py \
        --ds_points $ds_points \
        --ama \
        --ama_mode mlp \
        --mask_chunk 512 \
        --min_mask_chunk 16 \
        --gate_ckpt result/_gate_train_dump/gate_mlp_noBeta.pth \
        --gate_scaler result/_gate_train_dump/gate_scaler_noBeta.npz \
        --pattern 10000 \
        --division_region 32 \
        --division_point 64 \
        --cache_mode save; \
done
```

**SHAP-Based Coverage-Aware Downsampling**

```bash
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
        --run_tag allOn; \
done
```


# Task Execution with PointNeXt
## Usage

### 1. Classification (ModelNet40)

**Train:**

```bash
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml
```

**Test (Downsampled Point Cloud):**
Note: Regarding the input data, please modify the `[data_dir]` and `[test:num_points]` settings within `/workspace/PointNeXt/cfgs/modelnet40ply2048/default.yaml`.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py \
    --cfg cfgs/modelnet40ply2048/pointnext-s.yaml \
    mode=test \
    --pretrained_path /workspace/PointNeXt/log/modelnet40ply2048/modelnet40ply2048-train-pointnext-*/checkpoint/modelnet40ply2048-train-pointnext-*_ckpt_best.pth
```

### 2. Part Segmentation (ShapeNet)

**Train:**

```bash
CUDA_VISIBLE_DEVICES=0 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointnext-s.yaml
```

**Test:**

```bash
CUDA_VISIBLE_DEVICES=0 python examples/shapenetpart/main.py \
    --cfg cfgs/shapenetpart/pointnext-s.yaml \
    mode=test \
    --pretrained_path /workspace/PointNeXt/log/shapenetpart/shapenetpart-train-pointnext-*/checkpoint/shapenetpart-train-pointnext-*L_ckpt_best.pth
```

### 3. Segmentation (S3DIS, ScanNet)
**S3DIS:**

```bash
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py --cfg cfgs/s3dis/pointnext-xl.yaml
```

**ScanNet:**

```bash
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py --cfg cfgs/scannet/pointnext-xl.yaml
```
