
[[[Hybrid-SHAPダウンサンプリング]]]
[キャッシュを使って、RegionとPointで"k"を変えて実行]
for ds_points in 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python division_SHAP_DS_for_PointNeXt.py --ds_points $ds_points --num_groups 1 --weight 1 --pattern 10000 --division_region 32 --division_point 64; done

[モンテカルロシミュレーションにてRegionの近似誤差幅を計算]
CUDA_VISIBLE_DEVICES=0 python monte_region_SHAP_DS_for_PointNeXt.py --pattern 10000 --division 32

[Region内でそれぞれ何点を取っているかのヒストグラム出力]
[1サンプルの「スタックしていく様子」をgifにするスクリプト]
for N in 100 200 300 400 500 600 700 800 900 1000; do   CUDA_VISIBLE_DEVICES=0 python animate_Region_StackHistogram.py     --mid_cache_dir /workspace/PointNeXt/result/_shap_cache_mid/p10000_kr32_kp64_ckpt-modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT_ckpt_best     --dataset /workspace/PointNeXt/data/ModelNet40Ply2048/modelnet40_ply_hdf5_2048     --division_region 32     --division_point 64     --sample_index 0     --ama     --ama_mode mlp     --gate_ckpt result/_gate_train_dump/gate_mlp_n100_200_300_400_500_600_700_800_900_1000.pth     --gate_scaler result/_gate_train_dump/gate_scaler_n100_200_300_400_500_600_700_800_900_1000.npz     --out_gif result/_region_hist_debug/region_stack_sample0_N${N}.gif     --fps 10     --target_N ${N}     --step 5; done
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

[AMA]
<AMAヒューリス,AFなし>
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 && for ds_points in 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --ds_points $ds_points --ama --ama_mode heuristic --mask_chunk 512 --min_mask_chunk 16 --pattern 10000 --division_region 32 --division_point 64; done

<AMA=MLP(学習済みGate使用),AFあり(featをh5に保存)>（未確認）
for ds_points in 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --ds_points $ds_points --ama --ama_mode mlp --gate_ckpt gate_mlp.pth --gate_scaler gate_scaler.npz --pattern 10000 --division_region 32 --division_point 64; done


[[[AMA=MLPの本番推論準備方法と実行]]]
1. 特徴量ダンプ生成（精度重視でSHAP計算 → 教師データ作り）
2. MLP学習（ダンプから α, β を予測するモデルを構築）
3. 本番推論（MLPを使って α, β を自動決定）



1. [ゲートMLPの学習に使う教師データである入力特徴量(N,σ²_R,σ²_P,それに対応する最適α,β)を収集](教師ダンプ)
☆L1あり
CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --ds_points 300 --ama --ama_mode heuristic --gate_dump_train --gate_dump_dir result/_gate_train_dump --cache_mode save --pattern 10000 --division_region 32 --division_point 64

☆L1なし
CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --ds_points 300 --ama --ama_mode heuristic --gate_dump_train --gate_dump_dir result/_gate_train_dump --cache_mode save --pattern 10000 --division_region 32 --division_point 64 --no_l1


2. [ダンプデータを読み込み、(N,σ²_R,σ²_P)→(α,β)を回帰するMLPを学習](ダンプ→学習データ)
<pattern A>
CUDA_VISIBLE_DEVICES=0 python train_gate_AMAmlp.py --division_region 32 --division_point 64 --pattern 10000 --max_files 10 --max_samples_per_file 3000 --dump_dir result/_gate_train_dump --alpha_grid 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --beta_grid 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --out_npz result/_gate_train_dump/gate_mlp_train_data_n300_500_800_1000.npz --n_list 300 500 800 1000
<pattern B>
CUDA_VISIBLE_DEVICES=0 python train_gate_AMAmlp.py --division_region 32 --division_point 64 --pattern 10000 --max_files 10 --max_samples_per_file 3000 --alpha_grid 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --beta_grid 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --dump_dir result/_gate_train_dump --out_npz result/_gate_train_dump/gate_mlp_train_data_n300_400_500_600_700_800_900_1000.npz --n_list 300 400 500 600 700 800 900 1000


3. [NごとにGate-MLPを学習して別名保存](Gate-MLP学習)
<pattern A>
CUDA_VISIBLE_DEVICES=0 python fit_gate_AMAmlp.py --train_npz result/_gate_train_dump/gate_mlp_train_data_n300_500_800_1000.npz --n_list 300 500 800 1000
<pattern B>
CUDA_VISIBLE_DEVICES=0 python fit_gate_AMAmlp.py --train_npz result/_gate_train_dump/gate_mlp_train_data_n300_400_500_600_700_800_900_1000.npz --n_list 300 400 500 600 700 800 900 1000


4. [学習済みMLPを使って、各入力に対する最適α,βを自動推定し、AMA-SHAPを実行](AMA-SHAP実行)
☆L1あり
<pattern A>
for ds_points in 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --ds_points $ds_points --ama --ama_mode mlp --mask_chunk 512 --min_mask_chunk 16 --gate_ckpt result/_gate_train_dump/gate_mlp_n300_500_800_1000.pth --gate_scaler result/_gate_train_dump/gate_scaler_n300_500_800_1000.npz --pattern 10000 --division_region 32 --division_point 64; done
<pattern B>
for ds_points in 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --ds_points $ds_points --ama --ama_mode mlp --mask_chunk 512 --min_mask_chunk 16 --gate_ckpt result/_gate_train_dump/gate_mlp_n300_400_500_600_700_800_900_1000.pth --gate_scaler result/_gate_train_dump/gate_scaler_n300_400_500_600_700_800_900_1000.npz --pattern 10000 --division_region 32 --division_point 64; done

☆L1なし
<pattern A>
for ds_points in 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --ds_points $ds_points --ama --ama_mode mlp --mask_chunk 512 --min_mask_chunk 16 --gate_ckpt result/_gate_train_dump/gate_mlp_n300_500_800_1000.pth --gate_scaler result/_gate_train_dump/gate_scaler_n300_500_800_1000.npz --pattern 10000 --division_region 32 --division_point 64 --no_l1; done
<pattern B>
for ds_points in 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python AMA_SHAP_for_PointNeXt.py --ds_points $ds_points --ama --ama_mode mlp --mask_chunk 512 --min_mask_chunk 16 --gate_ckpt result/_gate_train_dump/gate_mlp_n300_400_500_600_700_800_900_1000.pth --gate_scaler result/_gate_train_dump/gate_scaler_n300_400_500_600_700_800_900_1000.npz --pattern 10000 --division_region 32 --division_point 64 --no_l1; done


[Region-SHAP順位に基づくラウンドロビンPoint-SHAP選択による均衡型点群ダウンサンプリング]
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 &&\
for ds_points in 100 200 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python RoundRobin_SHAP_for_PointNeXt.py --ds_points ${ds_points} --pattern 10000 --division_region 32--division_point 64 --cache_mode load --no_l1 --mask_chunk 512 --min_mask_chunk 16; done


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
[Regionのみ、Pointのみ]
for ds_points in 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python onlyRegion_SHAP_DS.py --ds_points $ds_points --pattern 10000 --division_region 32; done && for ds_points in 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python onlyRegion_SHAP_DS.py --ds_points $ds_points --pattern 10000 --division_region 64 ; done && for ds_points in 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python onlyPoint_SHAP_DS.py --ds_points $ds_points --division_point 32 --no_l1; done && for ds_points in 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python onlyPoint_SHAP_DS.py --ds_points $ds_points --division_point 64 --no_l1; done && for ds_points in 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python onlyPoint_SHAP_DS.py --ds_points $ds_points --division_point 32; done && for ds_points in 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python onlyPoint_SHAP_DS.py --ds_points $ds_points --division_point 64; done

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
[[[Region-SHAPをPoint-SHAPに比例分配してShapley定義を完全保持]]]
[espw]
for ds_points in 300 400 500 600 700 800 900 1000; do CUDA_VISIBLE_DEVICES=0 python espw_SHAP_DS_for_PointNeXt.py --ds_points $ds_points --pattern 10000 --division_region 64 --division_point 64 --alpha 0.001 --lambda_eps 0; done
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


1. classificationタスクをPointNeXtで実行
    <modelnet40>
    train:
        $ CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml

    test: (オリジナルModelNet40に対する)
        $ CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml model.encoder_args.width=32 mode=test --pretrained_path /workspace/PointNeXt/log/modelnet40ply2048/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT/checkpoint/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT_ckpt_best.pth

    ★test: (ダウンサンプリング点群に対する)
        入力データに関して：[/workspace/PointNeXt/cfgs/modelnet40ply2048/default.yaml]の、[data_dir]と[test:num_points]を変更する。

        $ CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml mode=test --pretrained_path /workspace/PointNeXt/log/modelnet40ply2048/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT/checkpoint/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT_ckpt_best.pth

    その後、以下のpythonコードにて実験結果を整理
        /workspace/PointNeXt/result/accuracyResult_for_csv.py
    /workspace/PointNeXt/result/formatted_data.txt のデータをExcelに記録。


    <scanobjectnn>
    $ CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/scanobjectnn/pointnext-s.yaml
    $ CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/scanobjectnn/pointnext-s.yaml  mode=test --pretrained_path ***(checkpoint/~.pth)



2. part segmentationタスクをPointNeXtで実行
    <ShapeNet>
    train:
    $ CUDA_VISIBLE_DEVICES=0 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointnext-s.yaml

    test:
    $ CUDA_VISIBLE_DEVICES=0 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointnext-s.yaml mode=test --pretrained_path /workspace/PointNeXt/log/shapenetpart/shapenetpart-train-pointnext-s-ngpus1-seed2621-20250612-074205-9QFombJZY6hbJFtx2VVHML/checkpoint/shapenetpart-train-pointnext-s-ngpus1-seed2621-20250612-074205-9QFombJZY6hbJFtx2VVHML_ckpt_best.pth



3. segmentationタスクをPointNeXtで実行
    <S3DIS>
    $ CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py --cfg cfgs/s3dis/pointnext-xl.yaml

    <ScanNet>
    $ CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py --cfg cfgs/scannet/pointnext-xl.yaml 
    RuntimeError： CUDAがメモリ不足です。Tried to allocate 512.00 MiB (GPU 0; 7.58 GiB total capacity; 5.32 GiB already allocated; 277.44 MiB free; 5.35 GiB reserved in total by PyTorch) もし予約メモリ>>割り当てメモリであれば、断片化を避けるためにmax_split_size_mbを設定してみてください。 Memory Management と PYTORCH_CUDA_ALLOC_CONF のドキュメントを参照してください。
        232, 235両方でGPUのメモリ不足のためsegmentationが実行できなかった。メモリ使用量を低減することを今後の予定とする。
    一旦、classificationタスクに対しての改善を検討する。

学習回数(epoch数)は以下のファイル内で指定。
/workspace/PointNeXt/cfgs/(dataset名)/default.yaml




エラー発生＆対処法について
1. segmentationタスクでGPUのメモリ不足

