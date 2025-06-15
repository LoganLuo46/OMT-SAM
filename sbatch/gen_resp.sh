#!/bin/bash
#SBATCH --job-name=BiomedClip              # 你可以自定义任务名称
#SBATCH --account=p32870
#SBATCH --output=./BiomedClip.out            # 标准输出文件路径
#SBATCH --error=./BiomedClip.err             # 标准错误输出文件路径
#SBATCH --partition=gengpu                 # 指定 GPU 分区名称，通常是 a100
#SBATCH --nodes=1                        # 节点数量
#SBATCH --gres=gpu:a100:1                     # 请求 4 块 A100 GPU
#SBATCH --ntasks=1                       # 任务数量
#SBATCH --cpus-per-task=12               # 每个任务使用的 CPU 核数
#SBATCH --time=48:00:00                  # 最长运行时间 48 小时
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=limingluo2025@u.northwestern.edu 

# 启动你的训练脚本或任务
python train_one_gpu.py --use_clip True --clip_model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
