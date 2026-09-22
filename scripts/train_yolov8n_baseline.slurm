#!/bin/bash
#SBATCH --job-name=pavement-yolov8n
#SBATCH --partition=gpu02
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=23:00:00
#SBATCH --output=logs/yolov8n_%j.out
#SBATCH --error=logs/yolov8n_%j.err

set -e

source /home/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate pavement

cd "$HOME/Pavement_Crack_Detection"

export YOLO_CONFIG_DIR="$HOME/.config/Ultralytics"
mkdir -p "$YOLO_CONFIG_DIR"
mkdir -p logs

echo "========================================"
echo "Pavement Crack Detection - YOLOv8n"
echo "========================================"
echo "Hostname: $(hostname)"
echo "Date:     $(date)"
echo "Python:   $(python --version)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "========================================"

yolo detect train \
    data=datasets/pavement/data.yaml \
    model=models/yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=-1 \
    device=0 \
    workers=4 \
    project=runs/pavement \
    name=yolov8n_baseline \
    pretrained=true \
    seed=0 \
    deterministic=true \
    amp=true

echo "========================================"
echo "Training finished: $(date)"
echo "========================================"
