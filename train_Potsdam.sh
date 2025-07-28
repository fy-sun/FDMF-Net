export CUBLAS_WORKSPACE_CONFIG=:4096:8
source activate ~/anaconda3/envs/FDMF-Net

export Dataset=Potsdam

echo "start training"
echo "Training Dataset: $Dataset"
echo "Seed: 42"
echo "Using GPUs: $1"

python main.py cuda_visible_devices=[$1] training_dataset=$Dataset seed=42 training.learning_rate=0.0001 training.alpha=0.01 training.beta=0.05 training.gamma=0.5