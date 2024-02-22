export CUDA_VISIBLE_DEVISES=2
MODEL_NAME=bigbird
DATASET_NAME=imdb
seed=0
for seed in 0
do
    # echo exp:$i
    python main.py \
    --epochs 1 \
    --train_batch_size 64 \
    --seed $seed \
    --device 2 \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --output_dir $MODEL_NAME$DATASET_NAME \
    --num_labels 2
done
