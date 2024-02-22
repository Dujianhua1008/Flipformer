export CUDA_VISIBLE_DEVISES=2
DEVICE=2

# MODEL_NAME=flipformer
# DATASET_NAME=imdb
# seed=0
# for i in 1 2 3 4 5
# do
#     echo exp:$i
#     python main.py \
#     --epochs 1 \
#     --train_batch_size 64 \
#     --seed $seed \
#     --device $DEVICE \
#     --model_name $MODEL_NAME \
#     --dataset_name $DATASET_NAME \
#     --output_dir $MODEL_NAME$DATASET_NAME \
#     --use_wandb \
#     --num_labels 2
# done

# MODEL_NAME=fastformer
# DATASET_NAME=imdb
# seed=0
# for i in 1 2 3 4 5
# do
#     echo exp:$i
#     python main.py \
#     --epochs 1 \
#     --train_batch_size 64 \
#     --seed $seed \
#     --device $DEVICE \
#     --model_name $MODEL_NAME \
#     --dataset_name $DATASET_NAME \
#     --output_dir $MODEL_NAME$DATASET_NAME \
#     --use_wandb \
#     --num_labels 2
# done

# MODEL_NAME=bigbird
# DATASET_NAME=imdb
# seed=0
# for i in 1
# do
#     echo exp:$i
#     python main.py \
#     --epochs 1 \
#     --train_batch_size 64 \
#     --seed $seed \
#     --device $DEVICE \
#     --model_name $MODEL_NAME \
#     --dataset_name $DATASET_NAME \
#     --output_dir $MODEL_NAME$DATASET_NAME \
#     --use_wandb \
#     --num_labels 2
# done

# MODEL_NAME=longformer
# DATASET_NAME=imdb
# seed=0
# for i in 1 2 3 4 5
# do
#     echo exp:$i
#     python main.py \
#     --epochs 1 \
#     --train_batch_size 64 \
#     --seed $seed \
#     --device $DEVICE \
#     --model_name $MODEL_NAME \
#     --dataset_name $DATASET_NAME \
#     --output_dir $MODEL_NAME$DATASET_NAME \
#     --use_wandb \
#     --num_labels 2
# done


# MODEL_NAME=transformer
# DATASET_NAME=imdb
# seed=0
# for i in 1 2 3 4 5
# do
#     echo exp:$i
#     python main.py \
#     --epochs 1 \
#     --train_batch_size 64 \
#     --seed $seed \
#     --device $DEVICE \
#     --model_name $MODEL_NAME \
#     --dataset_name $DATASET_NAME \
#     --output_dir $MODEL_NAME$DATASET_NAME \
#     --use_wandb \
#     --num_labels 2
# done

# MODEL_NAME=aft-simple
# DATASET_NAME=imdb
# seed=0
# for i in 1 2 3 4 5
# do
#     echo exp:$i
#     python main.py \
#     --epochs 1 \
#     --train_batch_size 64 \
#     --seed $seed \
#     --device $DEVICE \
#     --model_name $MODEL_NAME \
#     --dataset_name $DATASET_NAME \
#     --output_dir $MODEL_NAME$DATASET_NAME \
#     --use_wandb \
#     --num_labels 2
# done

# MODEL_NAME=linformer
# DATASET_NAME=imdb
# seed=0
# for i in 1 2 3 4 5
# do
#     echo exp:$i
#     python main.py \
#     --epochs 1 \
#     --train_batch_size 64 \
#     --seed $seed \
#     --device $DEVICE \
#     --model_name $MODEL_NAME \
#     --dataset_name $DATASET_NAME \
#     --output_dir $MODEL_NAME$DATASET_NAME \
#     --use_wandb \
#     --num_labels 2
# done

# MODEL_NAME=performer
# DATASET_NAME=imdb
# seed=0
# for i in 1 2 3 4 5
# do
#     echo exp:$i
#     python main.py \
#     --epochs 1 \
#     --train_batch_size 64 \
#     --seed $seed \
#     --device $DEVICE \
#     --model_name $MODEL_NAME \
#     --dataset_name $DATASET_NAME \
#     --output_dir $MODEL_NAME$DATASET_NAME \
#     --use_wandb \
#     --num_labels 2
# done


DATASET_NAME=ag_news
# MODEL_NAME=bigbird
for MODEL_NAME in {aft-simple,bigbird,fastformer,flipformer,linformer,longformer,performer,transformer}
do 
    echo Model_Name:$MODEL_NAME
    seed=0

    for i in {1}
    do
        echo exp:$i
        python main.py \
        --epochs 1 \
        --train_batch_size 64 \
        --seed $seed \
        --device $DEVICE \
        --model_name $MODEL_NAME \
        --dataset_name $DATASET_NAME \
        --output_dir $MODEL_NAME$DATASET_NAME \
        --use_wandb \
        --num_labels 2
    done
done