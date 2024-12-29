server=170
pretrain_dataset='voxceleb2'
# dataset
finetune_dataset='bigfive'
num_labels=5
# model
model_dir="pretrain_mutimae_dim64_512_patch4_160_a256_test"
output_model="pretrain_mutimae_dim_512_patch16_160_a256_val"
ckpt=49
# input
input_size=160
input_size_audio=256
sr=4
# parameter
lr=1e-3
epochs=50

# output directory
OUTPUT_DIR="./saved/model/finetuning/${finetune_dataset}/audio_visual/${pretrain_dataset}_${output_model}/checkpoint-${ckpt}/eval_lr_${lr}_epoch_${epochs}_size${input_size}_a${input_size_audio}_sr${sr}_server${server}"
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p $OUTPUT_DIR
fi
# path to split files
DATA_PATH="./saved/data/${finetune_dataset}/audio_visual/"
# path to pre-trained model
MODEL_PATH="./saved/model/pretraining/${pretrain_dataset}/audio_visual/${model_dir}/checkpoint-${ckpt}.pth"
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
    --master_port 13296 \
    test_run_class_finetuning_av.py \
    --model avit_dim512_patch16_160_a256 \
    --data_set ${finetune_dataset^^} \
    --nb_classes ${num_labels} \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 12 \
    --num_sample 1 \
    --input_size ${input_size} \
    --input_size_audio ${input_size_audio} \
    --short_side_size ${input_size} \
    --depth 16 \
    --save_ckpt_freq 1000 \
    --num_frames 16 \
    --sampling_rate ${sr} \
    --opt adamw \
    --lr ${lr} \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs ${epochs} \
    --dist_eval \
    --test_num_segment 1 \
    --test_num_crop 1 \
    --num_workers 16 \
    >${OUTPUT_DIR}/nohup.out 2>&1

echo "Done!"

