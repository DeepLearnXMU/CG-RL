#!/bin/bash
set -e

seed=${1:-0}
vocab="data/geo/vocab.freq2.bin"
train_file="data/geo/train.bin"
dev_file="data/geo/dev.bin"
dropout=0.5
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=32
type_embed_size=32
lr_decay=0.985
lr_decay_after_epoch=20
max_epoch=30
patience=1000   # disable patience since we don't have dev set
beam_size=5
batch_size=10
lr=0.0025
ls=0.1
lstm='lstm'
model_name=model.geo.pretrain

echo "**** Writing results to logs/geo/${model_name}.log ****"
mkdir -p logs/geo
echo commit hash: `git rev-parse HEAD` > logs/geo/${model_name}.log

python -u exp.py \
    --cuda \
    --seed ${seed} \
    --mode pretrain \
    --save_all_models \
    --max_epoch 10 \
    --batch_size ${batch_size} \
    --asdl_file asdl/lang/lambda_dcs/lambda_asdl.txt \
    --transition_system lambda_dcs \
    --train_file ${train_file} \
    --dev_file ${dev_file} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --primitive_token_label_smoothing ${ls} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --no_parent_field_embed \
    --no_parent_state \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience ${patience} \
    --max_epoch ${max_epoch} \
    --lr ${lr} \
    --no_copy \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --decay_lr_every_epoch \
    --glorot_init \
    --beam_size ${beam_size} \
    --decode_max_time_step 110 \
    --log_every 50 \
    --save_to saved_models/geo/${model_name} 2>&1 | tee -a logs/geo/${model_name}.log
