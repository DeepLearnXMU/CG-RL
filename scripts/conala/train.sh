#!/bin/bash
set -e

seed=${1:-1}
vocab="data/conala/vocab.src_freq3.code_freq3.bin"
train_file="data/conala/train.all_0.bin"
dev_file="data/conala/dev.bin"
test_file="data/conala/test.bin"
dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
lr=0.001
lr_decay=0.5
batch_size=10
max_epoch=80
beam_size=15
lstm='lstm'  # lstm
parser=parser_rl
lr_decay_after_epoch=10
lamda=1.0
weight_decay=1e-6
epsilon=0
model_name=conala.rl.${lstm}.lamda${lamda}.hidden${hidden_size}.embed${embed_size}.action${action_embed_size}.field${field_embed_size}.type${type_embed_size}.dr${dropout}.lr${lr}.lr_de${lr_decay}.lr_da${lr_decay_after_epoch}.beam${beam_size}.$(basename ${vocab}).$(basename ${train_file}).glorot.par_state.seed${seed}

pre_model=saved_models/conala/model.conala.pretrain.iter2180.bin

python -u exp.py \
    --cuda \
    --parser ${parser} \
    --lamda $lamda \
    --seed ${seed} \
    --mode train_rl \
    --load_model $pre_model \
    --weight_decay ${weight_decay} \
    --epsilon ${epsilon} \
    --batch_size ${batch_size} \
    --evaluator conala_evaluator \
    --asdl_file asdl/lang/py3/py3_asdl.simplified.txt \
    --transition_system python3 \
    --train_file ${train_file} \
    --dev_file ${dev_file} \
    --test_file ${test_file} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 5 \
    --glorot_init \
    --lr ${lr} \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --max_epoch ${max_epoch} \
    --beam_size ${beam_size} \
    --log_every 50 \
    --save_to saved_models/conala/${model_name} 2>&1 | tee logs/conala/${model_name}.log

test_file="data/conala/test.bin"
python exp.py \
    --cuda \
    --parser ${parser} \
    --mode test \
    --load_model saved_models/conala/${model_name}.bin \
    --beam_size 15 \
    --test_file ${test_file} \
    --evaluator conala_evaluator \
    --save_decode_to decodes/conala/$(basename $1).test.decode \
    --decode_max_time_step 100

