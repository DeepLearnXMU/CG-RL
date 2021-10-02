#!/bin/bash
set -e

seed=${1:-0}
vocab="data/django/vocab.freq15.bin"
train_file="data/django/train.bin"
dev_file="data/django/dev.bin"
test_file="data/django/test.bin"
dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
ptrnet_hidden_dim=32
lr=0.001
lr_decay=0.5
beam_size=15
lstm='lstm'  # lstm
parser=parser_rl
lamda=1
weight_decay=1e-6
epsilon=0
model_name=model.rl.sup.django.lamda${lamda}.${lstm}.hidden${hidden_size}.embed${embed_size}.action${action_embed_size}.field${field_embed_size}.type${type_embed_size}.dropout${dropout}.lr${lr}.lr_decay${lr_decay}.beam_size${beam_size}.$(basename ${vocab}).$(basename ${train_file}).glorot.par_state_w_field_embe.seed${seed}

pre_model=saved_models/django/model.pre.django.iter16000.bin

python exp.py \
    --cuda \
    --seed ${seed} \
    --lamda $lamda \
    --mode train_rl \
    --parser $parser \
    --load_model $pre_model \
    --weight_decay ${weight_decay} \
    --epsilon ${epsilon} \
    --batch_size 10 \
    --asdl_file asdl/lang/py/py_asdl.txt \
    --transition_system python2 \
    --evaluator django_evaluator \
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
    --beam_size ${beam_size} \
    --log_every 50 \
    --save_to saved_models/django/${model_name} 2>&1 | tee -a logs/django/${model_name}.log

test_file="data/django/test.bin"
python exp.py \
    --cuda \
    --mode test \
    --parser $parser \
    --shuffle_mode $shuffle_mode \
    --load_model saved_models/django/${model_name}.bin \
    --beam_size 15 \
    --test_file ${test_file} \
    --save_decode_to decodes/django/$(basename $1).test.decode \
    --decode_max_time_step 100

