#!/bin/bash
seed=${1:-0}
vocab="vocab.freq2.bin"
train_file="train.bin"
dev_file="dev.bin"
dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=32
type_embed_size=32
lr_decay=0.5
beam_size=5
lstm='lstm'
parser='parser_rl'
ls=0.1
lamda=1
weight_decay=1e-5
epsilon=0.1
model_name=model.atis.sup.${lstm}.hidden${hidden_size}.embed${embed_size}.action${action_embed_size}.field${field_embed_size}.type${type_embed_size}.dropout${dropout}.lr_decay${lr_decay}.beam${beam_size}.${vocab}.${train_file}.glorot.with_par_info.no_copy.ls${ls}.seed${seed}

pre_model=saved_models/atis/model.atis.pretrain.iter4440.bin

python -u exp.py \
    --cuda \
    --seed ${seed} \
    --lamda ${lamda} \
    --parser ${parser} \
    --weight_decay ${weight_decay} \
    --epsilon ${epsilon} \
    --load_model ${pre_model}\
    --mode train_rl \
    --batch_size 10 \
    --asdl_file asdl/lang/lambda_dcs/lambda_asdl.txt \
    --transition_system lambda_dcs \
    --train_file data/atis/${train_file} \
    --dev_file data/atis/${dev_file} \
    --vocab data/atis/${vocab} \
    --lstm ${lstm} \
    --primitive_token_label_smoothing ${ls} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --hidden_size ${hidden_size} \
    --att_vec_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 5 \
    --glorot_init \
    --no_copy \
    --lr_decay ${lr_decay} \
    --beam_size ${beam_size} \
    --decode_max_time_step 110 \
    --log_every 50 \
    --save_to saved_models/atis/${model_name} 2>&1 | tee -a logs/atis/${model_name}.log

python exp.py \
    --cuda \
    --parser $parser \
    --mode test \
    --load_model saved_models/atis/${model_name}.bin \
    --beam_size 5 \
    --test_file data/atis/test.bin \
    --save_decode_to decodes/atis/${model_name}.test.decode \
    --decode_max_time_step 110


