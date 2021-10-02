#!/bin/bash

model=checkpoint/conala
test_file="data/conala/test.bin"

python exp.py \
    --cuda \
    --mode test \
    --parser parser_rl \
    --load_model $model \
    --beam_size 15 \
    --test_file ${test_file} \
    --evaluator conala_evaluator \
    --save_decode_to decodes/conala/$(basename $1).test.decode \
    --decode_max_time_step 100

