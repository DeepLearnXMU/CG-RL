#!/bin/bash

model_name=checkpoint/geo

python exp.py \
    --cuda \
    --mode test \
    --parser parser_rl \
    --load_model $model_name \
    --beam_size 5 \
    --test_file data/geo/test.bin \
    --decode_max_time_step 110

