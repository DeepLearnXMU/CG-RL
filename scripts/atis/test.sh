#!/bin/bash

model_name=checkpoint/atis

python exp.py \
   --cuda \
   --parser parser_rl \
   --mode test \
   --load_model $model_name \
   --beam_size 5 \
   --test_file data/atis/test.bin \
   --save_decode_to decodes/atis/${model_name}.test.decode \
   --decode_max_time_step 110

