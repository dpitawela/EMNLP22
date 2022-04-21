
## HiMap
Acknowledgement --->    [https://github.com/Alex-Fabbri/Multi-News/tree/master/code](https://github.com/Alex-Fabbri/Multi-News/tree/master/code)  


## Transformer
Acknowledgement ---> [https://github.com/Alex-Fabbri/Multi-News/tree/master/code](https://github.com/Alex-Fabbri/Multi-News/tree/master/code)
Located in ---> OpenNMT_Baselines
**To train:**

    python train.py -data/newser/OpenNMT-input/newser -word_vec_size 512 -rnn_size 512 -layers 4 -encoder_type transformer 
    -decoder_type transformer -position_encoding -warmup_steps 8000 -learning_rate 2 -decay_method noam -label_smoothing 0.1 
    -max_grad_norm 0 -dropout 0.2 -optim adam -adam_beta2 0.998 -param_init 0 -batch_type tokens -normalization tokens 
    -max_generator_batches 2 -accum_count 4 -share_embeddings -param_init_glorot -seed 777
    -world_size 1 -gpu_ranks 0 -batch_size 4096 -train_steps 20000 -save_checkpoint_steps 2000 
    -save_model models/newser-transformer-ori/newser
**To Inference**

    python translate.py -gpu 0 -batch_size 50 -beam_size 5 -model models/newser-transformer-ori/newser_step_20000.pt
    -output data/newser/test-output/test.transformer_ori -max_length 300  -verbose
    -stepwise_penalty -coverage_penalty summary -beta 5 -length_penalty wu -alpha 0.9 -block_ngram_repeat 3
    -ignore_when_blocking "." "</t>" "<t>" "story_separator_special_tag" -src data_mx/testX.txt 
    -min_length 110




## Copy-Transformer
Acknowledgement ---> [https://github.com/Alex-Fabbri/Multi-News/tree/master/code](https://github.com/Alex-Fabbri/Multi-News/tree/master/code)
Located in ---> OpenNMT_Baselines
**To Train**

    python train.py -save_model models/newser-transformer/newser -data data/newser/OpenNMT-input/newser -copy_attn
    -word_vec_size 512 -rnn_size 512 -layers 4 -encoder_type transformer -decoder_type transformer 
    -position_encoding -warmup_steps 8000 -learning_rate 2 -decay_method noam -label_smoothing 0.1
    -max_grad_norm 0 -dropout 0.2 -optim adam -adam_beta2 0.998 -param_init 0 -batch_type tokens
    -normalization tokens -max_generator_batches 2 -accum_count 4 -share_embeddings -param_init_glorot
    -seed 777 -world_size 1 -gpu_ranks 0 -batch_size 4096 -train_steps 20000 -save_checkpoint_steps 2000

**To Inference**

    python translate.py -gpu 0 -batch_size 50 -beam_size 5 -model models/newser-transformer/newser_step_20000.pt
    -output data/newser/test-output/test.transformer -max_length 300 -verbose
    -stepwise_penalty -coverage_penalty summary -beta 5 -length_penalty wu -alpha 0.9 -block_ngram_repeat 3
    -ignore_when_blocking "." "</t>" "<t>" "story_separator_special_tag" -src data_mx/testX.txt
    -min_length 110
