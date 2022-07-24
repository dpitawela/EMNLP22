## Transformer

 - Acknowledgement ---> [https://github.com/Alex-Fabbri/Multi-News/tree/master/code/OpenNMT-py-baselines](https://github.com/Alex-Fabbri/Multi-News/tree/master/code/OpenNMT-py-baselines)
- Located in ---> OpenNMT_Baselines
- Change the input, output, model paths according to your directory structure

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

 - Acknowledgement ---> [https://github.com/Alex-Fabbri/Multi-News/tree/master/code/OpenNMT-py-baselines](https://github.com/Alex-Fabbri/Multi-News/tree/master/code/OpenNMT-py-baselines)
- Located in ---> OpenNMT_Baselines
- Change the input, output, model paths according to your directory structure

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


## Hierarchical Transformer

 - Acknowledgement ---> [https://github.com/nlpyang/hiersumm](https://github.com/nlpyang/hiersumm)
 - Located in ---> Hiersumm
 - The below codes are for sentence level HT. Change -trunc_src_nblock to 24 for paragraph level training along with a paragraph level dataset and increase the batch size according to the GPU capacity.
 - Change the model paths, data path, vocab path, log file path, result path according to your directory structure
 
**To Train**

    python $path"train_abstractive.py" -mode train -batch_size 128 -seed 666 -train_steps 100000 -save_checkpoint_steps 5000 
    -report_every 100 -trunc_src_nblock 250 -visible_gpus 0 -gpu_ranks 0 -accum_count 4 -dec_dropout 0.1 
    -enc_dropout 0.1 -label_smoothing 0.1 -accum_count 4 -inter_layers 6,7 -inter_heads 8 -hier -world_size 1 
    -data_path $path"input/hier" -vocab_path $path"vocab/vocab.model" -model_path $path"models/" -log_file $path"log/log.txt" 
    -trunc_tgt_ntoken 300
    
    #-train_from $path"models/model_step_50000.pt"

**To Inference**

    python $path"train_abstractive.py" -mode test -batch_size 32 -valid_batch_size 4096 -seed 666 -trunc_src_nblock 24 
    -visible_gpus 0 -gpu_ranks 0 -inter_layers 6,7 -inter_heads 8 -hier -max_wiki 100000
    -dataset test -alpha 0.4 -max_length 300 -data_path $path"input/hier" -vocab_path $path"vocab/vocab.model" 
    -model_path $path"models/" -log_file $path"log/log.txt" -trunc_tgt_ntoken 300 -enc_dropout 0 -beam_size 5 
    -test_from $path"models/model_step_50000.pt" -result_path $path"output/out" 
    
 ## HiMap
Acknowledgement --->    [https://github.com/Alex-Fabbri/Multi-News/tree/master/code](https://github.com/Alex-Fabbri/Multi-News/tree/master/code)
