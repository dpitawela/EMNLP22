import collections
import struct
import torch
import io
import gc
import glob
# from tensorflow.core.example import example_pb2
import sentencepiece as spm

VOCAB_SIZE = 87902 # news:87902 # science:54874
dir = "../../data_mx/multi_news/tokTrunc_1024_utf_nosep/"
vocab_counter = collections.Counter()
forVocab = []
tf_example_str_lst = []

def createBinFiles():
    for fname in ["train", "test", "val"]:  # "train", "test", "val"
        buffer = io.BytesIO()
        with open(dir + fname + "X.txt", "r", encoding="utf-8") as src, \
                open(dir + fname + "Y.txt", "r", encoding="utf-8") as tgt:
                # open("input/" + fname + ".bin", "wb") as srctgt:

            srcLines = src.readlines()
            tgtLines = tgt.readlines()

            data = []

            for src, tgt in zip(srcLines, tgtLines):
                forVocab.append(src)
                forVocab.append(tgt)

                src = src.split('story_separator_special_tag ')
                data.append({'src':src, 'tgt':tgt})
            n = 2000
            fname = 'valid' if fname == 'val' else fname
            [torch.save(data[i:i + n], "input/" + "hier." + fname + "." + str(j+1) + ".pt") for j, i in enumerate(range(0, len(data), n))]


def createSentPiece():
    spm.SentencePieceTrainer.train(sentence_iterator=iter(forVocab), model_prefix='vocab/vocab', vocab_size=VOCAB_SIZE,
                                   user_defined_symbols=['story_separator_special_tag'])


createBinFiles()
createSentPiece()