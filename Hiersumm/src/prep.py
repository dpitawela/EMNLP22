import collections
import struct
import torch
import io
import gc
import glob
from tensorflow.core.example import example_pb2
import sentencepiece as spm

VOCAB_SIZE = 54874
dir = "../../data_mx/multi_x/tokTrunc_1024_utf/"
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
                srcLine = src.encode(encoding='utf-8')
                tgtLine = tgt.encode(encoding='utf-8')

                forVocab.append(srcLine)
                forVocab.append(tgtLine)

                # tf_example = example_pb2.Example()
                # tf_example.features.feature['src'].bytes_list.value.extend([srcLine])  # (source) abstracts of papaers
                # tf_example.features.feature['tgt'].bytes_list.value.extend([tgtLine])  # (target) related work section
                #
                # tf_example_str = tf_example.SerializeToString()
                # str_len = len(tf_example_str)
                # srctgt.write(struct.pack('q', str_len))
                # srctgt.write(struct.pack('%ds' % str_len, tf_example_str))
                #
                # # # creating PT files
                # tf_example_str_lst.append(tf_example_str)
                # torch.save(tf_example_str, buffer)
                # torch.save(tf_example_str_lst, "input/" + "xx." +fname + ".pt")

                data.append({'src':srcLine, 'tgt':tgtLine})
            n = 2000
            fname = 'valid' if fname == 'val' else fname
            [torch.save(data[i:i + n], "input/" + "hier." + fname + "." + str(j+1) + ".pt") for j, i in enumerate(range(0, len(data), n))]


def createSentPiece():
    spm.SentencePieceTrainer.train(sentence_iterator=iter(forVocab), model_prefix='vocab/vocab', vocab_size=VOCAB_SIZE,
                                   user_defined_symbols=['story_separator_special_tag'])


def createPTFiles():
    text = ["hello hello hello", "hellohellohello hello"]
    torch.save(text, "input/train2.pt")

    # buffer = io.BytesIO()
    # torch.save("hello", buffer)
    # torch.save(buffer, "input/train.pt")


createBinFiles()
createSentPiece()
# createPTFiles()

#             # creating the vocabulary
#             art_tokens = src.split(' ')
#             abs_tokens = tgt.split(' ')
#             tokens = art_tokens + abs_tokens
#             tokens = [t.strip() for t in tokens]  # strip
#             tokens = [t for t in tokens if t != ""]  # remove empty
#             vocab_counter.update(tokens)
#
# print("Writing vocab file...")
# with open("vocab", 'w', encoding='utf-8') as writer:
#     for word, count in vocab_counter.most_common(VOCAB_SIZE):
#         writer.write(word + ' ' + str(count) + '\n')
