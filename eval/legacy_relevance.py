import re
import json
import numpy as np
from nltk.util import ngrams

def gram_maker(s):
    s = s.lower()
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    tokens = [token for token in s.split(" ") if token != ""]
    output = list(ngrams(tokens, 2))
    return output

def chunk(in_string,num_chunks):
    chunk_size = len(in_string)//num_chunks
    if len(in_string) % num_chunks: chunk_size += 1
    iterator = iter(in_string)
    for _ in range(num_chunks):
        accumulator = list()
        for _ in range(chunk_size):
            try: accumulator.append(next(iterator))
            except StopIteration: break
        yield ''.join(accumulator)

# golden_summary_path = "../data_mx/multi_x/tokTrunc_1024_utf/testY.txt" # multiX
# golden_summary_path = "../data_mx/multi_x/tokTrunc_1024_utf_nosep/testY.txt" # multiX nosep

golden_summary_path = "../data_mx/multi_news/tokTrunc_1024_utf/testY.txt" #multi News
# golden_summary_path = "../data_mx/multi_news/tokTrunc_1024_utf_nosep/testY.txt" #multi News nosep

generated_summary_path = "../Results/M2 - Impact of special token/Multi N/Tran_ori/test.transformer_ori.out.min_length200"

# generated summaries
with open(generated_summary_path, "r", encoding='utf-8') as f:
    generated = [line.strip()[2:] for line in f]  # removing the dash in the beginning and making the list

# golden summaries
with open(golden_summary_path, "r", encoding='utf-8') as f:
    golden = [line.strip()[2:] for line in f]  # removing the dash in the beginning and making the list

fromFirstSegment = []
fromSecondSegment = []
fromThirdSegment = []

for source_topic, reference in zip(golden, generated):
    # Counting ngrams presence in various segments.
    one = 0
    two = 0
    three = 0
    counter_tot = 0

    a,b,c = list(chunk(source_topic.lower(),3))
    grams = gram_maker(reference)
    for elem in grams:
        if ' '.join(list(elem)) in a:
            one += 1
        elif ' '.join(list(elem)) in b:
            two += 1
        elif ' '.join(list(elem)) in c:
            three += 1
    counter_tot += len(grams)

    fromFirstSegment.append(one/counter_tot)
    fromSecondSegment.append(two/counter_tot)
    fromThirdSegment.append(three/counter_tot)

relevance = {'content_from_first_segment': np.mean(fromFirstSegment), 'content_from_second_segment': np.mean(fromSecondSegment), 'content_from_third_segment': np.mean(fromThirdSegment)}

json_name = 'oriT_mn.json'
# reading the results json
with open(json_name, 'r') as file:
    results_dict = json.load(file)

results_dict['relevance'] = relevance
print(results_dict)

# writing to the results json
with open(json_name, 'w') as file:
    json.dump(results_dict, file)