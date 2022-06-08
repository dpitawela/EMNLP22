import math
import nltk
import numpy as np
import json

class Redundancy():
    # for summary in generated:
    def calculateRedundancy(self, summary, n_gram):
        semantic_unit_counts = {}
        grams = nltk.ngrams(summary.split(), n_gram)  # splitting to ngrams

        # counting the frequency of semantic units
        for gram in grams:
            if gram in semantic_unit_counts:
                semantic_unit_counts[gram] += 1
            else:
                semantic_unit_counts[gram] = 1

        # getting the total number of semantic units
        n_semantic_units = semantic_unit_counts.keys().__len__()

        entropy = 0  # initialising entropy for current summary
        for unit, count in semantic_unit_counts.items():
            prob = (count / n_semantic_units)
            entropy += -(prob * math.log(prob, 2))
        return -entropy

    def evaluateBatch(self, generatedSummaries):
        n_gram = 1  # set the semantic unit length
        redundancy = []  # initialising redundancy for the whole test set

        for summary in generatedSummaries:
            # adding redundancy calculated for a single example into the total redundancy
            redundancy.append(self.calculateRedundancy(summary=summary, n_gram=n_gram))

        return {'redundancy': np.mean(redundancy)}


# generated = ['The core idea The core idea The core idea The core idea The core idea The core ideaThe core idea The core idea The core idea The core idea The core idea The core idea']
# red = Redundancy()
# print(red.evaluateBatch(generated))

# --------------------------------------------------------------------------------------------------------------
# # golden_summary_path = "../data_mx/multi_x/tokTrunc_1024_utf/testY.txt" # multiX
# # golden_summary_path = "../data_mx/multi_x/tokTrunc_1024_utf_nosep/testY.txt" # multiX nosep
# # golden_summary_path = "../data_mx/multi_news/tokTrunc_1024_utf/testY.txt" #multi News
# golden_summary_path = "../data_mx/multi_news/tokTrunc_1024_utf_nosep/testY.txt" #multi News nosep
#
# generated_summary_path = "../Results/M2 - Impact of special token/Multi N/Copy T_nosep/test.transformer.out.min_length200"
#
# # generated summaries
# with open(generated_summary_path, "r", encoding='utf-8') as f:
#     generated = [line.strip()[2:] for line in f]  # removing the dash in the beginning and making the list
#
# # golden summaries
# with open(golden_summary_path, "r", encoding='utf-8') as f:
#     golden = [line.strip()[2:] for line in f]  # removing the dash in the beginning and making the list
#
# json_name = 'copyT_mn_nosep.json'
# # reading the results json
# with open(json_name, 'r') as file:
#     results_dict = json.load(file)
#
# results_dict['redundancy'] = Redundancy().evaluateBatch(generatedSummaries=generated)
# print(results_dict)
#
# # writing to the results json
# with open(json_name, 'w') as file:
#     json.dump(results_dict, file)