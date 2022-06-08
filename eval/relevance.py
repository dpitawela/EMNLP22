import math
import nltk
import numpy as np
import json

class Relevance():
    def calculateRelevance(self, generatedSummary, goldenSummary, n_gram):
        semantic_unit_counts_generated = {}
        semantic_unit_counts_golden = {}

        grams_generated = nltk.ngrams(generatedSummary.split(), n_gram)  # splitting to ngrams
        grams_golden = nltk.ngrams(goldenSummary.split(), n_gram)  # splitting to ngrams

        # counting the frequency of semantic units in generated
        for gram in grams_generated:
            if gram in semantic_unit_counts_generated:
                semantic_unit_counts_generated[gram] += 1
            else:
                semantic_unit_counts_generated[gram] = 1

        # counting the frequency of semantic units in generated
        for gram in grams_golden:
            if gram in semantic_unit_counts_golden:
                semantic_unit_counts_golden[gram] += 1
            else:
                semantic_unit_counts_golden[gram] = 1

        # getting the total number of semantic units
        n_semantic_units_generated = semantic_unit_counts_generated.keys().__len__()
        n_semantic_units_golden = semantic_unit_counts_golden.keys().__len__()

        relevance = 0  # initialising relevance for the current summary
        for (unit_gen, count_gen) in semantic_unit_counts_generated.items():

            prob_gen = (count_gen / n_semantic_units_generated)
            prob_gold = ((semantic_unit_counts_golden[
                              unit_gen] if unit_gen in semantic_unit_counts_golden else 0) / n_semantic_units_golden)

            if prob_gen != 0 and prob_gold != 0:
                relevance += (prob_gen * math.log(prob_gold, 2))
            else:
                relevance += 0

        return relevance


    def evaluateBatch(self, generated, golden):
        n_gram = 1  # set the semantic unit length
        relevance = []  # initialising relevance for the whole test set
        for generatedSummary, goldenSummary in zip(generated, golden):
            # adding relevance calculated for a single example into the total relevance
            relevance.append(self.calculateRelevance(generatedSummary=generatedSummary, goldenSummary=goldenSummary,
                                            n_gram=n_gram))

        return {'relevance':  np.mean(relevance)}

# generated = ['The By definition', ' yo yo']
# generated = ['The core idea The core idea The core idea The core idea The core idea The core idea']
# golden = ['The core idea The core idea The core idea The core idea The core idea The core idea']
# golden = ['rock The hard']
# rel = Relevance()
# print(rel.evaluateBatch(generated, golden))

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
# results_dict['relevance'] = Relevance().evaluateBatch(generated=generated, golden=golden)
# print(results_dict)
#
# # writing to the results json
# with open(json_name, 'w') as file:
#     json.dump(results_dict, file)