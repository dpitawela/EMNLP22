import math
import nltk
import numpy as np

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

generated = ['The By definition entropy encompasses the notion of maximum coverage.']
# generated = ['The core idea The core idea The core idea The core idea The core idea The core idea']
# golden = ['The core idea The core idea The core idea The core idea The core idea The core idea']
golden = ['The By definition entropy encompasses the notion of maximum coverage.']

rel = Relevance()
print(rel.evaluateBatch(generated, golden))
