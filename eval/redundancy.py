import math
import nltk
import numpy as np

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


generated = ['The core idea The core idea The core idea The core idea The core idea The core idea']

red = Redundancy()
print(red.evaluateBatch(generated))