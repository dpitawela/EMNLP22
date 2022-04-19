import gzip
import shutil
import os
import json
import re
from nltk.tokenize import ToktokTokenizer


def unzipFiles():
    for fname in ["train", "test", "val"]:
        if not os.path.exists(fname + ".json"):
            with gzip.open(fname + '.json.gz', 'rb') as f_in:
                with open(fname + '.json', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


def arrangeData():
    dir = 'arranged/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    for fname in ["train", "test", "val"]:  # "test", "val" json files
        with open("arranged/" + fname + "X.txt", "w", errors='ignore') as src, \
                open("arranged/" + fname + "Y.txt", "w", errors='ignore') as tgt:
            for i, dataItem in enumerate(json.load(open(fname + '.json'))):
                try:
                    # each record has an id and multiple sources per data item have the smae id
                    # in each trainX/valX/testX file, records are arranged as
                    # <id> - abstract \n <id> - source1 \n <id> source2 \n
                    src.write(str(i) + '-' + dataItem['abstract'] + '\n')
                    [src.write(str(i) + '-' + dataItem['ref_abstract'][ref]['abstract'] + '\n') for ref in
                     dataItem['ref_abstract']]
                    # writing golden summary to trainY/testY/valY
                    tgt.write(str(i) + '-' + dataItem['related_work'] + '\n')
                except:
                    print(fname + ' ' + str(i))


def determineLengths(docLens, lenQuota):
    finLen = []  # to store final document token lengths to extract
    currentLen = 0

    for doclen in docLens:  # iterating through lengths of each document
        if currentLen < lenQuota:
            if (lenQuota - currentLen) >= doclen:
                finLen.append(doclen)
                currentLen += doclen
            else:
                availableLen = lenQuota - currentLen
                finLen.append(availableLen)
                currentLen += availableLen

        else:
            finLen.append(0)
    return finLen


def tokenizeAndTruncate():
    lengthQuota = 500
    dir = 'tokTrunc/'
    dataPerSplit = {'train': 30369, 'test': 5093, 'val': 5066}  # 30369

    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    toktok = ToktokTokenizer()
    for fname in ["test"]:  # "train", "test", "val"
        with open("arranged/" + fname + "X.txt", "r", errors='ignore') as src, \
                open(dir + fname + "X.txt", "w", errors='ignore') as tokSrc:
            lines = src.readlines()
            lines = map(str.strip, lines)
            lines = [line for line in lines if len(line) > 10]  # as some lines are empty

            for i in range(dataPerSplit[fname]):
                documentsBelongTogether = [line for line in lines if line.startswith(str(i) + "-")]  # data item id
                # removing the prefix "<id>-" from each line
                documentsBelongTogether = [re.sub(r"^\d+-", "", doc, 0, re.MULTILINE) for doc in
                                           documentsBelongTogether]

                # tokenizing
                tokenizedDocs = [toktok.tokenize(doc) for doc in documentsBelongTogether]
                # determining number of tokens to take from each document to fill up the max length quota
                numTokensFromEachDoc = determineLengths([len(tokDoc) for tokDoc in tokenizedDocs], lengthQuota)
                # taking the determined number of tokens from each tokenized document (Truncating)
                tokenizedDocsTruncated = [" ".join(tokenizedDoc[:numTokensFromEachDoc[i]]) for i, tokenizedDoc in
                                          enumerate(tokenizedDocs)]

                # adding the special token between documents
                finalString = " story_separator_special_tag ".join(tokenizedDocsTruncated).lower()
                # replacing citations with @cite
                finalString = re.sub(r"@cite_*\d*", "@cite", finalString, 0, re.MULTILINE | re.IGNORECASE)

                # writing the tokenized summary
                tokSrc.write(finalString + '\n')
                # for logging
                if i % 1000 == 0:
                    print("Completed", fname, ":", i)

        with open("arranged/" + fname + "Y.txt", "r", errors='ignore') as tgt, \
                open(dir + fname + "Y.txt", "w", errors='ignore') as tokTgt:
            lines = tgt.readlines()
            lines = map(str.strip, lines)

            for i, summary in enumerate(lines):
                # removing the prefix "<id>-" from each line
                summary = re.sub(r"^\d+-", "", summary, 0, re.MULTILINE)
                # tokenizing and truncating
                tokenizedSummary = " ".join(toktok.tokenize(summary)[:lengthQuota]).lower()
                # replacing citations with @cite
                tokenizedSummary = re.sub(r"@cite_*\d*", "@cite", tokenizedSummary, 0, re.MULTILINE | re.IGNORECASE)
                # writing the tokenized golden summary
                tokTgt.write('- ' + tokenizedSummary + '\n')

                # for logging
                if i % 1000 == 0:
                    print("Completed", fname, ":", i)


if __name__ == "__main__":
    unzipFiles()
    arrangeData()
    tokenizeAndTruncate()
