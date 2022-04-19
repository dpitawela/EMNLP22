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
        with open("arranged/" + fname + "X.txt", "w", encoding="utf-8") as src, \
                open("arranged/" + fname + "Y.txt", "w", encoding="utf-8") as tgt:
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


def determineLengths(docLens, docContribution):
    finLen = []  # to store final document token lengths to extract
    # determining no. of tokens to be collected to fill up the quota
    toBeTaken = sum([(docContribution - docLen) for docLen in docLens if docLen < docContribution])
    if toBeTaken == 0:  # if no additional tokens to be taken by iterating
        return [docContribution for i in range(len(docLens))]

    for doclen in docLens:  # iterating through lengths of each document
        if toBeTaken > 0:  # if there are more tokens to be taken to fill up the quota
            balance = doclen - docContribution
            if balance > 0:  # checking the current document can contribute
                tok = toBeTaken - balance
                if tok <= 0:  # no more to be taken
                    finLen.append(docContribution + toBeTaken)
                    toBeTaken = 0
                else:  # more to be taken
                    finLen.append(docContribution + balance)
                    toBeTaken = tok
            else:  # if the current doc cannot contribute to fill up the quota
                finLen.append(doclen)
        else:  # if the quota is filled; only the required no. of tokens are taken
            finLen.append(min(doclen, docContribution))
    return finLen


def tokenizeAndTruncate():
    lengthQuota = 1024
    dir = 'tokTrunk_1024_nonutf_nosep/'
    dataPerSplit = {'train': 30369, 'test': 5093, 'val': 5066}  # 30369

    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    toktok = ToktokTokenizer()
    for fname in ["train", "test", "val"]:  # "train", "test", "val"
        with open("arranged/" + fname + "X.txt", "r", encoding="utf-8") as src, \
                open(dir + fname + "X.txt", "w", encoding="utf-8") as tokSrc:
            lines = src.readlines()
            lines = map(str.strip, lines)
            lines = [line for line in lines if len(line) > 10]  # as some lines are empty

            for i in range(dataPerSplit[fname]):
                documentsBelongTogether = [line for line in lines if line.startswith(str(i) + "-")]  # data item id
                # removing the prefix "<id>-" from each line
                documentsBelongTogether = [re.sub(r"^\d+-", "", doc, 0, re.MULTILINE) for doc in
                                           documentsBelongTogether]

                # no. of tokens to be taken from the sources
                tokensPerDocument = int(lengthQuota / len(documentsBelongTogether))
                # tokenizing
                tokenizedDocs = [toktok.tokenize(doc) for doc in documentsBelongTogether]
                # determining number of tokens to take from each document to fill up the max length quota
                numTokensFromEachDoc = determineLengths([len(tokDoc) for tokDoc in tokenizedDocs], tokensPerDocument)
                # taking the determined number of tokens from each tokenized document (Truncating)
                tokenizedDocsTruncated = [" ".join(tokenizedDoc[:numTokensFromEachDoc[i]]) for i, tokenizedDoc in
                                          enumerate(tokenizedDocs)]

                # adding the special token between documents
                # finalString = " story_separator_special_tag ".join(tokenizedDocsTruncated).lower()
                finalString = " ".join(tokenizedDocsTruncated).lower()  # to avoid adding the special token

                # replacing citations with @cite
                finalString = re.sub(r"@cite_*\d*", "@cite", finalString, 0, re.MULTILINE | re.IGNORECASE)

                # writing the tokenized summary
                tokSrc.write(finalString + '\n')
                # for logging
                if i % 1000 == 0:
                    print("Completed X", fname, ":", i)

        with open("arranged/" + fname + "Y.txt", "r") as tgt, \
                open(dir + fname + "Y.txt", "w") as tokTgt:
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
                    print("Completed Y", fname, ":", i)


if __name__ == "__main__":
    unzipFiles()
    # arrangeData()
    # tokenizeAndTruncate()
