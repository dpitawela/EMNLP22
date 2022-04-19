import os
import re

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
    dir = 'tokTrunc/'
    dataPerSplit = {'train': 44972, 'test': 5622, 'val': 5622}

    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    for fname in ["train", "test", "val"]:  # "train", "test", "val"
        with open("arranged/" + fname + "X.txt", "r", encoding="utf-8") as src, \
                open(dir + fname + "X.txt", "w", encoding="utf-8") as tokSrc:
            lines = src.readlines()
            lines = map(str.strip, lines)
            lines = [line for line in lines if len(line) > 10]

            for i in range(dataPerSplit[fname]):
                documentsBelongTogether = [doc.strip() for doc in lines[i].split("story_separator_special_tag")]
                # no. of tokens to be taken from the sources
                tokensPerDocument = int(lengthQuota / len(documentsBelongTogether))
                # splitting by spaces and eliminating empty strings
                tokenizedDocs = [' '.join(doc.split(' ')).split() for doc in documentsBelongTogether]

                # determining number of tokens to take from each document to fill up the max length quota
                numTokensFromEachDoc = determineLengths([len(tokDoc) for tokDoc in tokenizedDocs], tokensPerDocument)
                # taking the determined number of tokens from each tokenized document (Truncating)
                tokenizedDocsTruncated = [" ".join(tokenizedDoc[:numTokensFromEachDoc[i]]) for i, tokenizedDoc in
                                          enumerate(tokenizedDocs)]

                # adding the special token between documents
                finalString = " story_separator_special_tag ".join(tokenizedDocsTruncated).lower()
                # finalString = " ".join(tokenizedDocsTruncated).lower()  # to avoid adding the special token

                # replacing citations with @cite
                finalString = re.sub(r"@cite_*\d*", "@cite", finalString, 0, re.MULTILINE | re.IGNORECASE)

                # writing the tokenized summary
                tokSrc.write(finalString + '\n')
                # for logging
                if i % 1000 == 0:
                    print("Completed X", fname, ":", i)

        with open("arranged/" + fname + "Y.txt", "r", encoding="utf-8") as tgt, \
                open(dir + fname + "Y.txt", "w", encoding="utf-8") as tokTgt:
            lines = tgt.readlines()
            lines = map(str.strip, lines)

            for i, summary in enumerate(lines):
                # removing the prefix "<id>-" from each line
                summary = re.sub(r"^–\s+", "", summary, 0, re.MULTILINE)
                # tokenizing and truncating
                tokenizedSummary = " ".join(' '.join(summary.split(' ')).split()[:lengthQuota]).lower()
                # replacing citations with @cite
                tokenizedSummary = re.sub(r"@cite_*\d*", "@cite", tokenizedSummary, 0, re.MULTILINE | re.IGNORECASE)
                # writing the tokenized golden summary
                tokTgt.write('- ' + tokenizedSummary + '\n')

                # for logging
                if i % 1000 == 0:
                    print("Completed Y", fname, ":", i)


if __name__ == "__main__":
    pass
    # tokenizeAndTruncate()
