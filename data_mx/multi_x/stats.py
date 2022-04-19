# golden summaries list
with open("tokTrunc_1024_utf/trainY.txt", "r", encoding="utf-8") as f:
    ref = [line.strip()[2:] for line in f]  # removing the dash in the beginning and making the list

print("No. data", len(ref))
print("Ref-avg-len", sum(map(len, [sent.split(' ') for sent in ref])) / len(ref))
print("Ref-max-len", len(max([sent.split(' ') for sent in ref], key=len)))
print("Ref-min-len", len(min([sent.split(' ') for sent in ref], key=len)), "\n")