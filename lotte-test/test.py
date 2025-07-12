# lotte/lifestyle/dev
# lotte/lifestyle/dev/forum
# lotte/lifestyle/dev/search
# lotte/lifestyle/test
# lotte/lifestyle/test/forum
# lotte/lifestyle/test/search

import ir_datasets

# dataset = ir_datasets.load('lotte/lifestyle/test/search')
# for i in dataset.qrels_iter():
#     print(i)

dataset = ir_datasets.load('lotte/lifestyle/test/search')
for i in dataset.queries_iter():
    print(i)

perfect_query = 529