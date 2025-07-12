# wikir/es13k/test
# stopwords.words('spanish')

import ir_datasets

dataset = ir_datasets.load("wikir/es13k/test")
for i in dataset.qrels_iter():
    print(i)