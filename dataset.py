from nltk.stem.snowball import EnglishStemmer ,SpanishStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import ir_datasets

from inverted_index import InvertedIndex

class Datasets:
    def __init__(self):
        self.ID1 = InvertedIndex(
            ir_datasets.load('lotte/lifestyle/test/search'),
            " .!?,@/\#~:;'\"",
            stopwords.words('english'),
            EnglishStemmer(),
            WordNetLemmatizer()
        )

        self.ID2 = InvertedIndex(
            ir_datasets.load('wikir/es13k/test'),
            " .!?,@/\#~:;'\"",
            stopwords.words('spanish'),
            SpanishStemmer(),
            WordNetLemmatizer()
        )

    def query(self ,dataset ,query):
        if dataset == 'lotte':
            return self.ID1.lookup(query)
        elif dataset == 'wikir':
            return self.ID2.lookup(query)
        else:
            return None