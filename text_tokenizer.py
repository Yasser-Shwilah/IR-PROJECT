from nltk.tokenize import word_tokenize
import contractions

class TextTokenizer:
    def __init__(self ,stripChars ,stopwords ,stemmer ,lemmatizer):
        self.stripChars = stripChars
        self.lemmatizer = lemmatizer
        self.stopwords = stopwords
        self.stemmer = stemmer

    def tokenizeText(self ,text):
        text = contractions.fix(text.lower().strip(self.stripChars)).replace('`' ,"'").replace('"' ,"'")

        wordTokens = word_tokenize(text)

        tokens = []
        for word in wordTokens:
            if word not in self.stopwords:
                tokens.append(self.lemmatizer.lemmatize(self.stemmer.stem(word)))

        return tokens
