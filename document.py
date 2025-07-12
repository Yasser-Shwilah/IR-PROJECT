from collections import defaultdict

class Document:
    def __init__(self, doc_id, text, terms):
        self.termCnt = defaultdict(int)
        self.vec = defaultdict(float)
        self.uniqueTerms = set(terms)
        self.lenTerms = len(terms)
        self.doc_id = doc_id
        self.terms = terms
        self.text = text

        for term in self.uniqueTerms:
            self.termCnt[term] = self.terms.count(term)
    
    def insertToVec(self ,weight ,term):
        self.vec[term] = weight