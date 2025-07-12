from collections import defaultdict
import math

from text_tokenizer import TextTokenizer
from document import Document

class InvertedIndex:
    def __init__(self, dataset, stripChars, stopwords=None ,stemmer=None, lemmatizer=None):
        self.documents = {}
        self.dataset = dataset

        self.idf = defaultdict(float)
        self.query_docs = defaultdict(set)
        self.tf = defaultdict(lambda: defaultdict(float))
        self.tf_idf = defaultdict(lambda: defaultdict(float))
        self.invertedIndex = defaultdict(lambda: defaultdict(bool))
        self.textTokenizer = TextTokenizer(stripChars ,set(stopwords) ,stemmer ,lemmatizer)

        self.sumRR = 0.0
        self.sumAvp = 0.0
        self.queryCnt = 0
        self.threshold = 0.3

        self.build()
        
    def build(self):
        print("Start Building Inverted Index:")

        n_docs = 0
        for doc in self.dataset.docs_iter():
            n_docs += 1
            document = self.documents[doc.doc_id] = Document(doc.doc_id, doc.text, self.textTokenizer.tokenizeText(doc.text))
            for term in document.uniqueTerms:
                self.invertedIndex[term][document.doc_id] = True
                self.tf[term][document.doc_id] = math.log(1.0 + document.termCnt[term] / document.lenTerms)

        items = self.invertedIndex.items()
        for term ,doc_ids in items:
            self.idf[term] = math.log(n_docs / len(doc_ids))

        items = self.tf.items()
        for term ,docFrequency in items:
            items2 = docFrequency.items()
            for doc_id ,tf in items2:
                self.tf_idf[term][doc_id] = tf * self.idf[term]
                self.documents[doc_id].insertToVec(self.tf_idf[term][doc_id] ,term)

        for i in self.dataset.qrels_iter():
            self.query_docs[i.query_id].add(i.doc_id)

        print("-----------------------------------")
        print('Inverted Index: Build Successfully.')
        print("-----------------------------------")

    def evaluateQuery(self ,irsResult ,query ,queryId = None):
        if queryId == None :
            queryId = 0
            for query_id ,text in self.dataset.queries_iter():
                if query == text:
                    queryId = query_id
                    break

        precisionSum = 0.0
        numRelevantRetrieved = 0
        numRetrieved = len(irsResult)
        numRelevant = len(self.query_docs[queryId])

        firstRelevantIsFound = False
        enumerateRes = enumerate(irsResult)
        for i ,res in enumerateRes:
            if res['doc_id'] in self.query_docs[queryId]:
                if not firstRelevantIsFound:
                    self.sumRR += 1.0 / (i + 1.0)
                    firstRelevantIsFound = True
                numRelevantRetrieved += 1
                precisionSum += numRelevantRetrieved / (i + 1)

        precision = 0
        if numRetrieved != 0:
            precision = numRelevantRetrieved / numRetrieved
        
        recall = 0
        if numRelevant != 0:
            recall = numRelevantRetrieved / numRelevant
        
        avp = 0
        if numRetrieved != 0:
            avp = precisionSum / numRetrieved
            
        self.queryCnt += 1
        self.sumAvp += avp
        mapVal = self.sumAvp / self.queryCnt
        mrr = self.sumRR / self.queryCnt

        return {
            "precision": precision,
            "recall": recall,
            "avp": avp,
            "map": mapVal,
            "mrr": mrr
        }

    def lookup(self ,inputQuery ,queryId = None):
        query = Document(-1, inputQuery , self.textTokenizer.tokenizeText(inputQuery))
        for term in query.uniqueTerms:
            tf = math.log(1.0 + query.termCnt[term] / query.lenTerms)
            tf_idf = tf * self.idf[term]
            query.insertToVec(tf_idf ,term)

        irsResult = []
        visited = set()
        sim = defaultdict(float)
        items = self.documents.values()
        for document in items:
            sumDocTF_IDF_2 = 0.0
            for term in document.uniqueTerms:
                sumDocTF_IDF_2 += self.tf_idf[term][document.doc_id] ** 2
            sumDocTF_IDF_2 = math.sqrt(sumDocTF_IDF_2)

            for term in query.uniqueTerms:
                if term in document.uniqueTerms:
                    sim[document.doc_id] += self.tf_idf[term][document.doc_id] / sumDocTF_IDF_2
                    if not document.doc_id in visited and sim[document.doc_id] > self.threshold:
                        visited.add(document.doc_id)
                        irsResult.append({
                            "doc_id": document.doc_id,
                            "text": document.text
                        })

        def sortFunction(doc):
            return sim[doc['doc_id']]

        irsResult.sort(reverse=True,key=sortFunction)

        return {
            "irsResult": irsResult,
            "evaluation": self.evaluateQuery(irsResult ,inputQuery ,queryId)
        }
    
    def runAll(self):
        print("run all")
        for query in self.dataset.queries_iter():
            print(query.query_id)
            self.lookup(query.text ,query.query_id)
            mapVal = self.sumAvp / self.queryCnt
            mrr = self.sumRR / self.queryCnt
            print("map = " ,mapVal)
            print("mrr = " ,mrr)