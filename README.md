# 🔎 IR-PROJECT – Search & Retrieval System using FastAPI

مشروع نظام استرجاع معلومات (Information Retrieval System) متكامل باستخدام Python وFastAPI، يدعم البحث ضمن مجموعتي بيانات `wikIR1k` و `lifestyle`، ويستخدم خوارزميات فعالة مثل:

- ✅ TF-IDF
- ✅ BM25
- ✅ Hybrid (دمج بين TF-IDF و BM25)

## ⚙️ الميزات الأساسية

- 🔤 تصحيح تلقائي للاستعلامات (Spelling Correction)
- 🔁 توسيع ذكي للاستعلام باستخدام WordNet
- 📊 تقييم جودة النتائج باستخدام Precision, Recall, MAP, MRR
- 🧠 عنقودة النتائج باستخدام KMeans
- 🕵️‍♂️ تحليل الكيانات والتواريخ داخل المستندات (NER + Date Extraction)
- 🧭 اقتراح استعلامات مشابهة
- 🔍 فهرسة عكسية لسرعة التصفية الأولية

---

## 📁 هيكل المجلدات

```
IRProject/
├── main.py                ← الكود الأساسي لـ FastAPI
├── lifestyle/
│   └── dev/
│       ├── collection.tsv
│       ├── questions.search.tsv
│       └── qas.search.jsonl
├── wikIR1k/
│   ├── documents.csv
│   └── test/
│       ├── queries.csv
│       └── BM25.qrels.csv
├── common_words           ← قائمة الكلمات الشائعة (stopwords)
└── frontend/              ← واجهة React (اختيارية)
```

---

## 🚀 تشغيل المشروع

### 1. تثبيت المتطلبات

```bash
pip install -r requirements.txt
```

محتوى `requirements.txt` المقترح:

```
fastapi
uvicorn
numpy
spacy
pandas
scikit-learn
nltk
autocorrect
rank_bm25
datefinder
```

ثم تحميل الموارد الإضافية:

```bash
python -m nltk.downloader punkt wordnet
python -m spacy download en_core_web_sm
```

---

### 2. تشغيل خادم FastAPI

```bash
uvicorn main:app --reload
```

السيرفر سيكون متاح على: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

### 3. تشغيل الواجهة الأمامية (React)

من داخل مجلد `frontend/`:

```bash
npm install
npm start
```

> تأكد من إضافة `localhost:3000` إلى إعدادات `CORS` في الكود الخلفي.

---

## ✅ الميزات المدعومة

| تصحيح الاستعلامات
| استرجاع باستخدام TF-IDF
| استرجاع باستخدام BM25
| استرجاع هجين Hybrid
| تقييم (Precision, Recall, MAP)  
| اقتراح استعلامات مشابهة
| تحليل الكيانات (NER + Dates)
| عنقودة المستندات (KMeans)
| فهرسة عكسية Inverted Index

---

## 📊 مثال استعلام (API)

```http
GET http://127.0.0.1:8000/?q=how to feed my cat&dataset=lifestyle&method=tfidf
```

### بارامترات:

- `q`: نص الاستعلام (مطلوب)
- `dataset`: اسم مجموعة البيانات (`lifestyle` أو `wikIR1k`)
- `method`: طريقة الترتيب (`tfidf`, `bm25`, `hybrid`)

---

## 🧪 مثال مخرجات JSON

```json
{
  "query_corrected": "how to feed my cat",
  "query_expanded": "how, feed, cat, nourish, care...",
  "ranking_method": "tfidf",
  "top_similar_docs": [...],
  "evaluation": {
    "Precision@10": 0.8,
    "Recall": 0.7,
    "MAP": 0.6,
    "MRR": 0.9
  }
}
```

---

## 📚 مجموعات البيانات المستخدمة

### 1. [wikIR1k](https://github.com/iai-group/WikIR)

- تحتوي على 1000 استعلام ومجموعة وثائق موسوعية من ويكيبيديا.
- مرفقة بتقييم BM25 لكل استعلام.

### 2. Lifestyle Dataset (مبنية على StackExchange - Pets)

- تتضمن أسئلة وأجوبة واقعية من منتديات الحيوانات الأليفة.
- بصيغ `.tsv` و `.jsonl`.

---
