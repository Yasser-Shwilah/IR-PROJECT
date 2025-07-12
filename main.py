from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

from dataset import Datasets

app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000",
    "http://localhost:3000/",
    "localhost:3000/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

datasets = Datasets()

@app.get("/query")
def read_root(dataset: str ,query: str):
    return {
        "result" : datasets.query(dataset ,query)
    }
    