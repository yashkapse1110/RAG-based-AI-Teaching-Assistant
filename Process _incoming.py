import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import requests


df = joblib.load("embeddings.joblib")

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding




incoming_query = input("Ask any Question on video : ")
question_embedding = create_embedding([incoming_query])[0]
# print(question_embedding)
similarity = cosine_similarity(np.vstack(df["embedding"]),[question_embedding]).flatten()
# print(similarity)
top_result = 3
max_indx = similarity.argsort()[::-1][0:top_result]
# print(max_indx)
new_df = df.loc[max_indx]
# print(new_df[["title","number","text"]])

for index, item in new_df.iterrows():
    print(index,item["title"],item["text"],item["number"],round(item["start"]/60),round(item["end"]/60))

