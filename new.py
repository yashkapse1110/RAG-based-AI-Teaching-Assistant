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

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    print(response)
    return response



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

prompt = f'''I am teaching Python Basic in my Python course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer only video the video number and string in pythonTime stamp in minute and no other text
'''
with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)

# for index, item in new_df.iterrows():
#     print(index,item["title"],item["text"],item["number"],item["start"],item["end"])

