from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

embeddings = OpenAIEmbeddings(model = 'something', dimensions= 300)

document = ['it is something']
query = 'what is going on'

data = embeddings.embed_document(document)
query = embeddings.embed_query(query)

data = cosine_similarity([data], query)[0]

index, final = sorted(list(enumerate(data), key = lambda x:x[1]))[-1]


print(query)
print(document[index])
print("similarity score is:", final)