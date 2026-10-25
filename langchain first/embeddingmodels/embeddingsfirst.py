from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model = 'something', dimensions= 32)

result = embeddings.embed_query("what is the date today")

print(str(result))

