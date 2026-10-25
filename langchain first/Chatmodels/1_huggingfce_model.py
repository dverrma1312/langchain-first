from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id= ' ')

model = ChatHuggingFace(llm = llm)

result = model.invoke("what is the date today")

print(result)
