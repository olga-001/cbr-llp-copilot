import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

api_key = os.getenv("LANGSMITH_API_KEY")
project = os.getenv("LANGSMITH_PROJECT")

print("LANGSMITH_API_KEY loaded:", bool(api_key))
print("LANGSMITH_PROJECT:", project)

client = Client()

datasets = list(client.list_datasets(limit=1))

print("LangSmith API connection: OK")
print("Datasets found:", len(datasets))