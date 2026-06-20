import os
from langchain_openai import AzureChatOpenAI

os.environ["AZURE_OPENAI_API_KEY"]  = "your-api-key-here"
os.environ["AZURE_OPENAI_ENDPOINT"] = "your-endpoint-here"

# Temperature 0 — used for all structured reasoning calls:
# routing, planning, arg validation, RAG checks
llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    openai_api_version="2025-04-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=0.0,
)

# Temperature 1 — used for all free-form text generation:
# final responses, handlers, explanations
response_llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    openai_api_version="2025-04-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=1.0,
)