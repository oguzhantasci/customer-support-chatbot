import os
import json
import pandas as pd
import logging

# Retrieve environment variables
LANGSMITH_TRACING = "true"
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_API_KEY = "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"
LANGSMITH_PROJECT = "customer-support-chatbot"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ensure critical environment variables are set
if not OPENAI_API_KEY:
    raise ValueError("❌ OpenAI API key not found! Make sure to set it as an environment variable.")

# Set environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGSMITH_TRACING"] = LANGSMITH_TRACING
os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT

# Load dataset
dataset_path = "Customer-Churn-Records.csv"
df = pd.read_csv(dataset_path)

# Set up logging
logging.basicConfig(filename="banking_chatbot.log", level=logging.INFO, format="%(asctime)s - %(message)s")
