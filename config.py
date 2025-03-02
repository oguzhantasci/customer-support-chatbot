import os
import json
import pandas as pd
import datetime
import logging

# Retrieve OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OpenAI API key not found! Add it to a .env file.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # Set the key for OpenAI models

# Load dataset
dataset_path = "Customer-Churn-Records.csv"
df = pd.read_csv(dataset_path)

# Set up logging
logging.basicConfig(filename="banking_chatbot.log", level=logging.INFO, format="%(asctime)s - %(message)s")
