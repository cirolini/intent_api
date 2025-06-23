# app/config.py

import os

MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
DATA_DIR = os.getenv("DATA_DIR", "faiss_indices")
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.7"))
