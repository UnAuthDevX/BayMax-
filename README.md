# BayMax!
BayMax! is an AI-powered personal healthcare assistant developed by the Unauthorized Developers team. The purpose of this project is to create an intelligent assistant that can assist users with healthcare-related tasks, including answering medical queries, providing health advice, and offering personalized healthcare insights.

# Team - Unathorized Developers
1. Aravinda Lochan K
2. Dhakshin A V
3. Teepakraaj G
4. Dhanush R

# Files in the Repo:


## BayMax!.fig
This file contains the frontend design of the BayMax! application. It defines the user interface (UI) elements and layout that users will interact with, designed in Figma. The focus is on providing an intuitive and user-friendly interface for a seamless healthcare experience.

## BayMax_llm.ipynb
This notebook is responsible for training the large language model (LLM) used in the BayMax! system. It focuses on training a healthcare-specific LLM to ensure accurate responses and relevant medical insights. The LLM is likely fine-tuned on a dataset of medical documents or healthcare literature to provide personalized health assistance.

Used model for training: GPT2
```py
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
```

## Baymax_RAG.ipynb
This notebook covers the Retrieval-Augmented Generation (RAG) model training. The RAG model is designed to retrieve information from an external corpus (likely medical books or healthcare databases) and generate responses based on the retrieved data. This hybrid approach ensures the assistant can both generate coherent, contextually appropriate replies and provide fact-based medical information from the database.

Used model for training- thenlper/gte-large:
```py
# model downloaded from  (https://huggingface.co/) then trained the pre-trained model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
Settings.embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large", device=device)
```

