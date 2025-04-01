# rag_engine.py
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from utils import save_uploaded_files
from dotenv import load_dotenv

load_dotenv()
hf_token=os.getenv("HUGGINGFACE_API_KEY")

# === 1. Define prompts ===
system_prompt = """
You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the context provided.
"""
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

# === 2. Load tokenizer and base model ===
base_model_id = "meta-llama/Llama-2-7b-hf"  # Base model from Hugging Face (must be accessible with token)
adapter_path = "./llama2-guanaco-finetuned"     # Path to downloaded adapter folder

tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_auth_token=hf_token)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype="auto",
    use_auth_token=hf_token
)

# === 3. Apply the LoRA adapter ===
model = PeftModel.from_pretrained(base_model, adapter_path)

# === 4. Wrap model using LlamaIndex ===
llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    generate_kwargs={"max_new_tokens": 256, "temperature": 0.5},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
)

# === 5. Service context ===
service_context = ServiceContext.from_defaults(llm=llm)

# === 6. Document handling ===
def load_documents(uploaded_files):
    save_uploaded_files(uploaded_files, save_dir="data")
    return SimpleDirectoryReader("data").load_data()

# === 7. Build index ===
def create_retriever(documents):
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    return index.as_query_engine()

# === 8. Answer query ===
def answer_query(query, retriever):
    return retriever.query(query).response
