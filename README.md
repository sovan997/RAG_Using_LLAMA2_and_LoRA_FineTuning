# 🦙 LLaMA 2 RAG App with LoRA Fine-Tuning

This project demonstrates a complete end-to-end pipeline for combining **LoRA-based fine-tuning of LLaMA 2** with **Retrieval-Augmented Generation (RAG)** for document-based question answering. It is designed to run entirely **locally** using a lightweight **Streamlit-based interface**.

---

## 📘 Project Overview

This project addresses the problem of building a powerful document-based question answering system by integrating the strengths of two state-of-the-art techniques: **LLaMA 2 fine-tuned with LoRA** and **RAG (Retrieval-Augmented Generation)**.

The core idea is to make a language model not only capable of following domain-specific instructions, but also grounded in real content (PDFs or text files) provided by the user. First, LLaMA 2 is fine-tuned using **Low-Rank Adaptation (LoRA)** on an instruction-following dataset to make it more responsive, efficient, and cost-effective to adapt. LoRA significantly reduces training cost and resource usage by injecting small trainable layers into the frozen base model.

Once fine-tuned, the adapter weights are saved separately. During inference, the system dynamically loads the base LLaMA 2 model and applies the LoRA adapter using the PEFT library. In parallel, uploaded documents are processed using **LlamaIndex**, which converts them into a vector store for semantic search. When the user submits a query, the system first retrieves relevant document chunks and then passes them along with the query to the language model to generate a grounded, context-aware answer.

The project provides a local **Streamlit web app interface** for users to upload PDF documents and ask natural language questions. The model is downloaded automatically from Hugging Face on the first run and cached locally. There is no dependency on Gradio or any hosted interfaces, making this solution lightweight, private, and portable.

---

## ⚙️ Key Components

- **LoRA Fine-Tuning**: Efficient training of LLaMA 2 by only updating a small number of parameters.
- **RAG Pipeline**: Retrieves relevant document chunks before generating answers using the fine-tuned model.
- **Streamlit App**: Allows interactive querying over uploaded PDFs.
- **Tokenizer & Model Caching**: Base model is downloaded once and reused locally.
- **Model Merging (Optional)**: Merge LoRA weights into base model to remove adapter dependency.

---

## 🧠 Workflow Summary

1. Fine-tune LLaMA 2 using LoRA on an instruction dataset.
2. Save LoRA adapter weights locally.
3. Load base model and apply adapter using PEFT.
4. Use LlamaIndex to index PDF content.
5. Ask questions via Streamlit UI.

---

## 📂 Folder Contents

- `llama2-guanaco-finetuned/`: Directory containing LoRA adapter weights
- `app.py`: Streamlit app logic
- `rag_engine.py`: Backend logic for model loading, indexing, and querying
- `utils.py`: Document upload and file-saving utility
- `requirements.txt`: Python dependencies
- `.env`: Environment file storing Hugging Face API key
- `README.md`: Documentation and usage guide

---
