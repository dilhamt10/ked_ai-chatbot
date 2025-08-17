# 🍽️ Restaurant Chatbot (LangChain & on Hugging Face)

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-00A67E?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black)

An AI-powered restaurant chatbot that:
- Answers menu questions using Retrieval-Augmented Generation (RAG)
- Handles 10 food (`makanan`) and 10 drink (`minuman`) items
- Deploys easily to Hugging Face Spaces

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/your-username/your-space-name)
[![Open in GitHub Codespaces](https://img.shields.io/badge/GitHub-Codespaces-181717?style=flat&logo=github)](https://codespaces.new/your-repo-link)

## ✨ Features

- **Instant Menu Lookup**  
  "Nasi Goreng" → Shows price & description immediately
- **FAQ Retrieval**  
  Answers "opening hours?", "payment methods?" from knowledge base (under development)
- **Multilingual Support**  
  Works in English and Bahasa Indonesia
- **Low-Cost Deployment**  
  Uses free Hugging Face models

## 🛠️ Tech Stack

| Component           | Technology                          |
|---------------------|-------------------------------------|
| Framework           | Streamlit + LangChain               |
| LLM                 | Hugging Face (flan-t5-small)        |
| Vector Database     | FAISS                               |
| Embeddings          | sentence-transformers/all-MiniLM-L6 |
| Deployment          | Hugging Face Spaces                 |

## 🚀 Quick Start

### Check on HuggingFace🤗
```bash
https://huggingface.co/spaces/dilhamt10/kedai-chatbot/
