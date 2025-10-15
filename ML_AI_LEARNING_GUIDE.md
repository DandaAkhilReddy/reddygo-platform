# 🚀 ReddyGo ML/AI Learning Platform

## Complete Infrastructure for Learning Modern AI/ML

This project contains a **production-ready ML/AI learning environment** where you master every key technology in one codebase.

## 🎯 Technologies Mastered

- ✅ **PyTorch** - LSTM for time series prediction
- ✅ **TensorFlow** - Collaborative filtering recommender
- ✅ **LangChain** - Multi-agent orchestration
- ✅ **Qdrant** - Vector database for semantic search
- ✅ **LLaMA 3.1** - Local LLM via Ollama (FREE!)
- ✅ **Embeddings** - Sentence transformers
- ✅ **RAG** - Retrieval Augmented Generation
- ✅ **MLOps** - Model serving & inference

**Value: $8,000+ in courses!** 🎓

---

## 🏗️ What We Built

### 1. PyTorch LSTM (`backend/ml/pytorch/performance_predictor.py`)
- Predicts future workout performance
- 2-layer LSTM → Dense layers → 3 outputs
- Complete training & inference pipeline

### 2. TensorFlow Recommender (`backend/ml/tensorflow/recommender.py`)
- Collaborative filtering for challenges
- User + Item embeddings → MLP
- Similarity search in embedding space

### 3. Qdrant Vector DB (`backend/vector/qdrant_client.py`)
- Semantic search for workouts, memories, exercises
- 3 collections with metadata filtering
- Sentence transformer embeddings (384-dim)

### 4. LangChain Agent (`backend/llm/langchain_agent.py`)
- Multi-tool orchestration (5 tools)
- Automatic tool selection
- Conversation memory

### 5. Ollama Client (`backend/llm/ollama_client.py`)
- Local LLaMA 3.1 integration
- Zero API cost
- Chat & completion interfaces

---

## 🚀 Quick Start

bash
# Install dependencies
pip install -r backend/requirements.txt

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Install Ollama (optional)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b

# Test PyTorch
python backend/ml/pytorch/performance_predictor.py

# Test Vector DB
python backend/vector/qdrant_client.py

# Test LangChain
export OPENAI_API_KEY="your_key"
python backend/llm/langchain_agent.py


---

## 📚 Learning Path

**Week 1:** PyTorch + Vector DB
**Week 2:** TensorFlow + LangChain
**Week 3:** Local LLMs + Integration
**Week 4:** API + Frontend + Deploy

---

## 💡 Key Concepts

**Embeddings:** Text → Numerical vectors for similarity

**RAG:** Retrieve context → Augment prompt → Generate answer

**Collaborative Filtering:** Learn user/item embeddings → Recommend

---

## 💰 Cost Optimization

| LLM | Cost/1M tokens | Use Case |
|-----|----------------|----------|
| LLaMA (local) | $0.00 | 80% of queries |
| GPT-4o-mini | $0.15 | General use |
| GPT-4 | $5.00 | Complex reasoning |

**Savings: 90%+ with local LLaMA!**

---

*See source files for detailed examples and explanations.*
