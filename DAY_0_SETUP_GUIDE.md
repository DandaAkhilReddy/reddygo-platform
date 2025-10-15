# 🚀 Day 0 Setup Guide - ML/AI Bootcamp

**Complete this BEFORE starting Day 1** (Estimated time: 2 hours)

This guide will set up your complete ML/AI development environment for the 10-day intensive bootcamp.

---

## ✅ Prerequisites

- Python 3.11+ installed
- Git installed
- Docker Desktop installed (for Qdrant)
- 16GB+ RAM recommended
- GPU optional (CPU training will work but be slower)

---

## 📋 Step 1: Install Backend ML Dependencies (30 minutes)

### Install Python Dependencies

```bash
cd C:/Users/akhil/reddygo-platform/backend
pip install -r requirements.txt
```

This will install:
- **PyTorch 2.1.2** - Deep learning framework
- **TensorFlow 2.15.0** - Production ML framework
- **LangChain 0.1.0** - AI agent orchestration
- **Transformers 4.36.2** - Hugging Face models
- **Sentence Transformers** - Text embeddings
- **Qdrant Client** - Vector database
- **Ollama** - Local LLM client
- **scikit-learn, pandas, numpy** - ML utilities
- **MLflow** - Experiment tracking

### Verify Installation

```bash
cd C:/Users/akhil/reddygo-platform/backend
python verify_ml_setup.py
```

**Expected Output:**
```
🚀 ReddyGo ML/AI Setup Verification
======================================================================

✓ Python 3.11.x
✓ PyTorch 2.1.2 (CPU or CUDA)
✓ TensorFlow 2.15.0 (CPU or GPU)
✓ LangChain 0.1.0
✓ Transformers 4.36.2
✓ Sentence Transformers 2.2.2
✓ Qdrant client installed (server not running yet)
✓ Ollama client installed (server not running yet)
✓ scikit-learn 1.3.2
✓ pandas 2.1.4
✓ NumPy 1.26.3
✓ MLflow 2.9.2
✓ OpenAI SDK 1.33.0
✓ Mem0 AI 0.1.5

📊 Summary: 14/14 checks passed
```

---

## 📂 Step 2: Verify ML Directory Structure (5 minutes)

Your directory structure should now look like this:

```
reddygo-platform/
├── backend/
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── custom_llm/                 # Day 1-2: Custom transformer
│   │   │   ├── __init__.py
│   │   │   ├── transformer.py          # Self-attention, multi-head attention
│   │   │   ├── model.py                # GPT model
│   │   │   ├── tokenizer.py            # Character tokenizer
│   │   │   └── train.py                # Training loop
│   │   ├── finetuned_llama/            # Day 3: Fine-tuned LLaMA
│   │   ├── pytorch/                    # Day 5: PyTorch models
│   │   ├── tensorflow/                 # Day 6: TensorFlow models
│   │   └── inference.py                # Unified inference API
│   ├── vector/                         # Day 4: Vector database
│   ├── llm/                            # Day 7-8: LangChain agents
│   └── notebooks/                      # Jupyter notebooks
```

### Verify Directory Creation

```bash
cd C:/Users/akhil/reddygo-platform/backend
ls -la ml/
ls -la ml/custom_llm/
```

---

## 🐳 Step 3: Set Up Qdrant Vector Database (15 minutes)

### Option 1: Docker (Recommended)

```bash
# Start Qdrant with Docker
docker run -d -p 6333:6333 -p 6334:6334 \
  -v C:/Users/akhil/reddygo-platform/qdrant_storage:/qdrant/storage \
  --name reddygo-qdrant \
  qdrant/qdrant
```

### Option 2: Windows Docker Desktop

1. Open Docker Desktop
2. Go to "Images" tab
3. Search for "qdrant/qdrant"
4. Pull the latest image
5. Run with ports: 6333:6333, 6334:6334
6. Set volume: `C:/Users/akhil/reddygo-platform/qdrant_storage:/qdrant/storage`

### Verify Qdrant is Running

```bash
# Test connection
curl http://localhost:6333/collections

# Or visit in browser
http://localhost:6333/dashboard
```

**Expected Response:**
```json
{
  "collections": []
}
```

---

## 🦙 Step 4: Install Ollama and Download LLaMA (30 minutes)

### Install Ollama

1. Visit: https://ollama.ai/download
2. Download Ollama for Windows
3. Run installer
4. Open PowerShell and verify:

```bash
ollama --version
```

### Download LLaMA 3.1 Model

```bash
# Download LLaMA 3.1 8B (Recommended - 4.7GB)
ollama pull llama3.1:8b

# OR download LLaMA 3.1 70B (if you have 64GB+ RAM - 40GB download)
# ollama pull llama3.1:70b
```

### Verify LLaMA Installation

```bash
# List installed models
ollama list

# Test LLaMA
ollama run llama3.1:8b "What is running?"
```

**Expected Response:**
```
Running is a form of aerobic exercise involving moving at a faster pace than
walking, where both feet may be off the ground at the same time...
```

---

## 🧪 Step 5: Test Day 1 Components (30 minutes)

### Test Transformer Components

```bash
cd C:/Users/akhil/reddygo-platform/backend
python -m ml.custom_llm.transformer
```

**Expected Output:**
```
Testing Transformer Components...

1. Testing SelfAttention...
   Input shape: torch.Size([2, 10, 128])
   Output shape: torch.Size([2, 10, 128])
   ✓ SelfAttention working!

2. Testing MultiHeadAttention...
   Input shape: torch.Size([2, 10, 128])
   Output shape: torch.Size([2, 10, 128])
   ✓ MultiHeadAttention working!

3. Testing FeedForward...
   Input shape: torch.Size([2, 10, 128])
   Output shape: torch.Size([2, 10, 128])
   ✓ FeedForward working!

4. Testing TransformerBlock...
   Input shape: torch.Size([2, 10, 128])
   Output shape: torch.Size([2, 10, 128])
   ✓ TransformerBlock working!

✅ All components working correctly!
```

### Test GPT Model

```bash
python -m ml.custom_llm.model
```

**Expected Output:**
```
Testing GPT Model...

✓ Model created
  Parameters: 5,234,432

✓ Testing forward pass...
  Input shape: torch.Size([2, 32])
  Logits shape: torch.Size([2, 32, 5000])
  Loss: 8.5172

✓ Testing generation...
  Prompt shape: torch.Size([1, 10])
  Generated shape: torch.Size([1, 30])
  Generated 20 new tokens

✅ Model working correctly!
```

### Test Tokenizer

```bash
python -m ml.custom_llm.tokenizer
```

**Expected Output:**
```
Testing Character Tokenizer...

✓ Created fitness corpus: 30 texts

✓ Vocabulary built: 89 tokens
  Special tokens: ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
  Character tokens: 85

Original text: 'Running is fun!'
Encoded: [2, 52, 75, 70, 70, 65, 70, 63, 4, 65, 73, 4, 64, 75, 70, 3]
Decoded: 'Running is fun!'

✅ Round-trip encoding/decoding successful!
```

---

## 📚 Step 6: Review Learning Materials (15 minutes)

### Day 1 Learning Resources

1. **Watch Video** (1 hour - DO THIS TONIGHT):
   - "Let's build GPT: from scratch, in code, spelled out." by Andrej Karpathy
   - Link: https://www.youtube.com/watch?v=kCc8FmEb1nY

2. **Read Documentation**:
   - Open `ML_AI_LEARNING_GUIDE.md` in your project
   - Review the "Week 1: Transformer from Scratch" section

3. **Code Review**:
   - Skim through `backend/ml/custom_llm/transformer.py` - understand the TODO sections
   - Skim through `backend/ml/custom_llm/model.py` - see the complete architecture
   - Skim through `backend/ml/custom_llm/train.py` - see the training loop

---

## ✅ Step 7: Final Verification (5 minutes)

Run the complete setup verification again:

```bash
cd C:/Users/akhil/reddygo-platform/backend
python verify_ml_setup.py
```

**All 14 checks should pass:**
- ✓ Python 3.11+
- ✓ PyTorch
- ✓ TensorFlow
- ✓ LangChain
- ✓ Transformers
- ✓ Sentence Transformers
- ✓ Qdrant (server running)
- ✓ Ollama (server running + LLaMA downloaded)
- ✓ scikit-learn
- ✓ pandas
- ✓ NumPy
- ✓ MLflow
- ✓ OpenAI SDK
- ✓ Mem0 AI

---

## 🎯 Day 1 Preview - Tomorrow's Goals

**Day 1: Build Transformer from Scratch (10 hours)**

### Morning (4 hours):
1. Watch Karpathy video (if not watched tonight)
2. Implement self-attention mechanism from scratch
3. Build multi-head attention
4. Complete transformer block

### Afternoon (4 hours):
5. Build complete GPT model
6. Implement custom tokenizer
7. Set up training loop
8. Train on fitness corpus

### Evening (2 hours):
9. Deploy via FastAPI endpoint
10. Test text generation
11. Document learnings

---

## 🔧 Troubleshooting

### PyTorch Installation Issues

**Error:** `No module named 'torch'`

**Fix:**
```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

# OR for CUDA (if you have NVIDIA GPU)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

### Qdrant Connection Issues

**Error:** `Cannot connect to Qdrant at localhost:6333`

**Fix:**
```bash
# Check if Docker is running
docker ps

# Restart Qdrant
docker restart reddygo-qdrant

# Check logs
docker logs reddygo-qdrant
```

### Ollama Issues

**Error:** `Ollama not found`

**Fix:**
1. Restart your terminal/PowerShell
2. Verify PATH includes Ollama
3. Reinstall from https://ollama.ai/download

**Error:** `Model not found`

**Fix:**
```bash
# Re-download model
ollama pull llama3.1:8b
```

### Memory Issues

**Error:** `CUDA out of memory` or `Killed (OOM)`

**Fix:**
- Use smaller batch size in training (reduce BATCH_SIZE in train.py)
- Use smaller model (already using small_gpt)
- Close other applications
- Use CPU instead of GPU (edit DEVICE in train.py)

---

## 📊 Resource Requirements

### Disk Space:
- Python dependencies: ~5 GB
- LLaMA 3.1 8B: ~5 GB
- Qdrant + data: ~1 GB
- Training checkpoints: ~500 MB
- **Total: ~12 GB**

### RAM:
- Minimum: 8 GB (CPU training)
- Recommended: 16 GB (smooth experience)
- Optimal: 32 GB (LLaMA 70B + parallel training)

### GPU (Optional):
- Not required - CPU training works fine for small models
- If available: NVIDIA GPU with 4GB+ VRAM (speeds up training 10-20x)

---

## ✅ Day 0 Checklist

Before going to bed tonight, ensure:

- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] Setup verification passes (`python verify_ml_setup.py`)
- [ ] Qdrant running (`docker ps` shows reddygo-qdrant)
- [ ] Ollama installed (`ollama --version`)
- [ ] LLaMA downloaded (`ollama list` shows llama3.1:8b)
- [ ] Transformer components tested (`python -m ml.custom_llm.transformer`)
- [ ] GPT model tested (`python -m ml.custom_llm.model`)
- [ ] Tokenizer tested (`python -m ml.custom_llm.tokenizer`)
- [ ] Watched Karpathy video (or scheduled for tomorrow morning)
- [ ] Read `ML_AI_LEARNING_GUIDE.md` overview

---

## 🚀 Ready for Day 1?

If all checks pass, you're ready to start building a transformer from scratch tomorrow!

**Tomorrow morning, start with:**
```bash
cd C:/Users/akhil/reddygo-platform/backend
python -m ml.custom_llm.train
```

This will train your first custom LLM! 🎉

---

## 📞 Need Help?

If you encounter issues:
1. Check the troubleshooting section above
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Restart Docker/Ollama if needed
5. Check disk space and RAM availability

---

**Good luck with Day 0 setup! See you tomorrow for Day 1! 🚀**
