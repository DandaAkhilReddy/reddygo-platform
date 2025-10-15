# 🚀 ReddyGo Complete ML/AI Learning Platform

## 🎯 What You'll Learn - ALL in This ONE Project

This codebase is your **complete AI/ML engineering bootcamp** in one fitness app:

### Technologies You'll Master:
✅ **PyTorch** - Custom neural networks & deep learning
✅ **TensorFlow** - Production ML & model deployment
✅ **LangChain** - AI agent orchestration & tool calling
✅ **LLaMA** - Local LLM via Ollama (no API costs!)
✅ **Vector Databases** - Qdrant & Pinecone for embeddings
✅ **Mem0/Supermemory** - Long-term AI memory
✅ **OpenAI SDK** - GPT-4 integration (already working!)
✅ **FastAPI** - ML model serving
✅ **Firebase** - Real-time data & auth
✅ **Vue.js** - Reactive ML dashboard
✅ **MLOps** - Model versioning, monitoring, A/B testing

---

## 📊 Complete Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              iPhone PWA (Vue.js + Vite)                      │
│  ┌────────────┬────────────┬────────────┬────────────┐      │
│  │ AI Coach   │ ML Insights│ Challenges │  Profile   │      │
│  │ Chat       │ Dashboard  │ Map        │  Stats     │      │
│  └────────────┴────────────┴────────────┴────────────┘      │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│            FastAPI Backend (Python 3.11)                     │
├──────────────────────────────────────────────────────────────┤
│  EXISTING ROUTERS (Already Working!):                        │
│  ├─ /api/coaching/* → CoachAgent (Mem0 + OpenAI)            │
│  ├─ /api/challenges/* → Firebase challenges                 │
│  ├─ /api/users/* → Firebase auth & profiles                 │
│  ├─ /api/friends/* → Social features                        │
│  └─ /api/validation/* → GPS anti-cheat                      │
│                                                              │
│  NEW ML ROUTERS (You'll Build):                             │
│  ├─ /ml/predict-performance → PyTorch LSTM                  │
│  ├─ /ml/recommend-challenges → TensorFlow CF                │
│  ├─ /ml/analyze-form → MediaPipe + CNN                      │
│  ├─ /vector/search → Qdrant semantic search                 │
│  ├─ /llm/local → LLaMA via Ollama                          │
│  └─ /langchain/agent → Multi-tool orchestration            │
└─────────────────────────┬────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  LangChain   │  │ Vector DBs   │  │ LLM Layer    │
│  Agent       │  │              │  │              │
├──────────────┤  ├──────────────┤  ├──────────────┤
│ Orchestrates:│  │ Qdrant:      │  │ OpenAI:      │
│ • Mem0       │◄─┤ • Workouts   │  │ • GPT-4o-mini│
│ • OpenAI     │  │ • Exercises  │  │ • Embeddings │
│ • LLaMA      │◄─┤ • Memories   │  │              │
│ • PyTorch    │  │              │  │ Ollama:      │
│ • TensorFlow │  │ Pinecone:    │  │ • LLaMA 3.1  │
│ • Vector DB  │  │ • (optional) │  │   8B/70B     │
│ • Web Search │  │ • Cloud      │  │ • Local      │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## 🗂️ Project Structure

```
reddygo-platform/
├── backend/
│   ├── ml/                        # NEW: Machine Learning
│   │   ├── __init__.py
│   │   ├── pytorch/
│   │   │   ├── performance_predictor.py   # LSTM for time series
│   │   │   ├── injury_classifier.py       # CNN for risk scoring
│   │   │   ├── workout_embeddings.py      # Transformer embeddings
│   │   │   ├── train.py                   # Training scripts
│   │   │   └── models/                    # Saved .pth files
│   │   │
│   │   ├── tensorflow/
│   │   │   ├── recommender.py             # Collaborative filtering
│   │   │   ├── route_optimizer.py         # RL agent
│   │   │   ├── form_analyzer.py           # MediaPipe + CNN
│   │   │   ├── train.py
│   │   │   └── models/                    # Saved .h5/.pb files
│   │   │
│   │   └── inference.py                   # Unified inference API
│   │
│   ├── vector/                    # NEW: Vector Database
│   │   ├── __init__.py
│   │   ├── qdrant_client.py              # Qdrant setup
│   │   ├── embeddings.py                 # Generate embeddings
│   │   ├── search.py                     # Semantic search
│   │   └── collections.py                # Collection management
│   │
│   ├── llm/                       # NEW: LLM Layer
│   │   ├── __init__.py
│   │   ├── langchain_agent.py            # Multi-tool agent
│   │   ├── ollama_client.py              # Local LLaMA
│   │   ├── openai_client.py              # (existing, enhanced)
│   │   ├── tools/
│   │   │   ├── memory_tools.py           # Mem0 tools
│   │   │   ├── ml_tools.py               # PyTorch/TF tools
│   │   │   ├── vector_tools.py           # Search tools
│   │   │   └── web_tools.py              # SearXNG tools
│   │   └── prompts/
│   │       ├── coach.txt
│   │       ├── analyst.txt
│   │       └── planner.txt
│   │
│   ├── routers/
│   │   ├── coaching.py            # UPDATED: Now uses LangChain
│   │   ├── ml.py                  # NEW: ML endpoints
│   │   ├── vector.py              # NEW: Vector search
│   │   ├── challenges.py          # (existing)
│   │   ├── users.py               # (existing)
│   │   └── ... (16 other routers)
│   │
│   ├── agents/                    # EXISTING: AI Agents
│   │   ├── coach.py               # (existing - will integrate LangChain)
│   │   ├── base.py
│   │   └── tools.py               # (existing - will enhance)
│   │
│   └── requirements.txt           # ✅ UPDATED with all ML deps

├── pwa-vue/                       # Vue.js PWA Frontend
│   ├── src/
│   │   ├── views/
│   │   │   ├── HomeView.vue
│   │   │   ├── CoachView.vue              # AI Chat with agent
│   │   │   ├── MLDashboard.vue            # ML predictions viz
│   │   │   ├── MapView.vue                # Leaflet map
│   │   │   └── ProfileView.vue
│   │   │
│   │   ├── components/
│   │   │   ├── AIChat.vue                 # Chat interface
│   │   │   ├── MLPredictions.vue          # Show PyTorch/TF outputs
│   │   │   ├── VectorSearch.vue           # Semantic search UI
│   │   │   ├── ModelSelector.vue          # Choose LLaMA/GPT-4
│   │   │   └── BottomNav.vue
│   │   │
│   │   ├── stores/
│   │   │   ├── user.ts
│   │   │   ├── coach.ts                   # AI agent state
│   │   │   ├── ml.ts                      # ML predictions
│   │   │   ├── vector.ts                  # Vector search
│   │   │   └── challenges.ts
│   │   │
│   │   ├── api/
│   │   │   ├── client.ts                  # Axios base
│   │   │   ├── coaching.ts                # Coach endpoints
│   │   │   ├── ml.ts                      # ML endpoints
│   │   │   └── vector.ts                  # Vector endpoints
│   │   │
│   │   └── utils/
│   │       ├── embeddings.ts              # Client-side embeddings
│   │       └── ml-utils.ts                # ML helpers
│   │
│   └── package.json

├── notebooks/                     # NEW: Jupyter Notebooks
│   ├── 01_pytorch_lstm_training.ipynb
│   ├── 02_tensorflow_recommender.ipynb
│   ├── 03_vector_embeddings.ipynb
│   ├── 04_langchain_experiments.ipynb
│   ├── 05_llama_vs_gpt4_comparison.ipynb
│   └── 06_end_to_end_pipeline.ipynb

├── data/                          # NEW: Training Data
│   ├── workouts/                  # Workout history
│   ├── embeddings/                # Pre-computed embeddings
│   └── models/                    # ML model checkpoints

└── docs/
    ├── ML_AI_LEARNING_GUIDE.md    # THIS FILE
    ├── PYTORCH_TUTORIAL.md        # Step-by-step PyTorch
    ├── TENSORFLOW_TUTORIAL.md     # Step-by-step TensorFlow
    ├── LANGCHAIN_TUTORIAL.md      # Agent building
    └── DEPLOYMENT_GUIDE.md        # MLOps & production
```

---

## 🚀 Implementation Phases

### **Phase 1: ML Infrastructure** (Week 1) - IN PROGRESS ✅

**What You're Doing:**
- ✅ Install PyTorch, TensorFlow, LangChain
- ✅ Set up Qdrant locally
- ⏳ Install Ollama + LLaMA 3.1
- ⏳ Create ML directory structure
- ⏳ Test all connections

**Commands:**
```bash
# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com/download

# Pull LLaMA 3.1
ollama pull llama3.1:8b  # 8B params, 4GB RAM

# Start Qdrant with Docker
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

# Test Ollama
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

---

### **Phase 2: PyTorch Models** (Week 2)

**Model 1: Performance Predictor (LSTM)**

```python
# ml/pytorch/performance_predictor.py
import torch
import torch.nn as nn

class PerformancePredictor(nn.Module):
    """
    Predicts next workout performance based on past workouts.

    Architecture: LSTM → Dense → Output
    Input: Sequence of past workouts [distance, time, heart_rate, pace]
    Output: Predicted time for next workout
    """
    def __init__(self, input_size=10, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # Take last time step
        x = self.relu(self.fc1(last_output))
        return self.fc2(x)
```

**Training Script:**
```python
# ml/pytorch/train.py
def train_performance_predictor():
    # Load data from Firebase
    workouts = fetch_user_workouts(user_id="test_user")

    # Prepare sequences
    X, y = prepare_sequences(workouts, sequence_length=7)  # Last 7 workouts

    # Train
    model = PerformancePredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

    # Save
    torch.save(model.state_dict(), 'models/performance_predictor.pth')
```

**API Endpoint:**
```python
# routers/ml.py
@router.post("/ml/predict-performance")
async def predict_performance(user_id: str, challenge_distance: float):
    # Load model
    model = PerformancePredictor()
    model.load_state_dict(torch.load('ml/pytorch/models/performance_predictor.pth'))
    model.eval()

    # Get user's recent workouts
    workouts = get_recent_workouts(user_id, limit=7)
    X = prepare_input(workouts, challenge_distance)

    # Predict
    with torch.no_grad():
        predicted_time = model(X).item()

    return {
        "predicted_time_minutes": predicted_time,
        "confidence": 0.85,
        "based_on_workouts": len(workouts)
    }
```

**What You Learn:**
- LSTM architecture for time series
- PyTorch training loops
- Model saving/loading
- FastAPI integration

---

### **Phase 3: TensorFlow Models** (Week 2)

**Model 1: Challenge Recommender**

```python
# ml/tensorflow/recommender.py
import tensorflow as tf

class ChallengeRecommender(tf.keras.Model):
    """
    Collaborative filtering for challenge recommendations.

    Uses matrix factorization to learn user/challenge embeddings.
    """
    def __init__(self, num_users, num_challenges, embedding_dim=50):
        super().__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.challenge_embedding = tf.keras.layers.Embedding(num_challenges, embedding_dim)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_id, challenge_id = inputs
        user_vec = self.user_embedding(user_id)
        challenge_vec = self.challenge_embedding(challenge_id)
        concat = tf.concat([user_vec, challenge_vec], axis=-1)
        return self.dense(concat)
```

---

### **Phase 4: Vector Database** (Week 3)

**Qdrant Setup:**

```python
# vector/qdrant_client.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

# Initialize
client = QdrantClient(host="localhost", port=6333)
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Create collection
client.create_collection(
    collection_name="workouts",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Store workout
workout_text = "5km run in 25 minutes, moderate pace, felt good"
embedding = encoder.encode(workout_text)

client.upsert(
    collection_name="workouts",
    points=[{
        "id": "workout_123",
        "vector": embedding.tolist(),
        "payload": {
            "user_id": "john",
            "distance_km": 5,
            "time_minutes": 25,
            "type": "run"
        }
    }]
)

# Semantic search
query = "easy short run"
query_vector = encoder.encode(query)
results = client.search(
    collection_name="workouts",
    query_vector=query_vector.tolist(),
    limit=5
)
```

---

### **Phase 5: LangChain Agent** (Week 3)

```python
# llm/langchain_agent.py
from langchain.agents import initialize_agent, Tool
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory

# Tools
tools = [
    Tool(
        name="Get User Memory",
        func=get_mem0_memories,
        description="Retrieve user's fitness history and preferences from Mem0"
    ),
    Tool(
        name="Predict Performance",
        func=pytorch_predict,
        description="Use PyTorch LSTM to predict workout performance"
    ),
    Tool(
        name="Recommend Challenges",
        func=tensorflow_recommend,
        description="Use TensorFlow to recommend challenges"
    ),
    Tool(
        name="Vector Search",
        func=qdrant_search,
        description="Search workout history semantically"
    ),
    Tool(
        name="Local LLM",
        func=ollama_generate,
        description="Use local LLaMA for simple queries"
    ),
    Tool(
        name="GPT-4",
        func=openai_generate,
        description="Use GPT-4 for complex reasoning"
    )
]

# LLM (can switch between Ollama and OpenAI)
llm = Ollama(model="llama3.1")  # or OpenAI(model="gpt-4")

# Memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="conversational-react-description",
    memory=memory,
    verbose=True
)

# Use it!
response = agent.run("I want to run a 5K. Predict my time and recommend a challenge.")
```

---

## 🎓 Learning Path

### Week 1: Foundation
- [ ] Install all dependencies
- [ ] Set up Qdrant + Ollama
- [ ] Test connections
- [ ] Read PyTorch basics

### Week 2: Deep Learning
- [ ] Build PyTorch LSTM
- [ ] Train on sample data
- [ ] Build TensorFlow recommender
- [ ] Create inference API

### Week 3: LLMs & Orchestration
- [ ] Set up LangChain
- [ ] Build multi-tool agent
- [ ] Integrate LLaMA locally
- [ ] Compare LLaMA vs GPT-4

### Week 4: Frontend
- [ ] Vue.js ML dashboard
- [ ] Connect all APIs
- [ ] Real-time updates

### Week 5: Optimization
- [ ] Fine-tune models
- [ ] Reduce costs (use LLaMA more)
- [ ] A/B testing

### Week 6: Deploy
- [ ] Production deployment
- [ ] Monitoring
- [ ] Scale testing

---

## 📚 Resources

### PyTorch
- [Official Tutorial](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch)

### TensorFlow
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [Hands-On ML](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

### LangChain
- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)
- [Agent Tutorial](https://python.langchain.com/docs/modules/agents/)

### Vector DBs
- [Qdrant Tutorial](https://qdrant.tech/documentation/quick-start/)
- [Embeddings Guide](https://www.pinecone.io/learn/vector-embeddings/)

---

## 🚀 Next Steps

1. **Run Setup Script:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Start Services:**
   ```bash
   # Terminal 1: Qdrant
   docker run -p 6333:6333 qdrant/qdrant

   # Terminal 2: Ollama
   ollama serve

   # Terminal 3: Backend
   python main.py
   ```

3. **Test ML Endpoint:**
   ```bash
   curl -X POST http://localhost:8080/ml/predict-performance \
     -H "Content-Type: application/json" \
     -d '{"user_id": "test", "challenge_distance": 5}'
   ```

---

## 🎯 This ONE Project = Complete ML Engineering Portfolio

- ✅ Deep Learning (PyTorch + TensorFlow)
- ✅ LLM Orchestration (LangChain)
- ✅ Local LLMs (LLaMA)
- ✅ Vector Search (Qdrant)
- ✅ Production ML (FastAPI serving)
- ✅ Frontend (Vue.js dashboard)
- ✅ MLOps (versioning, monitoring)

**You'll be ready for ANY ML engineering role!** 🚀
