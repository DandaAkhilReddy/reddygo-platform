# 🏃‍♂️ ReddyGo Platform

**Geo-Fitness Platform with Real-Time Challenges & AI Coaching**

Transform outdoor fitness into a competitive, social experience with location-based challenges, anti-cheat GPS validation, and personalized AI coaching.

---

## 🎯 What is ReddyGo?

ReddyGo is a mobile-first fitness platform that creates **spontaneous, location-based challenges** between nearby users. Unlike traditional fitness apps that focus on solo tracking, ReddyGo turns every outdoor workout into a potential multiplayer experience.

### Key Features

- 🌍 **Geo-Challenges:** Real-time challenges with users within 200m radius
- 🔒 **5-Layer Anti-Cheat:** Device integrity, GPS quality, sensor fusion, sanity checks, ML anomaly detection
- 🤖 **AI Coach with Memory:** Personalized coaching powered by OpenAI + Supermemory (Mem0)
- 🔐 **Privacy-First:** Local-first storage with optional cloud sync
- 💰 **Cost-Optimized:** 99.2% margin at 100K users ($792/month operational cost)

---

## 🏗️ Architecture

```
reddygo-platform/
├── backend/              # FastAPI + PostGIS + Temporal.io
│   ├── routers/         # API endpoints (challenges, users, validation)
│   ├── database.py      # Supabase + PostGIS geospatial queries
│   ├── models.py        # Pydantic schemas
│   └── main.py          # FastAPI app entry point
│
├── mobile/              # React Native + Expo (TODO)
│   └── (Coming soon)
│
├── agents/              # OpenAI Agents SDK (TODO)
│   └── (Coming soon)
│
└── docs/                # Documentation
```

### Technology Stack

**Backend:**
- **API:** FastAPI (Python 3.11)
- **Database:** Supabase (PostgreSQL + PostGIS + Realtime)
- **Deployment:** Fly.io (global edge, auto-scaling)
- **Workflows:** Temporal.io (durable orchestration)
- **AI:** OpenAI GPT-4o-mini + Supermemory
- **Validation:** NumPy, sensor fusion algorithms

**Mobile (Planned):**
- **Framework:** React Native + Expo
- **Storage:** MMKV (encrypted KV), SQLite (relational), React Native FS (files)
- **GPS:** React Native Location with adaptive sampling
- **Offline:** Local-first with sync queue

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Supabase account (free tier works)
- OpenAI API key
- Fly.io account (for deployment)

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/reddygo-platform.git
cd reddygo-platform/backend
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 4. Set Up Database

1. Create Supabase project
2. Enable PostGIS extension:
   ```sql
   CREATE EXTENSION IF NOT EXISTS postgis;
   ```
3. Run schema from `backend/database.py` (SCHEMA_SQL)

### 5. Run Locally

```bash
python main.py
```

Visit http://localhost:8080/docs for interactive API documentation.

---

## 📦 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment guide.

### Backend Deployment (Fly.io)

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Authenticate
flyctl auth login

# Deploy
cd backend
flyctl launch --no-deploy
flyctl secrets set SUPABASE_URL="..." SUPABASE_KEY="..." OPENAI_API_KEY="..."
flyctl deploy
```

### Frontend Deployment (Vercel)

**Option 1: Deploy via Vercel Dashboard (Recommended)**

1. Push your code to GitHub
2. Visit [vercel.com](https://vercel.com) and sign in
3. Click "Add New Project"
4. Import your GitHub repository: `DandaAkhilReddy/reddygo-platform`
5. Vercel will auto-detect Next.js configuration
6. Add environment variables in Vercel dashboard:
   - `NEXT_PUBLIC_API_URL` - Your backend API URL (e.g., `https://your-app.fly.dev`)
   - `NEXT_PUBLIC_FIREBASE_API_KEY` - From Firebase Console
   - `NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN`
   - `NEXT_PUBLIC_FIREBASE_PROJECT_ID`
   - `NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET`
   - `NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID`
   - `NEXT_PUBLIC_FIREBASE_APP_ID`
   - `NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID`
   - `NEXT_PUBLIC_FIREBASE_DATABASE_URL`
7. Click "Deploy"

**Option 2: Deploy via Vercel CLI**

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd frontend
vercel

# For production deployment
vercel --prod
```

The frontend will be live at your Vercel URL (e.g., `https://reddygo-platform.vercel.app`)

---

## 🔐 Security & Anti-Cheat

### 5-Layer Validation System

1. **Device Integrity:** Detect rooted/jailbroken devices, mock location
2. **GPS Quality:** Filter low-accuracy points (>100m accuracy)
3. **Sensor Fusion:** Correlate GPS with accelerometer/gyroscope
4. **Physical Sanity:** Detect teleportation, excessive speed (>43 km/h)
5. **ML Anomaly Detection:** Isolation Forest for pattern recognition

**Validation Endpoint:** `POST /api/validation/validate`

```python
# Example response
{
  "is_valid": true,
  "confidence_score": 0.87,
  "flags": [],
  "details": {
    "total_points": 120,
    "low_accuracy_count": 5,
    "anomaly_score": 0.15
  }
}
```

---

## 🤖 AI Coach Agent (Planned)

**6-Agent System with OpenAI Agents SDK:**

1. **Coordinator Agent:** Challenge orchestration
2. **Validation Agent:** GPS track analysis
3. **Coach Agent:** Personalized training advice
4. **Social Agent:** Community management
5. **Safety Agent:** Restricted zone detection
6. **Reward Agent:** Achievement tracking

**Cost:** $0.023 per challenge (97% cheaper than monolithic GPT-4)

---

## 📊 API Endpoints

### Challenges

- `POST /api/challenges/create` - Create geo-challenge
- `POST /api/challenges/nearby` - Find nearby challenges
- `GET /api/challenges/{id}` - Get challenge details
- `POST /api/challenges/{id}/join` - Join challenge
- `GET /api/challenges/{id}/participants` - List participants

### Users

- `POST /api/users/register` - Register user
- `GET /api/users/{id}` - Get user profile
- `PUT /api/users/{id}/location` - Update location
- `POST /api/users/nearby` - Find nearby users

### Validation

- `POST /api/validation/validate` - Validate GPS track

### Health

- `GET /health` - Health check

---

## 💰 Cost Analysis

**At 100K Monthly Active Users:**

| Service | Cost/Month | Details |
|---------|------------|---------|
| Fly.io | $290 | 15 machines × $19.35 |
| Supabase | $125 | Pro plan (100GB) |
| OpenAI | $300 | 500K challenges × $0.023 |
| Temporal.io | $50 | 1M actions |
| SearXNG | $10 | Self-hosted on Hetzner |
| Cloudflare R2 | $15 | 1TB storage |
| **Total** | **$792** | **$0.0079 per user** |

**Revenue:** $99,900/month (10% conversion @ $9.99/month)
**Margin:** 99.2%

---

## 🗺️ Roadmap

### Phase 1: MVP (Current)
- ✅ Backend API (FastAPI + Supabase)
- ✅ Challenge creation & discovery
- ✅ 5-layer GPS validation
- ✅ User management & profiles
- 🔄 Fly.io deployment
- 🔄 Supabase schema setup

### Phase 2: Mobile App
- 📱 React Native + Expo setup
- 🗺️ GPS tracking with adaptive sampling
- 💾 Local-first storage (MMKV + SQLite)
- 🔄 Offline sync queue

### Phase 3: AI Agents
- 🤖 OpenAI Agents SDK integration
- 🧠 Supermemory (Mem0) for context
- 🔍 SearXNG integration
- ⚡ Temporal.io orchestration

### Phase 4: Production
- 🚀 GitHub Actions CI/CD
- 📊 Monitoring & analytics
- 👥 100 beta users
- 🌐 Custom domain & branding

---

## 🛠️ Development

### Run Tests

```bash
pytest backend/
```

### Lint Code

```bash
black backend/
flake8 backend/
```

### Database Migrations

```bash
# Apply schema changes via Supabase SQL Editor
```

---

## 📚 Documentation

- [Deployment Guide](DEPLOYMENT.md)
- [API Reference](http://localhost:8080/docs)
- [Research Documentation](../Research/)

---

## 🤝 Contributing

We're not accepting external contributions yet as this is in early development. Check back after Phase 1 MVP is complete!

---

## 📄 License

Proprietary - All Rights Reserved

---

## 🙋 Support

For issues or questions, please contact: [your-email@example.com]

---

Built with ❤️ for the fitness community
