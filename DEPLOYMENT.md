# ReddyGo Backend Deployment Guide

## Prerequisites

### 1. Fly.io Setup

✅ **Flyctl CLI installed** (v0.3.193)

**Next steps:**

```bash
# Authenticate with Fly.io
flyctl auth login

# This will open a browser for login
```

### 2. Supabase Setup

Create a Supabase project:

1. Go to https://supabase.com/dashboard
2. Click "New Project"
3. Name: `reddygo-production`
4. Database Password: (save this!)
5. Region: Choose closest to users (e.g., `us-west-1`)

**Enable PostGIS:**

```sql
-- Run in Supabase SQL Editor
CREATE EXTENSION IF NOT EXISTS postgis;
```

**Create Database Schema:**

Copy the SQL from `backend/database.py` (SCHEMA_SQL) and run in Supabase SQL Editor.

**Get Connection Details:**

- Project Settings → API → URL (SUPABASE_URL)
- Project Settings → API → anon/public key (SUPABASE_KEY)
- Project Settings → API → service_role key (SUPABASE_SERVICE_ROLE_KEY)
- Project Settings → Database → Connection string (DATABASE_URL)

### 3. OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Create new secret key
3. Save as `OPENAI_API_KEY`

### 4. Environment Variables

Create `.env` file in `backend/` directory:

```bash
# Copy from .env.example
cp backend/.env.example backend/.env

# Then edit with your values:
ENVIRONMENT=production
PORT=8080
ALLOWED_ORIGINS=http://localhost:3000,https://reddygo.app

# Supabase
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=eyJhbGc...
SUPABASE_SERVICE_ROLE_KEY=eyJhbGc...
DATABASE_URL=postgresql://postgres:password@db.xxxxx.supabase.co:5432/postgres

# OpenAI
OPENAI_API_KEY=sk-...

# Redis (optional for now)
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-super-secret-key-change-this
```

## Deployment Steps

### Step 1: Test Locally

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run locally
python main.py
```

Visit http://localhost:8080/docs to test API.

### Step 2: Initialize Fly.io App

```bash
cd backend

# Launch Fly app (will use existing fly.toml)
flyctl launch --no-deploy

# This will:
# - Create a new Fly.io app called "reddygo-api"
# - Use existing fly.toml configuration
# - Skip initial deployment
```

### Step 3: Set Secrets

```bash
# Set all environment variables as Fly secrets
flyctl secrets set \
  ENVIRONMENT=production \
  PORT=8080 \
  ALLOWED_ORIGINS="https://reddygo.app" \
  SUPABASE_URL="https://xxxxx.supabase.co" \
  SUPABASE_KEY="eyJhbGc..." \
  SUPABASE_SERVICE_ROLE_KEY="eyJhbGc..." \
  DATABASE_URL="postgresql://postgres:password@..." \
  OPENAI_API_KEY="sk-..." \
  SECRET_KEY="your-super-secret-key"
```

### Step 4: Deploy

```bash
# Deploy to Fly.io
flyctl deploy

# Monitor deployment
flyctl logs
```

### Step 5: Verify Deployment

```bash
# Check status
flyctl status

# Test health endpoint
curl https://reddygo-api.fly.dev/health

# Open in browser
flyctl open
```

## Scaling Configuration

Current `fly.toml` settings:

- **Min machines:** 1 (always on)
- **Max machines:** 10 (auto-scale)
- **Concurrency:** 200 soft / 250 hard
- **Health checks:** Every 10s on `/health`

**Cost estimate:** $1.94/month per machine (shared-cpu-1x, 256MB RAM)

## Monitoring

```bash
# View logs
flyctl logs

# Monitor metrics
flyctl dashboard

# SSH into machine
flyctl ssh console
```

## Troubleshooting

### Health Check Fails

```bash
# Check logs
flyctl logs

# Common issues:
# - Missing environment variables (check secrets)
# - Supabase connection error (verify DATABASE_URL)
# - Port mismatch (must be 8080)
```

### Database Connection Error

```bash
# Test database connection
flyctl ssh console

# Then inside machine:
python -c "from database import get_supabase_client; client = get_supabase_client(); print('Connected!')"
```

### Out of Memory

Update `fly.toml`:

```toml
[vm]
  memory = "512mb"  # Increase from 256mb
```

## Next Steps After Deployment

1. ✅ Backend deployed
2. 🔄 Set up Temporal.io for workflow orchestration
3. 🔄 Build React Native mobile app
4. 🔄 Implement AI Coach agent
5. 🔄 Set up GitHub Actions CI/CD
6. 🔄 Configure custom domain
7. 🔄 Deploy to production with 100 beta users

## Useful Commands

```bash
# Scale up/down
flyctl scale count 3  # Run 3 machines

# Update secrets
flyctl secrets set NEW_SECRET=value

# Destroy app (careful!)
flyctl apps destroy reddygo-api

# View app info
flyctl info

# Open dashboard
flyctl dashboard
```
