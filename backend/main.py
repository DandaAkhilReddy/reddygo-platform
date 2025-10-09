"""
ReddyGo Backend API - FastAPI Application

Main entry point for the ReddyGo geo-fitness platform backend.
Handles real-time geo-challenges, AI coaching, and anti-cheat validation.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import routers
from routers import challenges, users, validation, coaching, rewards, encryption, friends, communities, subscriptions

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("🚀 ReddyGo API starting up...")
    print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")

    # Initialize Firebase
    try:
        from firebase_client import initialize_firebase
        initialize_firebase()
        print("🔥 Firebase Firestore initialized")
    except Exception as e:
        print(f"⚠️  Firebase initialization failed: {e}")

    # Initialize Firebase Realtime Database
    try:
        from realtime_db_client import initialize_realtime_db
        initialize_realtime_db()
        print("🔥 Firebase Realtime Database initialized")
    except Exception as e:
        print(f"⚠️  Realtime Database initialization failed: {e}")

    # Initialize AI Agents
    print("🤖 Initializing AI Agents...")
    print("  ✅ Coordinator Agent (challenge orchestration)")
    print("  ✅ Coach Agent (personalized training)")
    print("  ✅ Validation Agent (GPS verification)")
    print("  ✅ Social Agent (community content)")
    print("  ✅ Safety Agent (restricted zones + weather)")
    print("  ✅ Reward Agent (usage-based rewards)")

    # Check encryption availability
    try:
        from encryption import E2EEManager
        print("🔐 E2EE encryption available")
    except ImportError:
        print("⚠️  E2EE encryption not available (install PyNaCl)")

    yield

    # Shutdown
    print("👋 ReddyGo API shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="ReddyGo API",
    description="Geo-fitness platform with real-time challenges and AI coaching",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "reddygo-api",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ReddyGo API",
        "docs": "/docs",
        "health": "/health"
    }

# Include routers
app.include_router(challenges.router, prefix="/api/challenges", tags=["challenges"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(validation.router, prefix="/api/validation", tags=["validation"])
app.include_router(coaching.router, prefix="/api/coaching", tags=["coaching"])
app.include_router(rewards.router, prefix="/api/rewards", tags=["rewards"])
app.include_router(encryption.router, prefix="/api/encryption", tags=["encryption"])
app.include_router(friends.router, prefix="/api/friends", tags=["friends"])
app.include_router(communities.router, prefix="/api/communities", tags=["communities"])
app.include_router(subscriptions.router, prefix="/api/subscriptions", tags=["subscriptions"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        reload=os.getenv("ENVIRONMENT") == "development"
    )
