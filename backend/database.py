"""
ReddyGo Database Configuration

Supabase connection and database utilities for PostgreSQL + PostGIS.
"""

import os
from supabase import create_client, Client
from typing import Optional

# Supabase client initialization
def get_supabase_client() -> Client:
    """Get configured Supabase client."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

    return create_client(url, key)

# Service role client for admin operations
def get_service_client() -> Client:
    """Get Supabase client with service role key (admin access)."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

    return create_client(url, key)

# Database schema SQL (run manually in Supabase SQL editor)
SCHEMA_SQL = """
-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_location GEOGRAPHY(POINT, 4326),
    stats JSONB DEFAULT '{"challenges_completed": 0, "total_distance_km": 0, "total_calories": 0}'::jsonb,
    preferences JSONB DEFAULT '{}'::jsonb
);

-- Challenges table with geospatial support
CREATE TABLE IF NOT EXISTS challenges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    creator_id UUID REFERENCES users(id),
    title TEXT NOT NULL,
    description TEXT,
    challenge_type TEXT NOT NULL, -- 'race', 'zone_control', 'scavenger_hunt'
    zone_center GEOGRAPHY(POINT, 4326) NOT NULL,
    zone_radius_meters INTEGER NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    max_participants INTEGER DEFAULT 10,
    rules JSONB NOT NULL,
    status TEXT DEFAULT 'pending', -- 'pending', 'active', 'completed', 'cancelled'
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Challenge participants
CREATE TABLE IF NOT EXISTS challenge_participants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    challenge_id UUID REFERENCES challenges(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id),
    joined_at TIMESTAMPTZ DEFAULT NOW(),
    status TEXT DEFAULT 'active', -- 'active', 'completed', 'disqualified'
    stats JSONB DEFAULT '{}'::jsonb,
    UNIQUE(challenge_id, user_id)
);

-- GPS tracks for validation
CREATE TABLE IF NOT EXISTS gps_tracks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    challenge_id UUID REFERENCES challenges(id),
    track_data JSONB NOT NULL, -- Array of {lat, lon, timestamp, accuracy, speed}
    sensor_data JSONB, -- Accelerometer, gyroscope data
    validation_score FLOAT, -- 0.0-1.0 confidence score
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Spatial index for fast nearby queries
CREATE INDEX IF NOT EXISTS idx_challenges_zone ON challenges USING GIST(zone_center);
CREATE INDEX IF NOT EXISTS idx_users_location ON users USING GIST(last_location);

-- Function to find nearby challenges
CREATE OR REPLACE FUNCTION find_nearby_challenges(
    user_lat FLOAT,
    user_lon FLOAT,
    radius_meters INTEGER DEFAULT 5000
)
RETURNS TABLE(
    challenge_id UUID,
    title TEXT,
    distance_meters FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.title,
        ST_Distance(
            c.zone_center::geography,
            ST_SetSRID(ST_MakePoint(user_lon, user_lat), 4326)::geography
        ) as distance
    FROM challenges c
    WHERE
        c.status = 'active'
        AND ST_DWithin(
            c.zone_center::geography,
            ST_SetSRID(ST_MakePoint(user_lon, user_lat), 4326)::geography,
            radius_meters
        )
    ORDER BY distance ASC;
END;
$$ LANGUAGE plpgsql;

-- Function to find nearby users
CREATE OR REPLACE FUNCTION find_nearby_users(
    user_lat FLOAT,
    user_lon FLOAT,
    radius_meters INTEGER DEFAULT 200
)
RETURNS TABLE(
    user_id UUID,
    name TEXT,
    distance_meters FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        u.id,
        u.name,
        ST_Distance(
            u.last_location::geography,
            ST_SetSRID(ST_MakePoint(user_lon, user_lat), 4326)::geography
        ) as distance
    FROM users u
    WHERE
        u.last_location IS NOT NULL
        AND ST_DWithin(
            u.last_location::geography,
            ST_SetSRID(ST_MakePoint(user_lon, user_lat), 4326)::geography,
            radius_meters
        )
    ORDER BY distance ASC;
END;
$$ LANGUAGE plpgsql;
"""
