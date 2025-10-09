# ReddyGo Creator Economy - Database Schema

**Complete Firestore & Realtime Database Schema for Creator Marketplace Platform**

---

## Firestore Collections

### 1. `/users/{user_id}`
Core user profile with social and creator features.

```json
{
  "email": "user@example.com",
  "name": "John Doe",
  "profile_photo_url": "https://...",
  "created_at": timestamp,
  "location": GeoPoint(37.7749, -122.4194),

  "stats": {
    "challenges_completed": 150,
    "total_distance_km": 500,
    "total_calories": 50000,
    "challenges_this_week": 5,
    "points_balance": 15000,
    "lifetime_points": 50000,
    "current_streak": 7,
    "longest_streak": 30
  },

  "preferences": {
    "fitness_level": "intermediate",
    "injuries": ["knee"],
    "privacy": {
      "show_stats_to_friends": true,
      "show_stats_to_family": true,
      "show_location": false
    }
  },

  "subscription": {
    "tier": "pro_individual | pro_family | creator_pro | free",
    "status": "active | canceled | expired",
    "expires_at": timestamp,
    "family_subscription_id": "sub_xxx" // If member of family plan
  },

  "social": {
    "friends_count": 42,
    "followers_count": 120,
    "following_count": 85,
    "communities": ["community_1", "community_2"]
  },

  "creator": {
    "is_creator": true,
    "creator_level": "beginner | rising | established | elite",
    "verified": true,
    "total_sales": 1250,
    "total_earnings": 8750.00,
    "revenue_share_percentage": 85,
    "followers": 1500
  },

  "referral": {
    "referral_code": "JOHN_ABC123",
    "referred_by": "user_id",
    "total_referrals": 15,
    "referral_earnings": 2500.00
  },

  "earnings": {
    "subscription_revenue": 125.50, // From family/community
    "creator_revenue": 8750.00, // From content sales
    "referral_revenue": 2500.00, // From referrals
    "total_lifetime": 11375.50,
    "pending_payout": 450.00,
    "last_payout_date": timestamp
  },

  // E2EE keys
  "public_key": bytes,
  "encrypted_private_key": string
}
```

---

### 2. `/subscriptions/{subscription_id}`
Subscription plans (Individual, Family, Creator).

```json
{
  "type": "pro_individual | pro_family | creator_pro",
  "user_id": "user_123", // Owner
  "status": "active | canceled | expired",
  "price": 14.99,
  "billing_cycle": "monthly | annual",
  "created_at": timestamp,
  "expires_at": timestamp,
  "auto_renew": true,

  "stripe": {
    "customer_id": "cus_xxx",
    "subscription_id": "sub_xxx",
    "payment_method_id": "pm_xxx"
  }
}
```

---

### 3. `/family_subscriptions/{subscription_id}`
Family plan with unlimited members and founder benefits.

```json
{
  "subscription_id": "sub_xxx",
  "owner_id": "user_123", // Family founder
  "plan": "pro_family",
  "status": "active",
  "price": 14.99,
  "created_at": timestamp,

  "members": [
    {"user_id": "user_123", "role": "owner", "joined_at": timestamp},
    {"user_id": "user_456", "role": "member", "joined_at": timestamp},
    {"user_id": "user_789", "role": "member", "joined_at": timestamp}
  ],

  "founder_benefits": {
    "founder_badge": true,
    "revenue_share_percentage": 20, // 20% of all member subscriptions
    "lifetime_revenue_earned": 1250.00,
    "merchandise_claimed": ["3_member_shirt", "5_member_bottle"],
    "custom_name": "The Reddy Runners",
    "custom_logo_url": "https://..."
  },

  "stats": {
    "total_members": 15,
    "active_members_this_week": 12,
    "total_challenges_completed": 450,
    "total_distance_km": 5000
  },

  "invite_codes": [
    {"code": "FAMILY_ABC123", "uses": 3, "max_uses": 10, "expires_at": timestamp}
  ]
}
```

---

### 4. `/friendships/{friendship_id}`
Bidirectional friendships.

```json
{
  "user1_id": "user_123",
  "user2_id": "user_456",
  "created_at": timestamp,
  "status": "active"
}
```

---

### 5. `/users/{user_id}/friends/{friend_id}` (subcollection)
User's friend list for fast queries.

```json
{
  "friend_id": "user_456",
  "name": "Jane Doe",
  "photo_url": "https://...",
  "added_at": timestamp
}
```

---

### 6. `/friend_requests/{request_id}`
Pending friend requests.

```json
{
  "from_user_id": "user_123",
  "from_user_name": "John Doe",
  "to_user_id": "user_456",
  "status": "pending | accepted | rejected",
  "created_at": timestamp,
  "responded_at": timestamp
}
```

---

### 7. `/communities/{community_id}`
Public/private communities like Strava clubs.

```json
{
  "name": "SF Bay Area Runners",
  "description": "Weekly group runs in SF",
  "photo_url": "https://...",
  "banner_url": "https://...",

  "founder_id": "user_123", // Community founder
  "created_at": timestamp,

  "type": "public | private",
  "sport_type": "running | cycling | mixed",

  "member_count": 1250,
  "active_members_this_week": 450,

  "monetization": {
    "is_premium": true,
    "monthly_fee": 9.99,
    "founder_revenue_share": 60 // Founder gets 60%
  },

  "founder_benefits": {
    "founder_badge": true,
    "verified": true, // 1000+ members
    "lifetime_revenue_earned": 5000.00,
    "merchandise_unlocked": ["tier1_stickers", "tier2_shirt", "tier3_bottle"],
    "custom_branding": {
      "logo_url": "https://...",
      "primary_color": "#FF5733",
      "banner_url": "https://..."
    }
  },

  "stats": {
    "total_challenges_created": 50,
    "total_participants": 5000,
    "total_distance_km": 50000
  }
}
```

---

### 8. `/communities/{community_id}/members/{user_id}` (subcollection)
Community members.

```json
{
  "user_id": "user_123",
  "joined_at": timestamp,
  "role": "admin | moderator | member",
  "subscription_status": "active | expired" // If premium community
}
```

---

### 9. `/creators/{creator_id}`
Creator profiles with monetization stats.

```json
{
  "user_id": "user_123",
  "creator_level": "beginner | rising | established | elite",
  "verified": true,

  "stats": {
    "total_sales": 1250,
    "total_earnings": 8750.00,
    "followers": 1500,
    "average_rating": 4.8,
    "total_reviews": 420
  },

  "content_created": {
    "challenges": 50,
    "programs": 10,
    "routes": 25
  },

  "revenue_share_percentage": 85, // Based on creator level

  "payout_info": {
    "stripe_account_id": "acct_xxx",
    "pending_payout": 450.00,
    "last_payout": {
      "amount": 1000.00,
      "date": timestamp
    }
  },

  "milestones_achieved": [
    {"type": "100_sales", "achieved_at": timestamp},
    {"type": "1000_sales", "achieved_at": timestamp}
  ]
}
```

---

### 10. `/marketplace/challenges/{challenge_id}`
Premium challenges for sale.

```json
{
  "creator_id": "user_123",
  "creator_name": "John Doe",

  "title": "Golden Gate 10K Challenge",
  "description": "...",
  "difficulty": "intermediate",
  "estimated_time": 60, // minutes

  "price": 5.00, // 0 for free
  "currency": "USD",

  "stats": {
    "total_purchases": 250,
    "total_revenue": 1250.00,
    "average_rating": 4.7,
    "review_count": 85
  },

  "challenge_data": {
    "type": "race | zone_control | scavenger_hunt",
    "zone_center": GeoPoint(37.7749, -122.4194),
    "zone_radius_meters": 5000,
    "rules": {...}
  },

  "tags": ["running", "scenic", "intermediate"],
  "featured": false,
  "created_at": timestamp,
  "updated_at": timestamp
}
```

---

### 11. `/marketplace/purchases/{purchase_id}`
User purchases from marketplace.

```json
{
  "buyer_id": "user_456",
  "challenge_id": "challenge_123",
  "creator_id": "user_123",

  "amount_paid": 5.00,
  "revenue_split": {
    "creator": 3.50, // 70%
    "platform": 1.00, // 20%
    "referrer": 0.50 // 10%
  },

  "purchased_at": timestamp,

  "stripe": {
    "payment_intent_id": "pi_xxx",
    "charge_id": "ch_xxx"
  }
}
```

---

### 12. `/referrals/{referral_code}`
Referral codes and tracking.

```json
{
  "referrer_id": "user_123",
  "referral_code": "JOHN_ABC123",

  "referred_users": [
    {"user_id": "user_456", "joined_at": timestamp, "subscription_tier": "pro_individual"},
    {"user_id": "user_789", "joined_at": timestamp, "subscription_tier": "pro_family"}
  ],

  "stats": {
    "total_referrals": 15,
    "active_referrals": 12, // Still subscribed
    "lifetime_earnings": 2500.00
  },

  "milestone_rewards_claimed": [
    {"milestone": 5, "reward": "3000_points", "claimed_at": timestamp},
    {"milestone": 10, "reward": "1_month_free_pro", "claimed_at": timestamp}
  ]
}
```

---

### 13. `/earnings/{user_id}`
Consolidated earnings tracking.

```json
{
  "user_id": "user_123",

  "breakdown": {
    "creator_revenue": 8750.00,
    "family_founder_revenue": 125.50,
    "community_founder_revenue": 500.00,
    "referral_revenue": 2500.00
  },

  "total_lifetime_earnings": 11875.50,
  "pending_payout": 450.00,

  "payout_history": [
    {"amount": 1000.00, "date": timestamp, "status": "completed", "stripe_transfer_id": "tr_xxx"},
    {"amount": 500.00, "date": timestamp, "status": "pending"}
  ],

  "payout_settings": {
    "auto_payout_enabled": true,
    "minimum_payout": 50.00,
    "stripe_account_id": "acct_xxx"
  }
}
```

---

### 14. `/founder_rewards/{user_id}`
Founder-specific benefits tracking.

```json
{
  "user_id": "user_123",

  "family_founder": {
    "subscription_id": "sub_xxx",
    "total_members": 15,
    "revenue_earned": 1250.00,
    "merchandise_claimed": ["3_member_shirt", "10_member_bottle"],
    "milestones_achieved": [3, 5, 10]
  },

  "community_founder": [
    {
      "community_id": "comm_123",
      "community_name": "SF Runners",
      "member_count": 1250,
      "revenue_earned": 5000.00,
      "verified": true,
      "merchandise_unlocked": ["tier1", "tier2", "tier3"]
    }
  ]
}
```

---

## Firebase Realtime Database Structure

### 1. `/live_challenges/{challenge_id}/participants/{user_id}`
Real-time participant positions during challenge.

```json
{
  "lat": 37.7749,
  "lon": -122.4194,
  "speed": 2.5, // m/s
  "distance": 1500, // meters
  "rank": 3,
  "last_update": 1640995200000 // milliseconds
}
```

---

### 2. `/leaderboards/{type}/{id}/{timeframe}/{user_id}`
Real-time leaderboards.

**Types:** `global`, `family`, `friends`, `community`
**Timeframes:** `weekly`, `monthly`, `all_time`

```json
{
  "user_id": "user_123",
  "name": "John Doe",
  "photo_url": "https://...",
  "points": 15000,
  "rank": 5,
  "last_updated": 1640995200000
}
```

**Examples:**
- `/leaderboards/global/all/weekly/user_123`
- `/leaderboards/family/sub_xxx/monthly/user_456`
- `/leaderboards/friends/user_123/all_time/user_789`
- `/leaderboards/community/comm_456/weekly/user_123`

---

### 3. `/notifications/{user_id}/{notification_id}`
Real-time notifications.

```json
{
  "type": "friend_request | challenge_invite | badge_earned",
  "title": "New Friend Request",
  "body": "Jane Doe sent you a friend request",
  "data": {"request_id": "req_123"},
  "timestamp": 1640995200000,
  "read": false
}
```

---

### 4. `/activity_feed/{user_id}/{activity_id}`
Real-time activity feed.

```json
{
  "user_id": "user_456",
  "user_name": "Jane Doe",
  "type": "challenge_completed | badge_earned | milestone_reached",
  "data": {
    "challenge_title": "Golden Gate 10K",
    "distance_km": 10.5,
    "time_minutes": 55
  },
  "timestamp": 1640995200000
}
```

---

## Revenue Models

### Creator Revenue Split
```python
FREE_USER = {
    "creator": 0%,
    "platform": 0%
}

PRO_INDIVIDUAL = {
    "creator": 70%,
    "platform": 20%,
    "referrer": 10%
}

PRO_FAMILY = {
    "creator": 75%,
    "platform": 15%,
    "referrer": 10%
}

CREATOR_PRO = {
    "creator": 85%,
    "platform": 10%,
    "referrer": 5%
}
```

### Family Founder Revenue
- **Base:** 15% of all family member subscriptions
- **Activity Bonus:** 10% of points earned by family (converted to revenue)
- **Milestones:** Free merchandise at 3, 5, 10, 20, 50, 100 members

### Community Founder Revenue
```python
FREE_COMMUNITY = {
    "founder": 0%,
    "max_members": 100
}

PREMIUM_COMMUNITY = {
    "founder": 60%,
    "platform": 30%,
    "referrer": 10%,
    "max_members": "unlimited"
}
```

---

## Indexes Required

**Firestore Composite Indexes:**

1. `users`: `(subscription.tier, stats.points_balance DESC)`
2. `communities`: `(type, member_count DESC)`
3. `marketplace/challenges`: `(price, stats.total_purchases DESC)`
4. `friendships`: `(user1_id, status)` + `(user2_id, status)`
5. `friend_requests`: `(to_user_id, status)`
6. `marketplace/purchases`: `(buyer_id, purchased_at DESC)`
7. `referrals`: `(referrer_id, stats.total_referrals DESC)`

---

## Summary

**Total Collections:** 14 Firestore + 4 Realtime DB structures
**Key Features:**
- ✅ Unlimited family members
- ✅ Friends with bidirectional relationships
- ✅ Public/private communities
- ✅ Creator marketplace with 70-90% revenue share
- ✅ Founder benefits (family & community)
- ✅ Two-sided referral rewards
- ✅ Real-time leaderboards (4 types × 3 timeframes = 12 leaderboards per user)
- ✅ Multi-tier subscriptions (Free, Pro Individual, Pro Family, Creator Pro)
- ✅ Comprehensive earnings tracking
