"""
ReddyGo - Qdrant Vector Database Client
Semantic search for workouts, memories, and exercise library

Learning Focus:
- Vector databases and embeddings
- Similarity search
- RAG (Retrieval Augmented Generation)
- Hybrid search (vector + metadata)
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import numpy as np
import uuid


class ReddyGoVectorDB:
    """
    Vector database for semantic search across ReddyGo data

    Collections:
    - workouts: Workout descriptions and metadata
    - memories: User preferences and coach memories
    - exercises: Exercise library with instructions
    """

    def __init__(self, host: str = "localhost", port: int = 6333, use_memory: bool = False):
        """
        Args:
            host: Qdrant server host
            port: Qdrant server port
            use_memory: If True, use in-memory mode (for testing)
        """
        if use_memory:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(host=host, port=port)

        # Initialize embedding model
        # Using all-MiniLM-L6-v2: Fast, 384-dim embeddings, good for semantic search
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384

        print(f"Vector DB initialized (dim={self.embedding_dim})")

    def setup_collections(self):
        """
        Create all collections if they don't exist
        """
        collections = {
            "workouts": "Workout descriptions and stats for similarity search",
            "memories": "User preferences, injuries, goals remembered by AI coach",
            "exercises": "Exercise library with descriptions and instructions"
        }

        for collection_name, description in collections.items():
            # Check if collection exists
            try:
                self.client.get_collection(collection_name)
                print(f"Collection '{collection_name}' already exists")
            except:
                # Create collection
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE  # Cosine similarity for semantic search
                    )
                )
                print(f"Created collection: {collection_name} - {description}")

    # ============================================
    # WORKOUT COLLECTION
    # ============================================

    def add_workout(self, user_id: str, workout_data: Dict) -> str:
        """
        Add a workout to the vector database

        Args:
            user_id: User ID
            workout_data: Dict with keys:
                - description: str (e.g., "5km morning run, felt strong")
                - distance_km: float
                - duration_min: float
                - pace_min_per_km: float
                - type: str ("run", "bike", "swim", etc.)
                - date: str (ISO format)

        Returns:
            Workout ID (UUID)
        """
        workout_id = str(uuid.uuid4())

        # Create searchable text
        description = workout_data.get("description", "")
        workout_type = workout_data.get("type", "workout")
        text = f"{workout_type}: {description}"

        # Generate embedding
        embedding = self.embedding_model.encode(text).tolist()

        # Insert into Qdrant
        self.client.upsert(
            collection_name="workouts",
            points=[
                PointStruct(
                    id=workout_id,
                    vector=embedding,
                    payload={
                        "user_id": user_id,
                        "description": description,
                        "distance_km": workout_data.get("distance_km"),
                        "duration_min": workout_data.get("duration_min"),
                        "pace_min_per_km": workout_data.get("pace_min_per_km"),
                        "type": workout_type,
                        "date": workout_data.get("date"),
                        "text": text  # Store for retrieval
                    }
                )
            ]
        )

        print(f"Added workout {workout_id} for user {user_id}")
        return workout_id

    def search_similar_workouts(self, query: str, user_id: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """
        Find workouts similar to a query

        Args:
            query: Natural language query (e.g., "long distance runs over 10km")
            user_id: If provided, only search this user's workouts
            limit: Maximum number of results

        Returns:
            List of similar workouts with metadata and similarity scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Build filter if user_id provided
        query_filter = None
        if user_id:
            query_filter = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            )

        # Search
        results = self.client.search(
            collection_name="workouts",
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=limit
        )

        # Format results
        workouts = []
        for result in results:
            workouts.append({
                "workout_id": result.id,
                "similarity": result.score,
                **result.payload
            })

        return workouts

    # ============================================
    # MEMORY COLLECTION (for AI Coach)
    # ============================================

    def add_memory(self, user_id: str, memory_text: str, memory_type: str = "preference") -> str:
        """
        Store a memory about the user for AI coach

        Args:
            user_id: User ID
            memory_text: Memory content (e.g., "User has knee pain, avoid squats")
            memory_type: Type of memory ("preference", "injury", "goal", "feedback")

        Returns:
            Memory ID
        """
        memory_id = str(uuid.uuid4())

        # Generate embedding
        embedding = self.embedding_model.encode(memory_text).tolist()

        # Insert
        self.client.upsert(
            collection_name="memories",
            points=[
                PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload={
                        "user_id": user_id,
                        "text": memory_text,
                        "type": memory_type
                    }
                )
            ]
        )

        print(f"Added memory for user {user_id}: {memory_text}")
        return memory_id

    def search_memories(self, user_id: str, query: str, limit: int = 5) -> List[Dict]:
        """
        Search user memories relevant to a query

        Args:
            user_id: User ID
            query: Query (e.g., "knee pain" or "morning workouts")
            limit: Max results

        Returns:
            List of relevant memories
        """
        query_embedding = self.embedding_model.encode(query).tolist()

        # Filter by user_id
        query_filter = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )

        results = self.client.search(
            collection_name="memories",
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=limit
        )

        memories = [
            {
                "memory_id": result.id,
                "relevance": result.score,
                **result.payload
            }
            for result in results
        ]

        return memories

    # ============================================
    # EXERCISE LIBRARY
    # ============================================

    def add_exercise(self, exercise_data: Dict) -> str:
        """
        Add exercise to library

        Args:
            exercise_data: Dict with:
                - name: str
                - description: str
                - instructions: str
                - muscle_groups: List[str]
                - difficulty: str ("beginner", "intermediate", "advanced")
                - equipment: List[str]

        Returns:
            Exercise ID
        """
        exercise_id = str(uuid.uuid4())

        # Create searchable text
        text = f"{exercise_data['name']}: {exercise_data['description']}. {exercise_data['instructions']}"

        # Generate embedding
        embedding = self.embedding_model.encode(text).tolist()

        # Insert
        self.client.upsert(
            collection_name="exercises",
            points=[
                PointStruct(
                    id=exercise_id,
                    vector=embedding,
                    payload={
                        **exercise_data,
                        "text": text
                    }
                )
            ]
        )

        print(f"Added exercise: {exercise_data['name']}")
        return exercise_id

    def search_exercises(self, query: str, difficulty: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """
        Search exercise library

        Args:
            query: Natural language query (e.g., "leg exercises for runners")
            difficulty: Filter by difficulty level
            limit: Max results

        Returns:
            List of exercises
        """
        query_embedding = self.embedding_model.encode(query).tolist()

        # Build filter
        query_filter = None
        if difficulty:
            query_filter = Filter(
                must=[FieldCondition(key="difficulty", match=MatchValue(value=difficulty))]
            )

        results = self.client.search(
            collection_name="exercises",
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=limit
        )

        exercises = [
            {
                "exercise_id": result.id,
                "relevance": result.score,
                **result.payload
            }
            for result in results
        ]

        return exercises


# Example usage
if __name__ == "__main__":
    """
    Example: Set up vector DB and test semantic search
    """
    # Initialize (in-memory for testing)
    vector_db = ReddyGoVectorDB(use_memory=True)
    vector_db.setup_collections()

    # Add some workouts
    user_id = "user_123"

    workouts = [
        {
            "description": "Easy 5km morning run, sunny weather",
            "distance_km": 5.0,
            "duration_min": 28.5,
            "pace_min_per_km": 5.7,
            "type": "run",
            "date": "2025-10-14"
        },
        {
            "description": "Hard 10km run with hills, felt exhausted",
            "distance_km": 10.0,
            "duration_min": 62.0,
            "pace_min_per_km": 6.2,
            "type": "run",
            "date": "2025-10-13"
        },
        {
            "description": "Recovery bike ride, 20km easy pace",
            "distance_km": 20.0,
            "duration_min": 55.0,
            "pace_min_per_km": 2.75,
            "type": "bike",
            "date": "2025-10-12"
        }
    ]

    for workout in workouts:
        vector_db.add_workout(user_id, workout)

    # Search for similar workouts
    query = "long distance running"
    print(f"\nSearching for: '{query}'")
    results = vector_db.search_similar_workouts(query, user_id=user_id, limit=3)

    for i, workout in enumerate(results, 1):
        print(f"\n{i}. Similarity: {workout['similarity']:.4f}")
        print(f"   {workout['description']}")
        print(f"   Distance: {workout['distance_km']}km, Pace: {workout['pace_min_per_km']:.2f} min/km")

    # Add memory
    vector_db.add_memory(user_id, "User prefers morning workouts", "preference")
    vector_db.add_memory(user_id, "Has knee pain when running downhill", "injury")
    vector_db.add_memory(user_id, "Goal is to run a sub-25 minute 5km", "goal")

    # Search memories
    print("\n\nSearching memories for: 'knee problems'")
    memories = vector_db.search_memories(user_id, "knee problems", limit=2)
    for memory in memories:
        print(f"  - {memory['text']} (relevance: {memory['relevance']:.4f})")

    # Add exercises
    exercises_data = [
        {
            "name": "Squats",
            "description": "Basic leg strengthening exercise",
            "instructions": "Stand with feet shoulder-width apart, lower body by bending knees",
            "muscle_groups": ["quads", "glutes", "hamstrings"],
            "difficulty": "beginner",
            "equipment": []
        },
        {
            "name": "Lunges",
            "description": "Single-leg strengthening exercise",
            "instructions": "Step forward with one leg, lower hips until both knees bent at 90°",
            "muscle_groups": ["quads", "glutes"],
            "difficulty": "beginner",
            "equipment": []
        }
    ]

    for exercise in exercises_data:
        vector_db.add_exercise(exercise)

    # Search exercises
    print("\n\nSearching exercises for: 'leg strength for runners'")
    exercise_results = vector_db.search_exercises("leg strength for runners", limit=2)
    for exercise in exercise_results:
        print(f"  - {exercise['name']}: {exercise['description']}")
        print(f"    Difficulty: {exercise['difficulty']}, Muscles: {', '.join(exercise['muscle_groups'])}")
