"""
ReddyGo - LangChain Multi-Agent Orchestrator
Connects all AI tools: Mem0, PyTorch, TensorFlow, LLaMA, GPT-4, Qdrant

Learning Focus:
- LangChain agents and tools
- Multi-step reasoning
- Tool selection and routing
- Cost optimization (local vs cloud LLMs)
"""

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from typing import List, Dict, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector.qdrant_client import ReddyGoVectorDB
# from ml.pytorch.performance_predictor import PerformancePredictor
# from ml.tensorflow.recommender import RecommenderSystem


class ReddyGoAICoach:
    """
    Multi-agent AI coach that orchestrates all ML/AI tools

    Tools Available:
    1. Memory Search - Search user memories (Qdrant)
    2. Workout Search - Find similar workouts (Qdrant)
    3. Exercise Search - Search exercise library (Qdrant)
    4. Performance Prediction - Predict next PR (PyTorch)
    5. Challenge Recommendation - Suggest challenges (TensorFlow)
    6. Local LLM - Use LLaMA for simple queries (cost-free)
    7. Cloud LLM - Use GPT-4 for complex reasoning

    Cost Optimization Strategy:
    - Simple queries → LLaMA (local, free)
    - Complex reasoning → GPT-4o-mini (cloud, $0.15/1M tokens)
    - Critical decisions → GPT-4 (cloud, $5/1M tokens)
    """

    def __init__(self,
                 openai_api_key: str,
                 use_local_llm: bool = True,
                 vector_db: Optional[ReddyGoVectorDB] = None):
        """
        Args:
            openai_api_key: OpenAI API key
            use_local_llm: Whether to use local LLaMA via Ollama
            vector_db: Vector database instance (if None, will create in-memory)
        """
        self.openai_api_key = openai_api_key
        self.use_local_llm = use_local_llm

        # Initialize vector DB
        if vector_db is None:
            self.vector_db = ReddyGoVectorDB(use_memory=True)
            self.vector_db.setup_collections()
        else:
            self.vector_db = vector_db

        # Initialize LLMs
        self.local_llm = None
        if use_local_llm:
            try:
                self.local_llm = Ollama(model="llama3.1:8b", temperature=0.7)
                print("Local LLaMA initialized via Ollama")
            except Exception as e:
                print(f"Warning: Could not initialize local LLM: {e}")
                print("Will use cloud LLM only")

        # Cloud LLM (GPT-4o-mini for cost efficiency)
        self.cloud_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=openai_api_key
        )

        # Initialize tools
        self.tools = self._create_tools()

        # Initialize memory for conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.cloud_llm,  # Use cloud LLM for agent orchestration
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            max_iterations=5
        )

        print(f"AI Coach initialized with {len(self.tools)} tools")

    def _create_tools(self) -> List[Tool]:
        """
        Create all available tools for the agent
        """
        tools = []

        # Tool 1: Memory Search
        def memory_search_tool(query_and_user: str) -> str:
            """
            Search user memories. Input format: 'user_id|query'
            Example: 'user_123|knee pain'
            """
            try:
                user_id, query = query_and_user.split("|", 1)
                memories = self.vector_db.search_memories(user_id.strip(), query.strip(), limit=3)
                if not memories:
                    return "No relevant memories found"
                result = "Relevant memories:\\n"
                for mem in memories:
                    result += f"- {mem['text']} (type: {mem['type']}, relevance: {mem['relevance']:.2f})\\n"
                return result
            except Exception as e:
                return f"Error searching memories: {str(e)}"

        tools.append(Tool(
            name="MemorySearch",
            func=memory_search_tool,
            description="Search user memories, preferences, injuries, and goals. Input: 'user_id|query'. Use this to personalize advice."
        ))

        # Tool 2: Workout Search
        def workout_search_tool(query_and_user: str) -> str:
            """
            Search similar workouts. Input: 'user_id|query'
            Example: 'user_123|long distance runs'
            """
            try:
                user_id, query = query_and_user.split("|", 1)
                workouts = self.vector_db.search_similar_workouts(query.strip(), user_id.strip(), limit=3)
                if not workouts:
                    return "No similar workouts found"
                result = "Similar workouts:\\n"
                for workout in workouts:
                    result += f"- {workout['description']} ({workout['distance_km']}km, {workout['pace_min_per_km']:.2f} min/km, similarity: {workout['similarity']:.2f})\\n"
                return result
            except Exception as e:
                return f"Error searching workouts: {str(e)}"

        tools.append(Tool(
            name="WorkoutSearch",
            func=workout_search_tool,
            description="Find similar past workouts. Input: 'user_id|query'. Use this to reference past performance."
        ))

        # Tool 3: Exercise Search
        def exercise_search_tool(query: str) -> str:
            """
            Search exercise library. Input: natural language query
            Example: 'leg exercises for runners'
            """
            try:
                exercises = self.vector_db.search_exercises(query, limit=5)
                if not exercises:
                    return "No exercises found"
                result = "Recommended exercises:\\n"
                for exercise in exercises:
                    result += f"- {exercise['name']}: {exercise['description']}\\n"
                    result += f"  Difficulty: {exercise['difficulty']}, Muscles: {', '.join(exercise['muscle_groups'])}\\n"
                return result
            except Exception as e:
                return f"Error searching exercises: {str(e)}"

        tools.append(Tool(
            name="ExerciseSearch",
            func=exercise_search_tool,
            description="Search exercise library for specific movements or muscle groups. Input: natural language query."
        ))

        # Tool 4: Simple Query Handler (Local LLM)
        if self.local_llm:
            def local_llm_tool(query: str) -> str:
                """
                Handle simple questions with local LLaMA (cost-free)
                """
                try:
                    response = self.local_llm.invoke(query)
                    return response
                except Exception as e:
                    return f"Error with local LLM: {str(e)}"

            tools.append(Tool(
                name="LocalLLM",
                func=local_llm_tool,
                description="Answer simple, general fitness questions using local LLaMA (free, no API cost). Use for basic advice."
            ))

        # Tool 5: Calculate Training Zones
        def calculate_zones_tool(max_hr: str) -> str:
            """
            Calculate heart rate training zones
            Input: max heart rate as string (e.g., "190")
            """
            try:
                max_hr_val = int(max_hr)
                zones = {
                    "Zone 1 (Recovery)": (int(max_hr_val * 0.50), int(max_hr_val * 0.60)),
                    "Zone 2 (Base)": (int(max_hr_val * 0.60), int(max_hr_val * 0.70)),
                    "Zone 3 (Tempo)": (int(max_hr_val * 0.70), int(max_hr_val * 0.80)),
                    "Zone 4 (Threshold)": (int(max_hr_val * 0.80), int(max_hr_val * 0.90)),
                    "Zone 5 (Max)": (int(max_hr_val * 0.90), max_hr_val)
                }
                result = f"Training zones for max HR {max_hr_val}:\\n"
                for zone_name, (low, high) in zones.items():
                    result += f"{zone_name}: {low}-{high} bpm\\n"
                return result
            except:
                return "Error: Provide max heart rate as a number"

        tools.append(Tool(
            name="CalculateZones",
            func=calculate_zones_tool,
            description="Calculate heart rate training zones. Input: max heart rate (number)."
        ))

        return tools

    def ask(self, user_id: str, query: str) -> Dict:
        """
        Main interface: User asks a question, agent figures out which tools to use

        Args:
            user_id: User ID
            query: User's question

        Returns:
            Dict with:
                - response: str (agent's answer)
                - tools_used: List[str] (which tools were called)
                - cost_estimate: float (estimated API cost)
        """
        # Add user context to query
        full_query = f"User ID: {user_id}\\n\\nQuestion: {query}"

        # Run agent
        try:
            response = self.agent.run(full_query)

            return {
                "response": response,
                "tools_used": [],  # TODO: Extract from agent execution
                "cost_estimate": 0.001,  # Approximate
                "model_used": "gpt-4o-mini"
            }
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "tools_used": [],
                "cost_estimate": 0.0,
                "model_used": "error"
            }

    def add_memory(self, user_id: str, memory_text: str, memory_type: str = "preference"):
        """
        Add a memory about the user
        """
        self.vector_db.add_memory(user_id, memory_text, memory_type)

    def add_workout(self, user_id: str, workout_data: Dict):
        """
        Add a workout to the database
        """
        self.vector_db.add_workout(user_id, workout_data)


# Example usage
if __name__ == "__main__":
    """
    Example: Multi-agent coaching with tool use
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Initialize coach
    coach = ReddyGoAICoach(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        use_local_llm=False  # Set to True if Ollama is running
    )

    # Add some test data
    user_id = "test_user_123"

    # Add memories
    coach.add_memory(user_id, "User has knee pain when running downhill", "injury")
    coach.add_memory(user_id, "Prefers morning workouts before 8 AM", "preference")
    coach.add_memory(user_id, "Goal is to run sub-25 minute 5km", "goal")

    # Add workouts
    workouts = [
        {
            "description": "Easy 5km morning run",
            "distance_km": 5.0,
            "duration_min": 28.5,
            "pace_min_per_km": 5.7,
            "type": "run",
            "date": "2025-10-14"
        },
        {
            "description": "Hard 10km with hills",
            "distance_km": 10.0,
            "duration_min": 62.0,
            "pace_min_per_km": 6.2,
            "type": "run",
            "date": "2025-10-13"
        }
    ]

    for workout in workouts:
        coach.add_workout(user_id, workout)

    # Ask questions (agent will use tools automatically)
    questions = [
        "What exercises should I do to improve my running? Consider my knee issues.",
        "Looking at my past runs, am I ready for a 10km race?",
        "Create a workout plan for next week to help me reach my 5km goal. Remember my morning preference.",
    ]

    for question in questions:
        print(f"\\n{'='*80}")
        print(f"Q: {question}")
        print(f"{'='*80}")

        result = coach.ask(user_id, question)

        print(f"\\nA: {result['response']}")
        print(f"\\nTools used: {result['tools_used']}")
        print(f"Cost: ${result['cost_estimate']:.4f}")
        print(f"Model: {result['model_used']}")
