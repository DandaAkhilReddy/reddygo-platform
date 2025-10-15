# ReddyGo ML/AI Showcase PWA

A Vue.js Progressive Web App showcasing machine learning and AI capabilities built with PyTorch, TensorFlow, Qdrant, and LangChain.

## 🚀 Features

- **PyTorch LSTM Predictor**: Predict workout performance using deep learning
- **TensorFlow Recommender**: Get personalized challenge recommendations
- **Qdrant Vector Search**: Semantic search across workouts and memories
- **LangChain AI Coach**: Multi-agent AI coaching system
- **Cost Analysis**: Compare local LLaMA vs cloud GPT-4 costs

## 🛠️ Tech Stack

### Frontend
- **Vue 3.5.22** - Progressive JavaScript framework
- **Vite 7.1.7** - Next-generation frontend tooling
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS 3** - Utility-first CSS framework
- **Pinia** - Vue state management
- **Vue Router** - Official routing library

### Authentication
- **Firebase** - Authentication and user management

### ML/AI Integration
- **Axios** - HTTP client for API calls
- **Chart.js** - Data visualization
- **Highlight.js** - Code syntax highlighting

## 📦 Installation

### Prerequisites
- Node.js 18+ and npm
- Firebase project (for authentication)
- Backend API running (FastAPI with ML models)

### Setup Steps

1. **Install dependencies:**
   ```bash
   cd pwa-vue
   npm install
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your Firebase credentials:
   ```env
   VITE_API_URL=http://localhost:8080
   VITE_FIREBASE_API_KEY=your_key
   VITE_FIREBASE_AUTH_DOMAIN=your_domain
   VITE_FIREBASE_PROJECT_ID=your_project_id
   # ... other Firebase config
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```

   App will be available at `http://localhost:5173`

4. **Build for production:**
   ```bash
   npm run build
   ```

## 📱 PWA Setup (iPhone)

### Test on iPhone:

1. Make sure your dev server is accessible on your local network
2. Open Safari on iPhone and navigate to your IP address (e.g., `http://192.168.1.100:5173`)
3. Tap the Share button → "Add to Home Screen"
4. The app will install as a PWA with offline capabilities

## 🏗️ Project Structure

```
pwa-vue/
├── src/
│   ├── views/              # Page components
│   │   ├── DashboardView.vue
│   │   ├── LoginView.vue
│   │   ├── PytorchDemoView.vue
│   │   ├── TensorflowDemoView.vue
│   │   ├── QdrantDemoView.vue
│   │   ├── LangchainChatView.vue
│   │   └── CostAnalysisView.vue
│   │
│   ├── components/         # Reusable components
│   │   └── ui/            # UI primitives
│   │       ├── Button.vue
│   │       ├── Card.vue
│   │       ├── Input.vue
│   │       ├── Badge.vue
│   │       └── LoadingSpinner.vue
│   │
│   ├── stores/            # Pinia state management
│   │   ├── auth.ts
│   │   ├── ml-models.ts
│   │   └── chat.ts
│   │
│   ├── services/          # API clients
│   │   ├── api.ts
│   │   ├── firebase.ts
│   │   ├── pytorch-api.ts
│   │   ├── tensorflow-api.ts
│   │   ├── qdrant-api.ts
│   │   └── langchain-api.ts
│   │
│   ├── types/             # TypeScript types
│   │   ├── models.ts
│   │   ├── api.ts
│   │   └── user.ts
│   │
│   ├── router/            # Vue Router config
│   │   └── index.ts
│   │
│   ├── App.vue            # Root component
│   ├── main.ts            # App entry point
│   └── style.css          # Global styles (Tailwind)
│
├── public/                # Static assets
├── vite.config.ts         # Vite configuration
├── tailwind.config.js     # Tailwind CSS config
└── package.json           # Dependencies
```

## 🔐 Authentication

The app uses Firebase Authentication with email/password login:

1. **Login Page**: `/login`
2. **Protected Routes**: All demo pages require authentication
3. **Auth Guard**: Implemented in Vue Router
4. **Token Management**: Firebase ID tokens stored in localStorage
5. **API Integration**: Tokens automatically sent with API requests

## 🚀 Next Steps

1. Configure Firebase credentials in `.env`
2. Start the backend API (`cd backend && python main.py`)
3. Run the dev server (`npm run dev`)
4. Test on your iPhone by adding to home screen

---

Built with ❤️ for learning ML/AI concepts
