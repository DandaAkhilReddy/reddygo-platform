# 🚀 ReddyGo ML Bootcamp - 10-Day Intensive Machine Learning Program

A comprehensive, interactive machine learning bootcamp platform built with Flask, featuring hands-on exercises, auto-graded exams, and persistent progress tracking.

## 🎯 Overview

This platform provides a structured 10-day journey from Python fundamentals to production ML, designed with a pedagogical approach for ReddyGo engineers to master ML/AI concepts through practice.

**Key Features:**
- 🏃 10 interactive coding exercises per day
- 📝 110 auto-graded exam questions across 10 days
- 💾 Persistent progress tracking with SQLite
- 🎨 Modern UI with TailwindCSS
- 🐍 Browser-based Python execution using Skulpt.js
- 📊 Real-time feedback and hints system
- 🎓 Automatic certificate generation on completion

## 📚 Curriculum

### **Day 1: Python + NumPy Fundamentals**
- Variables, data types, operators
- Control structures (if/else, loops)
- Functions and data structures
- NumPy arrays and operations
- 10 coding exercises + 10 exam questions

### **Day 2: Pandas & Data Preprocessing**
- DataFrames and Series
- Data cleaning and transformation
- Handling missing data
- Feature engineering basics
- 10 exam questions

### **Day 3: Machine Learning Foundations**
- Supervised vs Unsupervised Learning
- Train/test split and cross-validation
- Linear Regression, Logistic Regression
- Model evaluation metrics
- Scikit-learn fundamentals
- 10 exam questions

### **Day 4: Advanced ML Algorithms**
- Decision Trees
- Random Forests
- Gradient Boosting & XGBoost
- Hyperparameter tuning
- Feature importance
- 10 exam questions

### **Day 5: Deep Learning with PyTorch - Part 1**
- Neural network fundamentals
- PyTorch tensors and autograd
- Building and training networks
- Activation functions
- Backpropagation
- 10 exam questions

### **Day 6: Deep Learning Part 2 - CNNs & RNNs**
- Convolutional Neural Networks
- Pooling and feature maps
- Recurrent Neural Networks
- LSTMs and sequence modeling
- Dropout and batch normalization
- Transfer learning
- 10 exam questions

### **Day 7: Computer Vision**
- Image preprocessing
- Pre-trained models (ResNet, VGG)
- Object detection
- Semantic segmentation
- ImageNet and data augmentation
- 10 exam questions

### **Day 8: Natural Language Processing**
- Tokenization and embeddings
- Transformer architecture
- BERT and GPT models
- Hugging Face library
- Sentiment analysis and NER
- Attention mechanisms
- 10 exam questions

### **Day 9: Production ML & MLOps**
- Model serialization
- FastAPI deployment
- Docker containerization
- MLflow experiment tracking
- Model monitoring and drift detection
- CI/CD for ML
- 10 exam questions

### **Day 10: Final Comprehensive Exam**
- 50 questions covering all topics
- Passing score: 80%
- Certificate awarded on completion

## 🚀 Getting Started

### Prerequisites
- Python 3.13+
- pip package manager
- Modern web browser

### Installation

1. **Clone the repository**
```bash
cd C:/Users/akhil/reddygo-platform
cd ml-bootcamp
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize database**
```bash
python init_db.py
```

This creates:
- SQLite database at `instance/bootcamp.db`
- Default user account
- Progress tracking tables

### Running the Platform

1. **Start the Flask application**
```bash
python app.py
```

2. **Open your browser**
```
http://localhost:5000
```

3. **Start learning!**
   - View your progress dashboard
   - Navigate through daily lessons
   - Complete interactive exercises
   - Take exams and track scores

## 📁 Project Structure

```
ml-bootcamp/
├── app.py                      # Flask application entry point
├── models.py                   # SQLAlchemy database models
├── requirements.txt            # Python dependencies
├── instance/
│   └── bootcamp.db            # SQLite database (auto-created)
├── templates/
│   ├── index.html             # Dashboard homepage
│   ├── lesson.html            # Daily lesson page
│   ├── practice.html          # Interactive coding environment
│   ├── exam.html              # Exam interface
│   └── results.html           # Exam results page
├── practice/
│   ├── day1_exercises.py      # Day 1 coding exercises
│   └── ...                    # Days 2-10 (to be added)
└── exams/
    ├── day1_exam.json         # Day 1 exam questions
    ├── day2_exam.json         # Day 2 exam questions
    ├── ...
    └── day10_exam.json        # Final comprehensive exam
```

## 🎓 How It Works

### Interactive Practice System

1. **Exercise Structure**
Each exercise includes:
```python
{
    "id": 1,
    "title": "Distance Calculator",
    "description": "Calculate total distance and average",
    "difficulty": "Easy",
    "instructions": "Detailed problem description...",
    "starter_code": "# Initial code template",
    "solution": "# Complete solution",
    "expected_output": "Expected result format",
    "hints": ["Hint 1", "Hint 2", "Hint 3"],
    "explanation": "Learning objectives"
}
```

2. **Browser Python Execution**
- Uses Skulpt.js to run Python code in-browser
- No server-side execution needed
- Instant feedback on code output
- Safe sandboxed environment

3. **Progressive Hints**
- 3 hints per exercise
- Reveal one at a time
- Solution available after hints

### Exam System

1. **Exam Format**
```json
{
  "title": "Python + NumPy Fundamentals",
  "day": 1,
  "total_questions": 10,
  "passing_score": 70,
  "questions": [
    {
      "question": "Question text",
      "options": ["A", "B", "C", "D"],
      "correct_answer": 1,
      "explanation": "Why this is correct"
    }
  ]
}
```

2. **Auto-Grading**
- Immediate results after submission
- Detailed explanations for each answer
- Pass/fail determination (70% threshold)
- Progress saved to database

3. **Progress Tracking**
- Current day and completed days
- Exam scores and timestamps
- Overall completion percentage
- Certificate eligibility

## 💾 Database Schema

### User Model
```python
- id: Integer (Primary Key)
- username: String (unique)
- progress: Relationship → Progress
- exam_results: Relationship → ExamResult[]
```

### Progress Model
```python
- id: Integer (Primary Key)
- user_id: Foreign Key → User
- current_day: Integer (1-10)
- completed_days: JSON Array
```

### ExamResult Model
```python
- id: Integer (Primary Key)
- user_id: Foreign Key → User
- day: Integer (1-10)
- score: Float (0-100)
- correct: Integer
- total: Integer
- passed: Boolean
- timestamp: DateTime
```

## 🛠️ Technical Stack

**Backend:**
- Flask 3.0.0 - Web framework
- SQLAlchemy 2.0.44 - ORM and database
- SQLite - Embedded database

**Frontend:**
- TailwindCSS (CDN) - Styling
- Skulpt.js - Python interpreter in browser
- Vanilla JavaScript - Interactivity

**Python:**
- Python 3.13
- Jinja2 - Template engine

## 📊 Features in Detail

### 1. Progress Dashboard
- Visual progress cards for all 10 days
- Color-coded status (locked, current, completed)
- Quick stats: completed days, current day, total exams
- Direct navigation to lessons and exams

### 2. Interactive Coding Environment
- Syntax-highlighted code editor
- Run button for instant execution
- Output console with error handling
- Progressive hint system
- Solution viewer
- Reset code functionality

### 3. Exam Interface
- Clean multiple-choice format
- Real-time answer selection
- Timed completion (optional)
- Submit with confirmation
- Detailed results with explanations

### 4. Results & Analytics
- Score percentage and pass/fail status
- Question-by-question breakdown
- Correct answers highlighted
- Explanations for learning
- Retake option available

### 5. Certificate System
- Auto-generated on Day 10 completion
- Requires 80% on final exam
- Downloadable PDF (future enhancement)
- Shareable credential

## 🔧 Configuration

### Flask Configuration
```python
# app.py
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/bootcamp.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
```

### Exam Thresholds
```python
# Default passing scores
DAY_1_9_PASSING = 70  # 70% for daily exams
DAY_10_PASSING = 80   # 80% for final exam
```

## 🧪 Testing

### Test All Exams
```bash
cd ml-bootcamp
python -c "
import json
for day in range(1, 11):
    with open(f'exams/day{day}_exam.json') as f:
        exam = json.load(f)
        print(f'Day {day}: {exam[\"title\"]} - {len(exam[\"questions\"])} questions')
"
```

### Validate Database
```bash
python -c "
from app import app, db
with app.app_context():
    db.create_all()
    print('Database tables created successfully!')
"
```

## ✅ Completion Status

**Fully Implemented:**
- ✅ Complete 10-day curriculum defined
- ✅ All 10 exam files created (110 questions total)
- ✅ Interactive practice system (Day 1: 10 exercises)
- ✅ Database persistence with SQLAlchemy
- ✅ Progress tracking and analytics
- ✅ Auto-grading system
- ✅ Modern TailwindCSS UI
- ✅ Certificate generation logic

**In Progress:**
- ⏳ Practice exercises for Days 2-10
- ⏳ Detailed lesson content (markdown)
- ⏳ Jupyter notebook versions

## 🚧 Future Enhancements

### Planned Features
- [ ] PDF certificate generation
- [ ] Email notifications for progress
- [ ] Code quality checks (linting)
- [ ] Collaborative coding sessions
- [ ] Leaderboards and badges
- [ ] Mobile-responsive improvements
- [ ] Dark mode toggle
- [ ] Export progress reports
- [ ] ReddyGo-specific case studies
- [ ] Video tutorials integration

### Advanced Features
- [ ] AI code review feedback
- [ ] Personalized learning paths
- [ ] Adaptive difficulty
- [ ] Peer code review system
- [ ] Integration with ReddyGo production data
- [ ] Real-time multiplayer challenges

## 📖 Learning Methodology

This bootcamp follows evidence-based learning principles:

1. **Deliberate Practice**: Hands-on coding exercises before theory
2. **Immediate Feedback**: Instant validation of code and answers
3. **Spaced Repetition**: Progressive difficulty across days
4. **Active Recall**: Exam questions test understanding
5. **Scaffolding**: Hints system supports struggling learners
6. **Real-World Application**: ReddyGo-focused projects

## 🤝 Contributing

We welcome contributions! Areas for improvement:

1. **Content**: Additional exercises, better explanations
2. **Features**: UI improvements, new functionality
3. **Bug Fixes**: Report issues on GitHub
4. **Documentation**: Improve guides and examples

### Development Setup
```bash
# Fork and clone repo
git clone https://github.com/DandaAkhilReddy/reddygo-platform.git
cd reddygo-platform/ml-bootcamp

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
python app.py

# Commit and push
git add .
git commit -m "feat: your feature description"
git push origin feature/your-feature-name
```

## 📝 License

This project is part of the ReddyGo platform. Internal use only.

## 👥 Authors

Built with pedagogical expertise for ReddyGo engineers.

## 🙏 Acknowledgments

- **Skulpt.js** - Browser-based Python execution
- **TailwindCSS** - Modern utility-first CSS
- **Flask** - Lightweight web framework
- **SQLAlchemy** - Powerful ORM

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Contact the ReddyGo engineering team
- Check documentation at `/docs`

## 🎯 Success Metrics

**Complete Bootcamp Stats:**
- ✅ 10 days of structured content
- ✅ 110 exam questions (10 per day + 50 final)
- ✅ Day 1: 10 interactive coding exercises
- ⏳ Days 2-10: Exercises coming soon
- 🎓 Certificate on completion

## 🔥 Quick Start Guide

**First Time Users:**
1. Install: `pip install -r requirements.txt`
2. Initialize DB: `python init_db.py`
3. Start app: `python app.py`
4. Open http://localhost:5000
5. Click "Start Day 1"
6. Complete 10 coding exercises
7. Take the Day 1 exam (70% to pass)
8. Progress to Day 2
9. Repeat until Day 10
10. Pass final exam (80% threshold)
11. Receive certificate!

**Daily Workflow:**
```
1. Read lesson overview
2. Complete practice exercises (use hints if stuck)
3. Review solutions and explanations
4. Take the exam
5. Review incorrect answers
6. Move to next day (unlocked after passing exam)
```

## 📊 Exam Coverage

| Day | Topic | Questions | Pass % |
|-----|-------|-----------|--------|
| 1 | Python + NumPy | 10 | 70% |
| 2 | Pandas & Preprocessing | 10 | 70% |
| 3 | ML Foundations | 10 | 70% |
| 4 | Advanced ML | 10 | 70% |
| 5 | PyTorch Basics | 10 | 70% |
| 6 | CNNs & RNNs | 10 | 70% |
| 7 | Computer Vision | 10 | 70% |
| 8 | NLP | 10 | 70% |
| 9 | Production ML | 10 | 70% |
| 10 | Final Comprehensive | 50 | 80% |

**Total: 110 questions across all assessments**

---

**Built with ❤️ for ReddyGo Engineers**

*Master ML in 10 days. Ship AI features with confidence.*

Start your journey: http://localhost:5000
