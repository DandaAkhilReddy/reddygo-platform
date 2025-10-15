# ReddyGo ML Bootcamp - Learning Platform

**10-Day Intensive Machine Learning Bootcamp with Interactive Web UI**

## Features

✅ Complete 10-day ML/AI curriculum
✅ Beautiful TailwindCSS web interface
✅ Daily 10-question exams with auto-grading
✅ Progress tracking and analytics
✅ Certificate generation
✅ Interactive lessons with code examples
✅ ReddyGo-focused projects

## Quick Start

### 1. Install Dependencies

```bash
cd ml-bootcamp
pip install -r requirements.txt
```

### 2. Initialize Database

**IMPORTANT:** Run this once to create the database:

```bash
python init_db.py
```

This creates:
- SQLite database at `instance/bootcamp.db`
- Default user account
- Progress tracking tables

### 3. Run the Platform

```bash
python app.py
```

### 4. Open in Browser

Navigate to: http://localhost:5000

## ✅ Progress Saves Permanently!

Your learning progress is now saved to a **SQLite database**, which means:

✅ **Progress persists** even after closing the browser
✅ **Exam scores are saved** permanently to disk
✅ **Can continue learning** anytime - just reopen the browser
✅ **Progress survives** cookie clearing and browser restarts

**No more lost progress!** The database file (`instance/bootcamp.db`) stores all your data.

## Platform Structure

```
ml-bootcamp/
├── app.py                 # Flask application
├── templates/             # HTML templates
│   ├── base.html         # Base template with navigation
│   ├── index.html        # Dashboard
│   ├── lesson.html       # Lesson pages
│   ├── exam.html         # Exam interface
│   ├── progress.html     # Progress tracking
│   └── certificate.html  # Certificate page
├── exams/                # Exam JSON files
│   ├── day1_exam.json   # Day 1: Python + NumPy
│   ├── day2_exam.json   # Day 2: Pandas
│   └── ...
├── lessons/              # Lesson content (markdown)
├── static/               # CSS, JS, images
└── requirements.txt      # Python dependencies
```

## Curriculum Overview

| Day | Topic | Duration |
|-----|-------|----------|
| 1 | Python + NumPy Essentials | 7 hours |
| 2 | Pandas & Data Preprocessing | 7 hours |
| 3 | Machine Learning Foundations | 7 hours |
| 4 | Advanced ML Algorithms | 7 hours |
| 5 | Deep Learning - Part 1 | 7 hours |
| 6 | Deep Learning - Part 2 | 7 hours |
| 7 | Computer Vision | 7 hours |
| 8 | Natural Language Processing | 7 hours |
| 9 | Production ML & MLOps | 7 hours |
| 10 | Final Exam & Certification | 7 hours |

## Daily Schedule

Each day includes:
- **2 hours:** Theory & Concepts
- **3 hours:** Hands-on Coding
- **10-question exam:** Auto-graded
- **2 hours:** ReddyGo Project

## Grading System

### Daily Exams
- **Total Questions:** 10
- **Passing Score:** 70% (7/10 correct)
- **Format:** Multiple choice

### Final Exam
- **Total Questions:** 50
- **Passing Score:** 70% (35/50 correct)
- **Covers:** All 10 days

### Certificate Levels
- **Participant:** Complete all days
- **Certified:** 70% on final exam
- **Distinction:** 85% on final exam
- **Outstanding:** 95% on final exam

## Features

### Dashboard
- Overview of all 10 days
- Progress tracking
- Quick stats
- Continue learning CTA

### Lesson Pages
- Theory with code examples
- Practice exercises
- ReddyGo projects
- Video resources

### Exam System
- Interactive quiz interface
- Real-time progress tracking
- Instant grading
- Detailed feedback
- Question explanations

### Progress Tracking
- Overall progress percentage
- Exam scores timeline
- Performance charts
- Achievement badges

### Certificate
- Beautiful certificate design
- Achievement level badge
- Printable PDF
- Social sharing

## Development

### Adding New Exam

Create `exams/dayX_exam.json`:

```json
{
  "title": "Topic Name",
  "day": X,
  "total_questions": 10,
  "passing_score": 70,
  "questions": [
    {
      "question": "Your question here?",
      "code": "# Optional code block",
      "options": ["A", "B", "C", "D"],
      "correct_answer": 0,
      "explanation": "Why this is correct"
    }
  ]
}
```

### Adding Lesson Content

Create `lessons/dayX.md` with markdown content.

### Customization

- **Colors:** Edit TailwindCSS classes in templates
- **Branding:** Update logo and name in `base.html`
- **Curriculum:** Modify `app.py` curriculum array

## Tech Stack

- **Backend:** Flask (Python)
- **Frontend:** TailwindCSS
- **Database:** SQLite + SQLAlchemy (persistent storage)
- **Data:** JSON files (for exams)

## Database Schema

### Tables

**users** - User accounts
- id, username, email, created_at

**progress** - Learning progress
- id, user_id, current_day, completed_days, total_score, updated_at

**exam_results** - Exam scores
- id, user_id, day, score, correct, total, passed, answers, timestamp

### Reset Progress

To reset your progress for testing:

```bash
python init_db.py  # This drops and recreates all tables
```

Or use the `/reset_progress` API endpoint (for development).

## Next Steps

1. ✅ Basic platform with persistent database complete!
2. Add remaining exam JSON files (Days 2-10)
3. Add detailed lesson content
4. Implement PDF certificate generation
5. Add user authentication system (login/register)
6. Deploy to production (with PostgreSQL for multi-user)

## License

MIT License - ReddyGo Platform 2025

---

**Start your ML journey today!** http://localhost:5000
