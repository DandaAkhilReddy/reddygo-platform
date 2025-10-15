"""
ML Bootcamp - Interactive Learning Platform
10-Day Intensive Machine Learning Bootcamp with Daily Exams

Open-source platform for learning ML/AI from basics to production.
"""

from flask import Flask, render_template, request, jsonify, session
from datetime import datetime
import json
import os
import secrets
from models import db, User, Progress, ExamResult

# Load environment variables if .env exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use default values

app = Flask(__name__)

# Configuration from environment variables with sensible defaults
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.config['FLASK_ENV'] = os.getenv('FLASK_ENV', 'development')
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'True') == 'True'

# Database configuration
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.getenv('DATABASE_URL', f'sqlite:///{os.path.join(basedir, "instance", "bootcamp.db")}')
app.config['SQLALCHEMY_DATABASE_URI'] = db_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Application settings
app.config['APP_NAME'] = os.getenv('APP_NAME', 'ML Bootcamp')
app.config['MAX_GUEST_USERS'] = int(os.getenv('MAX_GUEST_USERS', 1000))

# Initialize database
db.init_app(app)

# Load exam data
def load_exam(day):
    """Load exam questions for specific day"""
    exam_path = f'exams/day{day}_exam.json'
    if os.path.exists(exam_path):
        with open(exam_path, 'r') as f:
            return json.load(f)
    return None

# Get or create guest user based on session
def get_current_user():
    """
    Get the current user based on session.
    Creates a unique guest user for each browser/session automatically.
    No authentication required - instant learning!
    """
    # Check if user already has a session ID
    if 'user_id' not in session:
        # Generate a unique guest user ID
        guest_id = f'guest_{secrets.token_hex(8)}'
        session['user_id'] = guest_id
        session.permanent = True  # Make session persistent

    user_id = session['user_id']

    # Try to find existing user
    user = User.query.filter_by(username=user_id).first()

    if not user:
        # Create new guest user
        user = User(username=user_id, email=f'{user_id}@guest.local')
        db.session.add(user)
        db.session.commit()

    return user

# Load user progress from database
def get_user_progress():
    """Get current user's progress from database"""
    user = get_current_user()
    progress_record = Progress.query.filter_by(user_id=user.id).first()

    if not progress_record:
        # Create new progress record
        progress_record = Progress(user_id=user.id, current_day=1, completed_days='[]')
        db.session.add(progress_record)
        db.session.commit()

    # Get all exam results for this user
    exam_results = ExamResult.query.filter_by(user_id=user.id).order_by(ExamResult.day).all()

    # Build exam_scores dictionary
    exam_scores = {}
    for result in exam_results:
        exam_scores[f'day{result.day}'] = {
            'score': result.score,
            'correct': result.correct,
            'total': result.total,
            'passed': result.passed,
            'timestamp': result.timestamp.isoformat()
        }

    return {
        'completed_days': progress_record.get_completed_days(),
        'exam_scores': exam_scores,
        'current_day': progress_record.current_day,
        'total_score': progress_record.total_score
    }

def save_user_progress(progress_data):
    """Save user progress to database"""
    user = get_current_user()
    progress_record = Progress.query.filter_by(user_id=user.id).first()

    if not progress_record:
        progress_record = Progress(user_id=user.id)
        db.session.add(progress_record)

    progress_record.current_day = progress_data.get('current_day', 1)
    progress_record.set_completed_days(progress_data.get('completed_days', []))
    progress_record.total_score = progress_data.get('total_score', 0.0)

    db.session.commit()

@app.route('/')
def index():
    """Main dashboard"""
    progress = get_user_progress()

    # Curriculum overview
    curriculum = [
        {
            'day': 1,
            'title': 'Python + NumPy Essentials',
            'topics': ['Variables & Data Types', 'NumPy Arrays', 'Vectorization'],
            'duration': '7 hours',
            'project': 'Distance Calculator'
        },
        {
            'day': 2,
            'title': 'Pandas & Data Preprocessing',
            'topics': ['DataFrames', 'Data Cleaning', 'Feature Engineering'],
            'duration': '7 hours',
            'project': 'User Data Analysis'
        },
        {
            'day': 3,
            'title': 'Machine Learning Foundations',
            'topics': ['ML Theory', 'scikit-learn', 'Model Evaluation'],
            'duration': '7 hours',
            'project': 'Workout Predictor'
        },
        {
            'day': 4,
            'title': 'Advanced ML Algorithms',
            'topics': ['Decision Trees', 'Random Forests', 'XGBoost'],
            'duration': '7 hours',
            'project': 'User Segmentation'
        },
        {
            'day': 5,
            'title': 'Deep Learning - Part 1',
            'topics': ['Neural Networks', 'PyTorch Basics', 'Training Loops'],
            'duration': '7 hours',
            'project': 'Feed-Forward Network'
        },
        {
            'day': 6,
            'title': 'Deep Learning - Part 2',
            'topics': ['CNNs', 'RNNs', 'LSTMs'],
            'duration': '7 hours',
            'project': 'Time Series Prediction'
        },
        {
            'day': 7,
            'title': 'Computer Vision',
            'topics': ['Image Processing', 'Transfer Learning', 'Object Detection'],
            'duration': '7 hours',
            'project': 'Form Analyzer'
        },
        {
            'day': 8,
            'title': 'Natural Language Processing',
            'topics': ['Text Processing', 'Transformers', 'BERT & GPT'],
            'duration': '7 hours',
            'project': 'AI Coach Chatbot'
        },
        {
            'day': 9,
            'title': 'Production ML & MLOps',
            'topics': ['Model Deployment', 'Docker', 'MLflow'],
            'duration': '7 hours',
            'project': 'Complete ML Pipeline'
        },
        {
            'day': 10,
            'title': 'Final Exam & Certification',
            'topics': ['Comprehensive Review', '50-Question Exam', 'Capstone'],
            'duration': '7 hours',
            'project': 'Final Presentation'
        }
    ]

    return render_template('index.html',
                         curriculum=curriculum,
                         progress=progress,
                         current_day=progress['current_day'])

@app.route('/day/<int:day>')
def day_lesson(day):
    """Display lesson content for specific day"""
    if day < 1 or day > 10:
        return "Invalid day", 404

    progress = get_user_progress()

    # Load lesson content
    lesson_path = f'lessons/day{day}.md'
    lesson_content = ""
    if os.path.exists(lesson_path):
        with open(lesson_path, 'r') as f:
            lesson_content = f.read()

    # Load practice exercises for this day
    exercises = load_practice_exercises(day)

    return render_template('lesson.html',
                         day=day,
                         lesson_content=lesson_content,
                         exercises=exercises,
                         progress=progress)

def load_practice_exercises(day):
    """Load practice exercises for a specific day"""
    try:
        # Import the exercises module
        import sys
        sys.path.insert(0, 'practice')

        if day == 1:
            from day1_exercises import exercises
            return exercises
        # Add more days as needed
        return []
    except:
        return []

@app.route('/practice/<int:day>/<int:exercise_id>')
def practice_exercise(day, exercise_id):
    """Display interactive practice exercise"""
    if day < 1 or day > 10:
        return "Invalid day", 404

    if exercise_id < 1 or exercise_id > 10:
        return "Invalid exercise", 404

    exercises = load_practice_exercises(day)

    if not exercises or exercise_id > len(exercises):
        return "Exercise not found", 404

    exercise = exercises[exercise_id - 1]
    progress = get_user_progress()

    return render_template('practice.html',
                         day=day,
                         exercise=exercise,
                         progress=progress)

@app.route('/exam/<int:day>')
def exam(day):
    """Display exam for specific day"""
    if day < 1 or day > 10:
        return "Invalid day", 404

    progress = get_user_progress()
    exam_data = load_exam(day)

    if not exam_data:
        return render_template('no_exam.html', day=day)

    return render_template('exam.html',
                         day=day,
                         exam=exam_data,
                         progress=progress)

@app.route('/submit_exam/<int:day>', methods=['POST'])
def submit_exam(day):
    """Grade and save exam results"""
    answers = request.json.get('answers', {})
    exam_data = load_exam(day)

    if not exam_data:
        return jsonify({'error': 'Exam not found'}), 404

    # Grade exam
    correct_count = 0
    total_questions = len(exam_data['questions'])

    results = []
    for i, question in enumerate(exam_data['questions']):
        user_answer = answers.get(str(i))
        is_correct = user_answer == question['correct_answer']

        if is_correct:
            correct_count += 1

        results.append({
            'question_num': i + 1,
            'correct': is_correct,
            'user_answer': user_answer,
            'correct_answer': question['correct_answer'],
            'explanation': question.get('explanation', '')
        })

    score = (correct_count / total_questions) * 100
    passed = score >= 70

    # Save exam result to database
    user = get_current_user()

    # Check if exam result already exists for this day
    existing_result = ExamResult.query.filter_by(user_id=user.id, day=day).first()

    if existing_result:
        # Update existing result
        existing_result.score = score
        existing_result.correct = correct_count
        existing_result.total = total_questions
        existing_result.passed = passed
        existing_result.set_answers(answers)
        existing_result.timestamp = datetime.utcnow()
    else:
        # Create new exam result
        exam_result = ExamResult(
            user_id=user.id,
            day=day,
            score=score,
            correct=correct_count,
            total=total_questions,
            passed=passed
        )
        exam_result.set_answers(answers)
        db.session.add(exam_result)

    db.session.commit()

    # Update progress if passed
    progress = get_user_progress()
    if passed and day not in progress['completed_days']:
        progress['completed_days'].append(day)
        if day < 10:
            progress['current_day'] = day + 1
        save_user_progress(progress)

    return jsonify({
        'score': score,
        'correct': correct_count,
        'total': total_questions,
        'passed': passed,
        'results': results
    })

@app.route('/progress')
def view_progress():
    """View detailed progress report"""
    progress = get_user_progress()
    return render_template('progress.html', progress=progress)

@app.route('/final_exam')
def final_exam():
    """Final comprehensive exam"""
    progress = get_user_progress()

    # Check if completed Days 1-9
    if len(progress['completed_days']) < 9:
        return render_template('not_ready.html',
                             completed=len(progress['completed_days']))

    exam_data = load_exam(10)
    return render_template('final_exam.html', exam=exam_data)

@app.route('/certificate')
def certificate():
    """Generate certificate"""
    progress = get_user_progress()

    # Calculate overall performance
    exam_scores = progress.get('exam_scores', {})

    if len(exam_scores) < 10:
        return "Complete all exams to get certificate", 403

    # Check final exam score
    final_score = exam_scores.get('day10', {}).get('score', 0)

    if final_score < 70:
        return "You need 70% on final exam to get certified", 403

    # Determine certificate level
    if final_score >= 95:
        level = "Outstanding"
        color = "gold"
    elif final_score >= 85:
        level = "Distinction"
        color = "silver"
    elif final_score >= 70:
        level = "Certified"
        color = "bronze"
    else:
        level = "Participant"
        color = "gray"

    return render_template('certificate.html',
                         level=level,
                         color=color,
                         final_score=final_score,
                         completion_date=datetime.now().strftime('%B %d, %Y'))

@app.route('/reset_progress', methods=['POST'])
def reset_progress():
    """Reset all progress (for testing)"""
    user = get_current_user()

    # Delete all exam results
    ExamResult.query.filter_by(user_id=user.id).delete()

    # Reset progress
    progress_record = Progress.query.filter_by(user_id=user.id).first()
    if progress_record:
        progress_record.current_day = 1
        progress_record.set_completed_days([])
        progress_record.total_score = 0.0

    db.session.commit()
    session.clear()
    return jsonify({'success': True})

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True') == 'True'

    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                      ML BOOTCAMP SERVER                       ║
    ║           10-Day Interactive ML Learning Platform             ║
    ╚═══════════════════════════════════════════════════════════════╝

    🚀 Server running at: http://{host}:{port}
    📚 Guest Mode: Enabled (No login required!)
    💾 Database: {app.config['SQLALCHEMY_DATABASE_URI']}
    🔧 Environment: {app.config['FLASK_ENV']}

    Press Ctrl+C to stop the server
    """)

    app.run(debug=debug, host=host, port=port)
