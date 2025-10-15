"""
ReddyGo ML Bootcamp - Learning Platform
10-Day Intensive Machine Learning Bootcamp with Daily Exams
"""

from flask import Flask, render_template, request, jsonify, session
from datetime import datetime
import json
import os

app = Flask(__name__)
app.secret_key = 'reddygo-ml-bootcamp-2025'

# Load exam data
def load_exam(day):
    """Load exam questions for specific day"""
    exam_path = f'exams/day{day}_exam.json'
    if os.path.exists(exam_path):
        with open(exam_path, 'r') as f:
            return json.load(f)
    return None

# Load user progress
def get_user_progress():
    """Get current user's progress"""
    if 'progress' not in session:
        session['progress'] = {
            'completed_days': [],
            'exam_scores': {},
            'current_day': 1,
            'total_score': 0
        }
    return session['progress']

def save_user_progress(progress):
    """Save user progress to session"""
    session['progress'] = progress
    session.modified = True

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

    return render_template('lesson.html',
                         day=day,
                         lesson_content=lesson_content,
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

    # Update progress
    progress = get_user_progress()
    progress['exam_scores'][f'day{day}'] = {
        'score': score,
        'correct': correct_count,
        'total': total_questions,
        'passed': passed,
        'timestamp': datetime.now().isoformat()
    }

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
    session.clear()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
