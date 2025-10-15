"""
Database Models for ReddyGo ML Bootcamp
Stores user progress, exam results, and learning data permanently
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    """User model - for future user accounts/authentication"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, default='default_user')
    email = db.Column(db.String(120), unique=True, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    progress = db.relationship('Progress', backref='user', uselist=False, cascade='all, delete-orphan')
    exam_results = db.relationship('ExamResult', backref='user', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<User {self.username}>'

class Progress(db.Model):
    """User progress tracking"""
    __tablename__ = 'progress'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)
    current_day = db.Column(db.Integer, default=1)
    completed_days = db.Column(db.String(100), default='[]')  # JSON array as string
    total_score = db.Column(db.Float, default=0.0)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<Progress user_id={self.user_id} day={self.current_day}>'

    def get_completed_days(self):
        """Parse completed days from JSON string"""
        import json
        try:
            return json.loads(self.completed_days) if self.completed_days else []
        except:
            return []

    def set_completed_days(self, days_list):
        """Store completed days as JSON string"""
        import json
        self.completed_days = json.dumps(days_list)

class ExamResult(db.Model):
    """Exam results for each day"""
    __tablename__ = 'exam_results'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    day = db.Column(db.Integer, nullable=False)
    score = db.Column(db.Float, nullable=False)
    correct = db.Column(db.Integer, nullable=False)
    total = db.Column(db.Integer, nullable=False)
    passed = db.Column(db.Boolean, nullable=False)
    answers = db.Column(db.Text)  # JSON string of user's answers
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # Unique constraint: one latest result per user per day
    __table_args__ = (
        db.Index('idx_user_day', 'user_id', 'day'),
    )

    def __repr__(self):
        return f'<ExamResult user_id={self.user_id} day={self.day} score={self.score}%>'

    def get_answers(self):
        """Parse answers from JSON string"""
        import json
        try:
            return json.loads(self.answers) if self.answers else {}
        except:
            return {}

    def set_answers(self, answers_dict):
        """Store answers as JSON string"""
        import json
        self.answers = json.dumps(answers_dict)
