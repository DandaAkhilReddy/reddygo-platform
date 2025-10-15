"""
Initialize the database for ReddyGo ML Bootcamp
Run this script once to create all database tables
"""

import os
from app import app, db
from models import User, Progress, ExamResult

def init_database():
    """Create all database tables and initialize with default user"""

    with app.app_context():
        # Create instance directory if it doesn't exist
        instance_path = os.path.join(os.path.dirname(__file__), 'instance')
        os.makedirs(instance_path, exist_ok=True)

        # Drop all existing tables (be careful with this in production!)
        print("Dropping existing tables...")
        db.drop_all()

        # Create all tables
        print("Creating database tables...")
        db.create_all()

        # Create default user
        print("Creating default user...")
        default_user = User(username='default_user', email='user@reddygo.com')
        db.session.add(default_user)
        db.session.commit()

        # Create initial progress record
        print("Initializing progress...")
        progress = Progress(
            user_id=default_user.id,
            current_day=1,
            completed_days='[]',
            total_score=0.0
        )
        db.session.add(progress)
        db.session.commit()

        print("\n" + "="*60)
        print("Database initialized successfully!")
        print("="*60)
        print(f"Database location: {instance_path}/bootcamp.db")
        print(f"Default user created: {default_user.username}")
        print("You can now run the Flask app with: python app.py")
        print("="*60)

if __name__ == '__main__':
    init_database()
