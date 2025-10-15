# Contributing to ML Bootcamp

Thank you for your interest in contributing to ML Bootcamp! This document provides guidelines and instructions for contributing to this open-source project.

## 🌟 How to Contribute

We welcome contributions of all kinds:
- 🐛 **Bug Reports**: Found a bug? Let us know!
- 💡 **Feature Requests**: Have an idea? We'd love to hear it!
- 📝 **Content Improvements**: Better explanations, more exercises
- 🎨 **UI/UX Enhancements**: Make the platform more beautiful
- 📚 **Documentation**: Help others understand the code
- 🧪 **Tests**: Add test coverage for reliability

## 🚀 Quick Start for Contributors

###1. Fork and Clone
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/ml-bootcamp.git
cd ml-bootcamp
```

### 2. Set Up Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python init_db.py

# Run the app
python app.py
```

### 3. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 4. Make Your Changes
- Write clean, readable code
- Follow existing code style
- Add comments for complex logic
- Update documentation if needed

### 5. Test Your Changes
```bash
# Run the app locally
python app.py

# Test manually in browser
open http://localhost:5000

# Run tests (if available)
pytest
```

### 6. Commit and Push
```bash
git add .
git commit -m "feat: Add new feature description"
# or
git commit -m "fix: Fix bug description"

git push origin feature/your-feature-name
```

### 7. Create Pull Request
- Go to GitHub and create a Pull Request
- Describe your changes clearly
- Reference any related issues

## 📝 Commit Message Guidelines

Use conventional commits format:

- `feat: Add new feature`
- `fix: Fix bug`
- `docs: Update documentation`
- `style: Format code (no logic changes)`
- `refactor: Refactor code`
- `test: Add tests`
- `chore: Update dependencies`

## 🎯 Areas to Contribute

### 1. Practice Exercises
Create more interactive exercises for Days 2-10:
- File location: `practice/dayX_exercises.py`
- Follow the structure in `practice/day1_exercises.py`
- Include: title, description, starter_code, solution, hints, explanation

### 2. Exam Questions
Add more comprehensive exam questions:
- File location: `exams/dayX_exam.json`
- Include clear explanations for correct answers
- Test both concepts and practical application

### 3. Lesson Content
Write detailed lesson content:
- File location: `lessons/dayX.md`
- Use Markdown format
- Include code examples
- Add visual aids (diagrams, charts)

### 4. UI Improvements
- Enhance the user interface
- Add animations and transitions
- Improve mobile responsiveness
- Add dark/light theme toggle

### 5. Features
- Add progress charts and analytics
- Implement leaderboards
- Add social sharing
- Create downloadable PDF certificates
- Add video tutorials integration

### 6. Testing
- Write unit tests for Python code
- Add integration tests for routes
- Test browser compatibility
- Add automated testing (CI/CD)

## 🏗️ Project Structure

```
ml-bootcamp/
├── app.py                # Main Flask application
├── models.py             # Database models
├── init_db.py            # Database initialization
├── templates/            # HTML templates
├── exams/                # Exam JSON files
├── practice/             # Practice exercises
├── lessons/              # Lesson content (Markdown)
├── static/               # CSS, JS, images
├── quick-start.sh        # Unix startup script
├── quick-start.bat       # Windows startup script
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose config
└── requirements.txt      # Python dependencies
```

## 💻 Code Style

- **Python**: Follow PEP 8 style guide
- **HTML/CSS**: Use consistent indentation (2 spaces)
- **JavaScript**: Use modern ES6+ syntax
- **Comments**: Write clear, helpful comments

## 🐛 Reporting Bugs

When reporting bugs, please include:
1. **Description**: What happened?
2. **Expected Behavior**: What should happen?
3. **Steps to Reproduce**: How to trigger the bug?
4. **Environment**: OS, Python version, browser
5. **Screenshots**: If applicable

## 💡 Suggesting Features

When suggesting features, please include:
1. **Problem**: What problem does this solve?
2. **Solution**: How should it work?
3. **Alternatives**: Other ways to solve this?
4. **Context**: Why is this important?

## 📜 Code of Conduct

### Our Pledge
- Be respectful and inclusive
- Accept constructive feedback
- Focus on what's best for the community
- Show empathy toward others

### Our Standards
✅ **DO**:
- Use welcoming and inclusive language
- Respect differing viewpoints
- Accept constructive criticism gracefully
- Focus on learning and teaching

❌ **DON'T**:
- Use inappropriate language or imagery
- Make personal attacks
- Harass or discriminate
- Publish others' private information

## 🏆 Recognition

Contributors will be:
- Listed in the project README
- Mentioned in release notes
- Given credit in code comments
- Part of an amazing learning community!

## 📞 Questions?

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Email**: Contact maintainers

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for making ML education better for everyone!** 🎓✨
