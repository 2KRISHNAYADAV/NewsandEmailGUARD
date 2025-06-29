# 🤝 Contributing to NewsGuardian.AI

Thank you for your interest in contributing to NewsGuardian.AI! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [🎯 How Can I Contribute?](#-how-can-i-contribute)
- [🐛 Reporting Bugs](#-reporting-bugs)
- [💡 Suggesting Enhancements](#-suggesting-enhancements)
- [🔧 Development Setup](#-development-setup)
- [📝 Code Style](#-code-style)
- [🧪 Testing](#-testing)
- [📚 Documentation](#-documentation)
- [🚀 Pull Request Process](#-pull-request-process)
- [📞 Getting Help](#-getting-help)

---

## 🎯 How Can I Contribute?

### 🐛 Bug Reports
- Report bugs you find in the application
- Help improve error handling and edge cases
- Test the application on different platforms

### 💡 Feature Requests
- Suggest new features or improvements
- Propose enhancements to existing functionality
- Share ideas for better user experience

### 🔧 Code Contributions
- Fix bugs and implement features
- Improve code quality and performance
- Add new machine learning models
- Enhance the UI/UX design

### 📚 Documentation
- Improve README and documentation
- Add code comments and docstrings
- Create tutorials and examples
- Translate documentation to other languages

### 🧪 Testing
- Write unit tests and integration tests
- Test the application on different devices
- Validate model performance and accuracy

---

## 🐛 Reporting Bugs

### Before Submitting a Bug Report

1. **Check Existing Issues**: Search the [issues page](https://github.com/yourusername/NewsGuardian.AI/issues) to see if the bug has already been reported.

2. **Test Latest Version**: Make sure you're using the latest version of the code.

3. **Reproduce the Issue**: Try to reproduce the bug consistently.

### Bug Report Template

```markdown
## Bug Description
Brief description of the bug.

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. Enter '...'
4. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment Information
- **OS**: [e.g., Windows 10, macOS 12.0, Ubuntu 20.04]
- **Python Version**: [e.g., 3.8.10]
- **Browser**: [e.g., Chrome 96.0, Firefox 95.0] (if applicable)
- **Streamlit Version**: [e.g., 1.28.0]

## Additional Information
- Screenshots (if applicable)
- Error messages or logs
- Any other relevant information
```

---

## 💡 Suggesting Enhancements

### Before Submitting a Feature Request

1. **Check Existing Requests**: Search existing issues to see if your feature has already been suggested.

2. **Think About Use Cases**: Consider how the feature would benefit users.

3. **Consider Implementation**: Think about how the feature could be implemented.

### Feature Request Template

```markdown
## Feature Description
Brief description of the feature you'd like to see.

## Problem Statement
What problem does this feature solve?

## Proposed Solution
How would you like this feature to work?

## Use Cases
Describe specific scenarios where this feature would be useful.

## Alternative Solutions
Are there any alternative approaches you've considered?

## Additional Information
Any other relevant information or context.
```

---

## 🔧 Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- pip or conda

### Setup Instructions

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/NewsGuardian.AI.git
   cd NewsGuardian.AI
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Download NLTK Data**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

6. **Train Models**
   ```bash
   python train_models_improved.py
   ```

7. **Run Tests**
   ```bash
   python -m pytest tests/
   ```

8. **Start Development Server**
   ```bash
   streamlit run app/app.py
   ```

---

## 📝 Code Style

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines:

- **Indentation**: 4 spaces (no tabs)
- **Line Length**: Maximum 88 characters (Black formatter)
- **Naming**: 
  - Variables and functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`

### Code Formatting

We use automated tools for code formatting:

```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Check code style with flake8
flake8 .
```

### Docstring Style

Use Google-style docstrings:

```python
def predict_news(text: str, models: dict) -> dict:
    """Predict if news is fake or real.
    
    Args:
        text: The news article text to analyze.
        models: Dictionary containing trained models.
        
    Returns:
        Dictionary containing prediction results:
        - prediction: 0 for real, 1 for fake
        - probability: Confidence scores
        - model_type: Type of model used
    """
    pass
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run tests with coverage
python -m pytest --cov=.

# Run specific test file
python -m pytest tests/test_models.py

# Run tests with verbose output
python -m pytest -v
```

### Writing Tests

- Write tests for new features
- Ensure good test coverage
- Use descriptive test names
- Test both success and failure cases

Example test:

```python
def test_predict_news_real_article():
    """Test prediction for a real news article."""
    text = "Scientists discover new species in the Pacific Ocean..."
    models = load_test_models()
    
    result = predict_news(text, models)
    
    assert result['prediction'] == 0  # Real news
    assert result['confidence'] > 0.8
    assert 'model_type' in result
```

---

## 📚 Documentation

### Code Documentation

- Add docstrings to all functions and classes
- Include type hints for function parameters
- Document complex algorithms and logic
- Add inline comments for non-obvious code

### User Documentation

- Update README.md for new features
- Add usage examples
- Include screenshots for UI changes
- Document configuration options

### API Documentation

- Document all public functions
- Include parameter descriptions
- Provide return value explanations
- Add usage examples

---

## 🚀 Pull Request Process

### Before Submitting a PR

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run tests
   python -m pytest
   
   # Check code style
   flake8 .
   black --check .
   
   # Test the application
   streamlit run app/app.py
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

### PR Guidelines

1. **Title**: Use conventional commit format
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `style:` for formatting changes
   - `refactor:` for code refactoring
   - `test:` for adding tests

2. **Description**: Include:
   - What the PR does
   - Why the changes are needed
   - How to test the changes
   - Any breaking changes

3. **Checklist**: Ensure:
   - [ ] Code follows style guidelines
   - [ ] Tests pass
   - [ ] Documentation is updated
   - [ ] No breaking changes (or documented)

### PR Template

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes

## Screenshots (if applicable)
Add screenshots for UI changes.
```

---

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions and discussions
- **Email**: support@newsguardian.ai
- **Discord**: [NewsGuardian.AI Community](https://discord.gg/newsguardian)

### Resources

- [Python Documentation](https://docs.python.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [NLTK Documentation](https://www.nltk.org/)

---

## 🙏 Recognition

Contributors will be recognized in:

- The project README
- Release notes
- Contributor hall of fame
- GitHub contributors page

Thank you for contributing to NewsGuardian.AI! 🛡️✨ 