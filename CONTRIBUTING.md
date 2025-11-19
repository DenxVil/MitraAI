# Contributing to Mitra AI

Thank you for your interest in contributing to Mitra AI! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

- Be respectful and constructive
- Focus on what is best for the community
- Show empathy towards other contributors
- Accept constructive criticism gracefully

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages

### Suggesting Features

For feature requests, please:
- Check existing issues to avoid duplicates
- Provide clear use cases
- Explain how it aligns with Mitra's goals
- Consider proposing an implementation approach

### Pull Requests

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/MitraAI.git
   cd MitraAI
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

3. **Set Up Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install pytest pytest-cov black flake8 mypy
   ```

4. **Make Your Changes**
   - Write clear, readable code
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation as needed

5. **Test Your Changes**
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Check coverage
   pytest --cov=mitra --cov-report=html
   
   # Format code
   black mitra/ tests/ main.py
   
   # Lint
   flake8 mitra/ tests/ main.py --max-line-length=100
   
   # Type check
   mypy mitra/ main.py
   ```

6. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```
   
   Commit message guidelines:
   - Use present tense ("Add feature" not "Added feature")
   - Be descriptive but concise
   - Reference issues when relevant (#123)

7. **Push and Create PR**
   ```bash
   git push origin your-branch-name
   ```
   Then create a Pull Request on GitHub with:
   - Clear title and description
   - Link to related issues
   - Screenshots if UI changes
   - Notes on testing performed

## Development Guidelines

### Code Style

- **Python**: Follow PEP 8
- **Line Length**: Max 100 characters
- **Formatting**: Use Black for automatic formatting
- **Type Hints**: Add type hints to all functions
- **Docstrings**: Use Google-style docstrings

Example:
```python
def process_message(user_id: str, message: str) -> str:
    """
    Process a user message and generate a response.
    
    Args:
        user_id: The user's unique identifier
        message: The message text
        
    Returns:
        The AI-generated response
        
    Raises:
        MitraError: If processing fails
    """
    # Implementation
```

### Testing

- **Coverage**: Aim for >80% test coverage
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Naming**: Use descriptive test names

Example:
```python
def test_emotion_analyzer_detects_sadness():
    """Test that EmotionAnalyzer correctly identifies sadness."""
    analyzer = EmotionAnalyzer()
    result = analyzer.analyze("I'm feeling very sad today")
    assert "sadness" in result.emotions
```

### Documentation

- Update README.md for major changes
- Add inline comments for complex logic
- Update ARCHITECTURE.md for architectural changes
- Keep docstrings up to date

### Git Workflow

1. Keep commits focused and atomic
2. Rebase on main before submitting PR
3. Squash commits if needed for cleaner history
4. Don't commit secrets, credentials, or sensitive data
5. Update .gitignore for new file types

## Project Structure

```
MitraAI/
├── mitra/              # Main package
│   ├── core/          # AI engine
│   ├── bot/           # Telegram interface
│   ├── models/        # Data models
│   └── utils/         # Utilities
├── tests/             # Test suite
├── .github/           # CI/CD workflows
└── docs/              # Documentation
```

## Areas for Contribution

### High Priority
- Integration tests for bot and engine
- Persistent storage (database integration)
- Improved emotion detection algorithms
- Multi-language support
- Performance optimizations

### Medium Priority
- Web dashboard for monitoring
- Advanced conversation memory
- Voice message support
- User preference system
- Analytics and insights

### Good First Issues
- Documentation improvements
- Test coverage improvements
- Code refactoring
- Bug fixes
- UI/UX enhancements

## Architecture Decisions

When proposing architectural changes:
1. Maintain modularity and separation of concerns
2. Preserve backward compatibility when possible
3. Consider performance and scalability
4. Document the reasoning behind decisions
5. Update ARCHITECTURE.md

## Security

- **Never** commit secrets or credentials
- Use environment variables for sensitive data
- Report security issues privately via email
- Follow secure coding practices
- Keep dependencies updated

## Performance

- Profile before optimizing
- Consider async/await for I/O operations
- Cache where appropriate
- Monitor memory usage
- Test with realistic data volumes

## Communication

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Pull Requests**: For code contributions
- **Email**: For private/security concerns

## Getting Help

- Check existing documentation
- Search closed issues
- Ask in GitHub Discussions
- Review the ARCHITECTURE.md file

## Recognition

Contributors will be:
- Listed in the README
- Credited in release notes
- Acknowledged for their work

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Development Tips

### Local Testing

Test locally before pushing:
```bash
# Run specific test
pytest tests/unit/test_emotion_analyzer.py::TestEmotionAnalyzer::test_positive_sentiment -v

# Run with debugging
pytest tests/ -v -s

# Run only failed tests
pytest tests/ --lf
```

### Environment Setup

Create a `.env` file for local development:
```bash
cp .env.example .env
# Edit .env with your credentials
```

Never commit your `.env` file!

### Common Issues

**Import errors**: Make sure you're in the virtual environment
```bash
source venv/bin/activate
```

**Test failures**: Check if dependencies are up to date
```bash
pip install -r requirements.txt --upgrade
```

**Type errors**: Run mypy to catch type issues early
```bash
mypy mitra/ main.py
```

## Release Process

(For maintainers)

1. Update version in `pyproject.toml` and `__init__.py`
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite
5. Create GitHub release with notes
6. Deploy to production

## Thank You!

Your contributions make Mitra AI better for everyone. We appreciate your time and effort! 🙏
