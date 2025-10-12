# Contributing to LLM Market Decision Agent

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Areas for Contribution](#areas-for-contribution)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). By participating, you agree to:

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

---

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork:
git clone https://github.com/YOUR-USERNAME/market-master-trading-decision-agent.git
cd market-master-trading-decision-agent

# Add upstream remote:
git remote add upstream https://github.com/ORIGINAL-OWNER/market-master-trading-decision-agent.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install black flake8 pytest mypy
```

### 3. Create a Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

---

## Development Workflow

### Making Changes

1. **Keep changes focused**: One feature/fix per PR
2. **Write tests**: Add tests for new functionality
3. **Update documentation**: Update README.md and docstrings
4. **Follow style guide**: Use Black for formatting

### Testing Locally

```bash
# Run pipeline
python run_pipeline.py

# Launch Streamlit
streamlit run app/streamlit_app.py

# Run evaluation notebook
jupyter notebook app/evaluate_llm.ipynb
```

---

## Coding Standards

### Python Style

Follow [PEP 8](https://pep8.org/) with these specifics:

```python
# ✅ Good
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with added feature columns
    """
    logger.info("Computing features...")
    return df

# ❌ Bad
def compute_features(df):
    return df
```

### Code Formatting

Use [Black](https://black.readthedocs.io/):

```bash
# Format all files
black .

# Check without modifying
black --check .
```

### Type Hints

Use type hints for function signatures:

```python
from typing import List, Dict, Optional

def fetch_data(symbols: List[str], period: str = "60d") -> pd.DataFrame:
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def compute_wss(
    rsi: pd.Series,
    atr: pd.Series,
    volume_bias: pd.Series
) -> pd.Series:
    """
    Compute Weighted Sentiment Score.
    
    Args:
        rsi: RSI series (0-100)
        atr: ATR series
        volume_bias: Volume bias series
    
    Returns:
        WSS values (0-1 range)
    
    Example:
        >>> wss = compute_wss(rsi, atr, volume_bias)
        >>> print(wss.mean())
        0.56
    """
    ...
```

### Logging

Use appropriate log levels:

```python
import logging
logger = logging.getLogger(__name__)

logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical failure")
```

---

## Testing

### Running Tests

```bash
# Run all tests (if implemented)
pytest

# Run with coverage
pytest --cov=app --cov-report=html
```

### Writing Tests

```python
# tests/test_features.py
import pytest
import pandas as pd
from app.compute_features import compute_rsi

def test_compute_rsi():
    """Test RSI computation."""
    # Arrange
    data = pd.DataFrame({
        'close': [100, 102, 101, 103, 105, 104, 106, 108]
    })
    
    # Act
    rsi = compute_rsi(data, period=7)
    
    # Assert
    assert len(rsi) == len(data)
    assert rsi.iloc[-1] > 50  # Should be bullish
    assert 0 <= rsi.iloc[-1] <= 100
```

---

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts with main

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
How did you test this?

## Screenshots (if applicable)
Add screenshots for UI changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings generated
```

### Review Process

1. **Submit PR** with clear title and description
2. **CI checks** must pass (if configured)
3. **Code review** by maintainer(s)
4. **Address feedback** in new commits
5. **Approval** and merge by maintainer

---

## Areas for Contribution

### 🐛 Bug Fixes
- Fix data fetching errors
- Handle edge cases in indicator computation
- Improve error messages

### ✨ New Features

**High Priority:**
- [ ] RAG integration with NewsAPI
- [ ] Multi-timeframe analysis (1h, 4h, 1d)
- [ ] Backtesting framework
- [ ] Additional technical indicators (MACD, Bollinger Bands)

**Medium Priority:**
- [ ] FastAPI REST endpoint
- [ ] Alert system (email/SMS)
- [ ] Real-time WebSocket streaming
- [ ] User authentication

**Low Priority:**
- [ ] Alternative LLM providers (Anthropic, Llama)
- [ ] Portfolio simulation
- [ ] Social media sentiment integration

### 📚 Documentation
- Improve README examples
- Add tutorials/guides
- Create video walkthrough
- Translate documentation

### 🎨 UI/UX
- Enhance Streamlit dashboard
- Add dark mode
- Improve mobile responsiveness
- Create comparison charts

### 🧪 Testing
- Add unit tests
- Integration tests
- Performance benchmarks
- Load testing

### ⚡ Performance
- Optimize data fetching
- Cache LLM responses
- Parallelize processing
- Database integration (PostgreSQL)

---

## Commit Message Guidelines

Use conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**

```
feat(llm): add multi-persona prompt support

- Implement bull/bear/neutral personas
- Add persona selection in config
- Update prompt template

Closes #42
```

```
fix(data): handle missing Yahoo Finance data gracefully

Previously, missing data caused pipeline to crash.
Now falls back to cached data or skips symbol.

Fixes #38
```

---

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Create an Issue with template
- **Security issues**: Email maintainers directly (do NOT create public issue)
- **Feature requests**: Open an Issue with "enhancement" label

---

## Recognition

Contributors will be:
- Listed in README.md
- Mentioned in release notes
- Invited to join project team (for significant contributions)

---

**Thank you for contributing to LLM Market Decision Agent!** 🚀

