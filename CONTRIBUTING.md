# Contributing to triples-sigfast

Thank you for your interest in contributing! This guide will get you set up in minutes.

---

##  Setting Up Your Development Environment

### 1. Fork and clone the repository
```bash
git clone https://github.com/SamdaniSayam/triples-sigfast.git
cd triples-sigfast
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .  # Install your local copy in editable mode
```

---

## Running Tests
```bash
pytest --cov=triples_sigfast --cov-report=term-missing
```

All tests must pass before submitting a pull request.

---

## Code Style

We use **Ruff** for linting and formatting. Before committing, always run:
```bash
ruff check --fix .
ruff format .
```

The CI pipeline will reject any code that fails Ruff checks.

---

## Submitting a Pull Request

1. Create a new branch from `main`:
```bash
   git checkout -b feat/your-feature-name
```

2. Make your changes

3. Run linting and tests:
```bash
   ruff check --fix . && ruff format . && pytest
```

4. Commit using clear messages:
```bash
   git commit -m "feat: add your feature description"
```

5. Push and open a Pull Request on GitHub

---

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Use for |
|---|---|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation changes |
| `test:` | Adding or fixing tests |
| `ci:` | CI/CD pipeline changes |
| `refactor:` | Code refactoring |
| `release:` | Version bumps |

---

## Reporting Bugs

Open an issue on GitHub with:
- Your OS and Python version
- Minimal code to reproduce the bug
- Expected vs actual output

---

## License

By contributing, you agree your changes will be licensed under the MIT License.
