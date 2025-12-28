# Contributing to MISATA-CGS

Thank you for your interest in contributing to MISATA-CGS!

## 🐛 Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub.

## 🔧 Development Setup

```bash
# Clone the repository
git clone https://github.com/rasinmuhammed/misata-cgs.git
cd misata-cgs

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📝 Code Style

- Follow PEP 8 guidelines
- Add docstrings to all public functions
- Include type hints where possible

## 🧪 Running Experiments

Key experiments are in the `experiments/` folder. Start with:

```bash
jupyter notebook experiments/01B_fair_performance_benchmark.ipynb
```

## 📬 Contact

For questions about the research, please open a GitHub issue or contact the author.

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
