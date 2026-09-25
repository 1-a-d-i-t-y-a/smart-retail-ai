# Contributing

This repository is an educational/reconstructed AI/ML project. Contributions should preserve reproducibility and should not introduce credentials or private data.

## Development

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run `python scripts/build_all_models.py` if model artifacts need to be regenerated.
4. Run `python -m pytest -q` before submitting changes.

## Pull Requests

- Explain the change clearly.
- Include or update tests where appropriate.
- Do not commit `.env` files, API keys, passwords, tokens, or personal data.
- Keep README and technical documentation synchronized with behavior.
