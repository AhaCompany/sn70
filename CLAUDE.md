# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run validator: `python -m validator.api_server` or `python -m validator.validator_daemon`
- Run miner: `python -m miner.perplexity.miner` or `python -m miner.perplexica.miner`
- Install dependencies: `pip install -r requirements.txt`
- Run lint check: `python -m pylint **/*.py`
- Run type check: `mypy .`

## Code Style Guidelines
- Use Python type hints with `typing` module
- Import order: standard library, third-party, project modules
- Use dataclasses for structured data
- Error handling: use try/except with specific exceptions
- Variable naming: snake_case for variables, CamelCase for classes
- Comments: docstrings for classes and functions
- Follow PEP 8 style guidelines
- Use meaningful variable/function names
- Log errors using bittensor logging (`bt.logging`)