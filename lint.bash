#!/bin/bash

poetry run black .
yes | poetry run isort . --interactive
poetry run flake8 .
poetry run mypy .
