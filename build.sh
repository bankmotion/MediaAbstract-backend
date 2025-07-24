#!/usr/bin/env bash

# Install dependencies
pip install -r requirements.txt

# Download the SpaCy language model
python -m spacy download en_core_web_md
