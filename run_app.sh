#!/bin/bash

# Terminate any existing process on port 8000
lsof -ti:8000 | xargs kill

# Start FastAPI server in the background
pipenv run uvicorn app:app --reload &

# Wait a bit for the server to start
sleep 3

# Open the HTML file in the default web browser (use 'open' for macOS)
open index.html