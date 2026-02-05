#!/bin/bash
echo "Running unit tests..."
venv/bin/python -m unittest discover tests
if [ $? -ne 0 ]; then
    echo "Unit tests failed. Push aborted."
    exit 1
fi
echo "Unit tests passed."
