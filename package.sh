#!/bin/bash

# Name of the output zip file
OUTPUT_ZIP="package.zip"

# Find and zip all .py and .yml files recursively from current directory
find . -type f \( -name "*.py" -o -name "*.yml" \) -print | zip -@ "$OUTPUT_ZIP"

echo "Created $OUTPUT_ZIP with all .py and .yml files."
