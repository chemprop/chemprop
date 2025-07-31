#!/bin/bash

# This script iterates through all Chemprop Docker image versions, runs pip freeze in each,
# and saves the output to a file named after the version.

# Check for required tools
command -v docker >/dev/null 2>&1 || { echo "Error: Docker is not installed."; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "Error: curl is not installed."; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "Error: jq is not installed."; exit 1; }

# Docker Hub repository
REPO="chemprop/chemprop"

# Docker Hub API URL for tags
API_URL="https://hub.docker.com/v2/repositories/${REPO}/tags?page_size=100"

# Temporary file for API response
TEMP_FILE=$(mktemp)

# Function to clean up temporary file
cleanup() {
    rm -f "$TEMP_FILE"
}
trap cleanup EXIT

echo "Fetching tags for ${REPO}..."

# Fetch tags from Docker Hub API
curl -s "$API_URL" > "$TEMP_FILE"

# Check if API request was successful
if ! jq -e '.results' "$TEMP_FILE" >/dev/null; then
    echo "Error: Failed to fetch tags from Docker Hub."
    exit 1
fi

# Extract tags
TAGS=$(jq -r '.results[].name' "$TEMP_FILE" | sort -V)

if [ -z "$TAGS" ]; then
    echo "Error: No tags found for ${REPO}."
    exit 1
fi

SKIP_TAGS=("1.6.1" "1.7.0" "latest")

# Iterate through each tag
for TAG in $TAGS; do
    if [[ " ${SKIP_TAGS[@]} " =~ " ${TAG} " ]]; then
        echo "Skipping $TAG"
        continue
    fi
    echo "Processing tag: $TAG"

    # Pull the Docker image
    echo "Pulling ${REPO}:${TAG}..."
    if ! docker pull "${REPO}:${TAG}" >/dev/null 2>&1; then
        echo "Warning: Failed to pull ${REPO}:${TAG}. Skipping..."
        continue
    fi

    # Run the container and execute pip freeze with Conda environment's Python
    echo "Running pip freeze in ${REPO}:${TAG}..."
    OUTPUT_FILE="${TAG}_requirements.txt"
    ERROR_FILE="${TAG}_pip_error.txt"
    # Use /bin/bash to ensure Conda environment activation
    if ! docker run --rm "${REPO}:${TAG}" -c "python -m pip list --format=freeze" > "$OUTPUT_FILE" 2> "$ERROR_FILE"; then
        echo "Warning: Failed to run pip freeze in ${REPO}:${TAG}. See $ERROR_FILE for details. Skipping..."
        cat "$ERROR_FILE" >&2
        rm -f "$OUTPUT_FILE"
        continue
    fi

    # Clean up error file if successful
    rm -f "$ERROR_FILE"

    # Check if output file is empty
    if [ ! -s "$OUTPUT_FILE" ]; then
        echo "Warning: No output from pip freeze for ${REPO}:${TAG}. Removing empty file."
        rm -f "$OUTPUT_FILE"
        continue
    fi

    echo "Saved pip freeze output to $OUTPUT_FILE"
done

echo "Processing complete."
