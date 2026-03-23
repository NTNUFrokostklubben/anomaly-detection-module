#!/usr/bin/env bash
#
# Fetches the anomaly detection test data from GitHub Packages.
# Pulls a pinned version from ghcr.io using the OCI distribution API.
# Verifies SHA256 and extracts to test_data/.
# Version and SHA256 are read from bundle-versions.json in the project root.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
VERSION_FILE="${PROJECT_ROOT}/bundle-versions.json"

if [ ! -f "$VERSION_FILE" ]; then
    echo "ERROR: bundle-versions.json not found at $VERSION_FILE" >&2
    exit 1
fi

VERSION=$(python3 -c "import json; d=json.load(open('${VERSION_FILE}')); print(d['anomaly-detection-test-data']['version'])")
EXPECTED_SHA256=$(python3 -c "import json; d=json.load(open('${VERSION_FILE}')); print(d['anomaly-detection-test-data']['sha256'])")

IMAGE="ntnufrokostklubben/anomaly-detection-test-data"
OUTPUT_DIR="${PROJECT_ROOT}/test_data"
VERSION_MARKER="${OUTPUT_DIR}/.version"

# Check if already up to date
if [ -f "$VERSION_MARKER" ]; then
    installed=$(cat "$VERSION_MARKER" | tr -d '[:space:]')
    if [ "$installed" = "$VERSION" ]; then
        echo "Test data $VERSION already up to date."
        exit 0
    fi
fi

# Remove existing and re-download
if [ -d "$OUTPUT_DIR" ]; then
    rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

# Step 1: Get anonymous pull token
echo "Downloading test data $VERSION..."
token=$(curl -s "https://ghcr.io/token?service=ghcr.io&scope=repository:${IMAGE}:pull" | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")

# Step 2: Fetch manifest to get the layer digest
layer_digest=$(curl -s \
    -H "Authorization: Bearer $token" \
    -H "Accept: application/vnd.oci.image.manifest.v1+json" \
    "https://ghcr.io/v2/${IMAGE}/manifests/${VERSION}" | python3 -c "import sys,json; print(json.load(sys.stdin)['layers'][0]['digest'])")

if [ -z "$layer_digest" ]; then
    echo "ERROR: Failed to resolve layer digest from manifest." >&2
    exit 1
fi

# Step 3: Download the blob
zip_path=$(mktemp /tmp/anomaly-detection-test-data.XXXXXX.zip)
curl -L \
    -H "Authorization: Bearer $token" \
    -o "$zip_path" \
    "https://ghcr.io/v2/${IMAGE}/blobs/${layer_digest}"

if [ ! -f "$zip_path" ]; then
    echo "ERROR: Download failed." >&2
    exit 1
fi

# Verify SHA256
if [ -n "$EXPECTED_SHA256" ]; then
    actual_hash=$(sha256sum "$zip_path" | awk '{print $1}')
    expected_lower=$(echo "$EXPECTED_SHA256" | tr '[:upper:]' '[:lower:]')
    if [ "$actual_hash" != "$expected_lower" ]; then
        echo "ERROR: SHA256 mismatch!" >&2
        echo "  Expected: $expected_lower" >&2
        echo "  Got:      $actual_hash" >&2
        rm -f "$zip_path"
        exit 1
    fi
else
    echo "WARNING: No SHA256 configured in bundle-versions.json. Skipping verification."
fi

# Extract
unzip -o "$zip_path" -d "$OUTPUT_DIR" || true
rm -f "$zip_path"

echo "$VERSION" > "$VERSION_MARKER"

echo "Done. Test data $VERSION ready."