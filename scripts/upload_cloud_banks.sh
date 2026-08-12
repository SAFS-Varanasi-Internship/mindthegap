#!/usr/bin/env bash
# Upload cloud-bank binaries as GitHub Release assets.
#
# Cloud banks are NOT tracked in git (see .gitignore) and are NOT shipped in the
# PyPI wheel. They are distributed as GitHub Release assets and downloaded on
# first use by mindthegap.cloud_bank.fetch_bank (which verifies the sha256
# recorded in mindthegap/cloud_banks_manifest.json).
#
# Prerequisites:
#   - gh CLI installed and authenticated:  gh auth login
#   - run from the repository root
#
# Usage:
#   bash scripts/upload_cloud_banks.sh
#
# The release tag must match "release_tag" in cloud_banks_manifest.json.

set -euo pipefail

TAG="cloud-banks-v1"
ASSET_DIR="assets/cloud_banks"

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI not found. Install it and run 'gh auth login'." >&2
  exit 1
fi

shopt -s nullglob
assets=("${ASSET_DIR}"/*.nc)
if [ ${#assets[@]} -eq 0 ]; then
  echo "error: no .nc files found in ${ASSET_DIR}." >&2
  echo "Generate one first, e.g.:" >&2
  echo "  python scripts/make_cloud_bank.py --help" >&2
  exit 1
fi

if gh release view "${TAG}" >/dev/null 2>&1; then
  echo "Release ${TAG} exists; uploading (overwriting) assets..."
  gh release upload "${TAG}" "${assets[@]}" --clobber
else
  echo "Creating release ${TAG} and attaching assets..."
  gh release create "${TAG}" "${assets[@]}" \
    --title "Cloud banks v1" \
    --notes "Precomputed synthetic-cloud banks for prepare_model_data(cloud_mode='synthetic_bank')."
fi

echo "Done. Assets uploaded to release ${TAG}."
