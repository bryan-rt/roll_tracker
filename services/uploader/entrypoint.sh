#!/bin/bash
set -e

echo "Starting uploader worker..."

python -m services.uploader.uploader.cli --worker

