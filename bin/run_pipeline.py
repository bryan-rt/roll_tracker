#!/usr/bin/env python3
"""Thin wrapper to run the roll-tracker pipeline CLI.

Use this script as a stable `docker exec` target. It reads environment variables
and prints resolved directories, without changing ingestion or pipeline logic.
"""
from bjj_pipeline.stages.orchestration.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
