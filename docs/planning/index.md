---
layout: default
title: Planning Documentation
---

# Roll Tracker Planning

This section contains the core planning documentation for the Roll Tracker project.

## 📋 Core Documents

### [Planning README](README.html)
The main planning pack document covering:
- **Workflow**: How to use worker threads effectively
- **Pipeline Architecture**: Hybrid execution model (multiplex_AC)
- **Stage Contracts**: F0/F1/F2 artifact and configuration contracts
- **Current Locks**: Agreed-upon decisions and constraints
- **Checkpoint Discipline**: Requirements for keeping the pipeline runnable
- **Stage D POC Map**: Detailed specifications for the MCF stitcher

### [Worker Thread Index](WORKER_THREAD_INDEX.html)
Comprehensive index of all worker threads organized by stage (A through X), including:
- Worker IDs and responsibilities
- Current status and deliverables
- Cross-references and dependencies

## 🧵 Worker Thread Specifications

### [Worker Threads Directory](worker_threads/)
Individual worker thread specification files, each designed to be pasted as the first message in a dedicated worker chat.

**Usage**:
1. Create a new worker chat with the worker ID (e.g., `D2`)
2. Paste the entire worker markdown file
3. Ask the worker to deliver required outputs
4. Return summaries to the Manager thread

---

## ⚙️ Key Concepts

- **Min-Cost Flow (MCF)**: Mandatory stitching approach for Stage D
- **Multiplex Mode**: Single-pass execution (A+C together)
- **Artifact-Driven**: All stages communicate through versioned artifacts (F0 contract)
- **Checkpoint Discipline**: Pipeline must remain runnable end-to-end at every checkpoint

---

[← Back to Home](../)
