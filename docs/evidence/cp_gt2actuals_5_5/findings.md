# CP-GT2ACTUALS-5.5: FP7oJQ 94%-Unclassifiable Resolution

**Date:** 2026-06-10

## 1. Decomposition of Unclassifiable Split Events

| Camera | Total | Outside-annotated (a) | Inside-no-match (b) | Lineage-gap (c) | Classified |
|--------|-------|-----------------------|---------------------|-----------------|------------|
| FP7oJQ | 725 | 692 (95.4%) | 0 | 0 | 42 (5.8%) |
| vid1 | 308 | 138 (44.8%) | 3 (1.0%) | 0 | 167 (54.2%) |
| vid2 | 354 | 0 (0%) | 2 (0.6%) | 0 | 352 (99.4%) |
| PPDmUg | 444 | 300 (67.6%) | 0 | 0 | 148 (33.3%) |

**Cause (a) dominates everywhere except vid2.** FP7oJQ's 94% unclassifiable is
a pure GT-coverage artifact: only 301 annotated frames (0-300), but 725 split
events with products spanning 0-4500+. Only 9 of 358 unique products (3%)
overlap the annotated window at all.

**Cause (b) is trivial**: 5 events total (3 vid1 + 2 vid2). All have post-split
products with zero detection matches in the dense join — short fragments that
no GT person's bounding box overlaps. Benign.

**Cause (c) (sibling/lineage-lookup failure) is ZERO across all cameras.** No
classification-logic gap found. The sibling-product lookup works correctly.

## 2. Annotated-Coverage Hypothesis — CONFIRMED

| Camera | Annotated frames | Total products | Products overlapping annotated | Overlap % |
|--------|-----------------|----------------|-------------------------------|-----------|
| FP7oJQ | 301 (0-300) | 358 | 9 | 3% |
| vid1 | 3,001 (0-3000) | 308 | 170 | 55% |
| vid2 | 4,491 (0-4490) | 354 | 354 | 100% |
| PPDmUg | 300 (0-2990, stride 10) | 213 | 117 | 55% |

The contrast is entirely explained by annotation coverage. vid2 has full
dense coverage (stride-1, 4491 frames) so 100% of products can be classified.
FP7oJQ has only 301 contiguous frames at the clip start, so 97% of products
fall outside and cannot be classified.

## 3. Net-Effect Table with Honest Scope

| Camera | Correct | False | Unclass | Classified% | Net | Scope |
|--------|---------|-------|---------|-------------|-----|-------|
| **vid2** | **35** | **317** | **2** | **99.4%** | **-282** | **FULL** |
| vid1 | 43 | 124 | 141 | 54.2% | -81 | Partial |
| PPDmUg | 64 | 84 | 296 | 33.3% | -20 | Thin |
| FP7oJQ | 7 | 35 | 683 | 5.8% | -28 | Very thin |

**Direction (net-negative) holds on ALL four cameras.** But only vid2 supports
a trustworthy magnitude (99.4% classified). vid1 is partially characterized
(54%). FP7oJQ and PPDmUg magnitudes should not be cited as equivalent evidence.

**Authoritative figure: vid2 D0.5 is net -282 (35 correct / 317 false) at
99.4% classification coverage.** This is the number to cite.

## 4. Trusted Cameras — Confirmed Clean

The 5 inside-but-unclassified events (3 vid1 + 2 vid2) are all benign
(post-split product has zero GT detection matches — short fragments with no
bounding box overlap). No sibling-lookup gap (cause c) found on any camera.

**vid2's 352 classified splits are not contaminated by any classification-logic
gap.** The correct/false tallies are trustworthy.

vid1's 167 classified splits are also clean — the 3 inside-unclassified are
zero-match fragments, not misclassifications.
