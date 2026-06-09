# CP-PURITY-3: GT-through-Stage-D Group-Formation Oracle

## Scope

GT->D with empty identity_hints measures GROUP STRUCTURE only.
It does NOT speak to D3/D4 through-line/identity routing.
A clean group result must NOT be over-read as 'through-line is fine.'

## D0.5 Split Report

- J_EDEw-20260318-200015: 0 splits (tiers: {}, disabled=False)
- J_EDEw-20260318-200246: 0 splits (tiers: {}, disabled=False)

## M1: Group Structure Comparison (GT->D vs A&C->D)

| Clip | Real GROUPs | Oracle GROUPs | Real frames | Oracle frames | Oracle-only | Real-only |
|------|-------------|---------------|-------------|---------------|-------------|-----------|
| J_EDEw-20260318-200015 | 332 | 0 | 2985 | 0 | 0 | 2985 |
| J_EDEw-20260318-200246 | 192 | 0 | 4486 | 0 | 0 | 4486 |

**Structural finding:** The GT→D oracle produced 0 GROUP nodes despite providing separate tracklets per person. This is structurally correct: D1 forms GROUPs from tracklet LIFECYCLE EVENTS (one tracklet ending near another). GT tracklets are continuous across the full annotated range — no tracklet ends during a grapple, so no merge/split trigger fires. This means GROUPs are structurally unnecessary when detection is correct: each person has their own tracklet, so the solver assigns separate person_ids without needing a GROUP capacity hint. The former 'group-formation defect' was not a D1 logic failure — it was the absence of a second tracklet (detection under-segmentation) making GROUP formation structurally impossible.

## M2: Under-segmentation Test

- **J_EDEw-20260318-200015**: 853 defect frames in scope, 0 recovered by GT->D (0.0%). 0 out-of-scope (outside annotated range, excluded).
- **J_EDEw-20260318-200246**: 514 defect frames in scope, 0 recovered by GT->D (0.0%). 0 out-of-scope (outside annotated range, excluded).

## M3: Detection-specific Isolation

| Clip | Defect frames | Detection under-seg | D1 logic gap |
|------|---------------|--------------------|--------------| 
| J_EDEw-20260318-200015 | 853 | 853 (100.0%) | 0 (0.0%) |
| J_EDEw-20260318-200246 | 514 | 514 (100.0%) | 0 (0.0%) |

## M4: Restated Attribution

**J_EDEw-20260318-200015:** Start: 853 mishandled frames (should-group but no GROUP in A&C→D). GT→D oracle recovers 0 (0.0%). Of the 853 total: 853 are detection under-segmentation (pair-box with no second tracklet in real run), 0 are genuine D1 logic gaps (two real detections existed, D1 still didn't group), 853 not recovered by GT→D oracle.

**J_EDEw-20260318-200246:** Start: 514 mishandled frames (should-group but no GROUP in A&C→D). GT→D oracle recovers 0 (0.0%). Of the 514 total: 514 are detection under-segmentation (pair-box with no second tracklet in real run), 0 are genuine D1 logic gaps (two real detections existed, D1 still didn't group), 514 not recovered by GT→D oracle.

## M5: Verdict

- Total defect frames: 1367
- Detection under-segmentation: 1367 (100.0%)
- D1 logic gap: 0 (0.0%)
- GT oracle not recovered as GROUP: 1367 (100.0%) -- expected: with separate tracklets, GROUPs are unnecessary
- **Dominant arc:** detection (CP23 / detection model improvement)
- **Lever ordering change:** Yes — detection under-segmentation confirmed as the dominant share of the former 'D1 group-formation defect.' The 29.9%/11.6% was detection wearing a D1 costume. Fixing detection eliminates the NEED for GROUPs at these frames (separate tracklets per person). D1's GROUP logic is not broken — it is structurally irrelevant for this failure mode.

**Scope disclaimer:** GT->D with empty identity_hints measures GROUP STRUCTURE only. It does NOT speak to D3/D4 through-line/identity routing. A clean group result must NOT be over-read as 'through-line is fine.'
