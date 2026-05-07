# OpenReview Reviewer Export

Extract your complete peer-review service record from OpenReview for O-1 visa evidence or other purposes.

## What it does

The script authenticates with your OpenReview account and extracts:

1. **Service roles** -- every venue-level group you belong to as Reviewer, Area Chair, Senior Area Chair, etc.
2. **Per-paper assignments** -- individual anonymous reviewer/AC IDs assigned to you per submission (e.g., `Reviewer_5LPR`), with conference, year, and workshop flag.
3. **Public reviews** -- all reviews you authored that are publicly visible, with submission titles and links (uses the official `openreview.tools.get_own_reviews()` helper).
4. **Profile snapshot** -- your public OpenReview profile data (names, affiliations, expertise).

It queries both the v2 API (`api2.openreview.net`, post-2023 venues) and v1 API (`api.openreview.net`, legacy venues).

## Setup

```bash
# From the repo root
python -m venv .venv
.venv/bin/pip install openreview-py pandas
```

## Usage

```bash
# From the repo root
.venv/bin/python scripts/openreview_reviewer_export.py
```

You will be prompted for your OpenReview email and password. No credentials are stored.

## Output

All output files are written to `.openreview/` (git-ignored):

| File | Description |
|------|-------------|
| `reviewer_groups.csv` | Venue-level service roles (venue, year, role) |
| `reviewer_assignments.csv` | Per-paper assignments (conference, submission ID, anon reviewer ID, workshop flag) |
| `reviews_authored.csv` | Public reviews with submission titles and links |
| `reviewer_export.json` | Full raw dump of all data for audit/backup |

### Note on public vs. total reviews

`reviews_authored.csv` only contains reviews where both the review and the submission are publicly readable (`readers: everyone`). Reviews at ongoing conferences (e.g., papers still under review) will not appear here but will show up in `reviewer_assignments.csv` as per-paper assignments.

## Example summary output

```
Papers reviewed per conference (excl. workshops):
  ICLR.cc 2024                    2 papers
  ICLR.cc 2025                    3 papers
  ICLR.cc 2026                    5 papers
  ICML.cc 2025                    5 papers
  ICML.cc 2026                    6 papers
  NeurIPS.cc 2023                 6 papers
  NeurIPS.cc 2024                 6 papers
  NeurIPS.cc 2025                 5 papers
  TMLR                            3 papers
  aistats.org 2025                2 papers
  TOTAL                           43 papers
```
