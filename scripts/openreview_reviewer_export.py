#!/usr/bin/env python3
"""
OpenReview reviewer record extractor -- for O-1 / visa evidence.

Uses the official openreview.tools.get_own_reviews() helper plus group
membership queries to build a complete reviewer service record.

Outputs (written to .openreview/ in the repo root):
  - reviewer_groups.csv         (venue + role + year)
  - reviewer_assignments.csv    (per-paper assignments with anon IDs)
  - reviews_authored.csv        (submission title + links for public reviews)
  - reviewer_export.json        (raw dump for audit)

Install:
    python -m venv .venv
    .venv/bin/pip install openreview-py pandas

Run (from repo root):
    .venv/bin/python scripts/openreview_reviewer_export.py
    (you'll be prompted for username + password)
"""

import openreview
import openreview.tools
import pandas as pd
import json
import getpass
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Output directory: .openreview/ at the repo root (git-ignored)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = REPO_ROOT / ".openreview"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------- 1. Login ----------
USERNAME = input("OpenReview email: ").strip()
PASSWORD = getpass.getpass("OpenReview password: ")

client = openreview.api.OpenReviewClient(
    baseurl='https://api2.openreview.net',
    username=USERNAME, password=PASSWORD,
)
PROFILE_ID = client.profile.id
print(f"Logged in. Profile: {PROFILE_ID}")

# ---------- 2. Fetch reviews via official helper ----------
# Queries both v1 (legacy) and v2 (current) APIs automatically.
print("\nFetching authored reviews (this may take a while) ...")
reviews = openreview.tools.get_own_reviews(client)
print(f"  -> {len(reviews)} public reviews found")

# ---------- 3. List venue-level service groups ----------
# Groups like ICLR.cc/2026/Conference/Reviewers, .../Area_Chairs, etc.
print("\nFetching group memberships ...")
SERVICE_KEYWORDS = ['Reviewer', 'Area_Chair', 'Senior_Area_Chair', 'Program_Chair',
                    'Action_Editor', 'Editor', 'Meta_Reviewer', 'Expert_Reviewer']

def is_service(g_id):
    if 'Authors' in g_id or '/Submission' in g_id:
        return False
    return any(k in g_id for k in SERVICE_KEYWORDS)

service_groups = []
paper_assignments = []
all_groups = []
for api_label, get_groups_fn in [('v2', client.get_groups),
                                  ('v1', openreview.Client(baseurl='https://api.openreview.net', token=client.token).get_groups)]:
    try:
        groups = get_groups_fn(member=PROFILE_ID, limit=2000)
        print(f"  [{api_label}] {len(groups)} group memberships")
        all_groups.extend(groups)
    except Exception as e:
        print(f"  [{api_label}] error: {e}")

for g in all_groups:
    gid = g.id
    if is_service(gid):
        service_groups.append(gid)
    # Individual per-submission assignments: VENUE/.../Submission1234/Reviewer_aBcD
    # The startswith check distinguishes individual anon IDs (Reviewer_5LPR)
    # from aggregate groups (Reviewers, Reviewers/Submitted, etc.)
    if '/Submission' in gid or '/Paper' in gid:
        parts = gid.split('/')
        role = parts[-1]
        if not any(role.startswith(k) for k in ['Reviewer_', 'Area_Chair_', 'Senior_Area_Chair_']):
            continue
        venue = '/'.join(parts[:-2])
        sub = parts[-2]
        year = next((p for p in parts if p.isdigit() and len(p) == 4), '')
        conf = parts[0] + (f' {year}' if year else '')
        is_workshop = 'Workshop' in gid
        paper_assignments.append({
            'conference': conf, 'venue': venue, 'year': year,
            'submission': sub, 'role': role, 'is_workshop': is_workshop,
        })

service_groups = sorted(set(service_groups))
paper_assignments_dedup = {a['venue'] + '/' + a['submission'] + '/' + a['role']: a for a in paper_assignments}
paper_assignments = sorted(paper_assignments_dedup.values(), key=lambda x: (x['conference'], x['submission']))
print(f"  -> {len(service_groups)} venue-level service roles")
print(f"  -> {len(paper_assignments)} per-paper assignments")

# ---------- 4. Public profile snapshot ----------
print("\nFetching profile ...")
profile = client.get_profile(email_or_id=PROFILE_ID)
profile_snapshot = {
    'id': profile.id,
    'names': profile.content.get('names'),
    'emails_confirmed': profile.content.get('emailsConfirmed') or profile.content.get('preferredEmail'),
    'history': profile.content.get('history'),
    'expertise': profile.content.get('expertise'),
    'relations': profile.content.get('relations'),
}

# ---------- 5. Parse and write outputs ----------
def parse_service(gid):
    parts = gid.split('/')
    role = parts[-1]
    venue = '/'.join(parts[:-1])
    year = next((p for p in parts if p.isdigit() and len(p) == 4), '')
    venue_short = parts[0] + (f' {year}' if year else '')
    return {'venue_id': venue, 'venue_short': venue_short, 'year': year, 'role': role, 'group_id': gid}

pd.DataFrame([parse_service(g) for g in service_groups]).to_csv(OUTPUT_DIR / 'reviewer_groups.csv', index=False)
pd.DataFrame(paper_assignments).to_csv(OUTPUT_DIR / 'reviewer_assignments.csv', index=False)
pd.DataFrame(reviews).to_csv(OUTPUT_DIR / 'reviews_authored.csv', index=False)

with open(OUTPUT_DIR / 'reviewer_export.json', 'w') as f:
    json.dump({
        'profile': profile_snapshot,
        'service_groups': service_groups,
        'paper_assignments': paper_assignments,
        'reviews': reviews,
        'exported_at': datetime.now(timezone.utc).isoformat(),
    }, f, indent=2, default=str)

# ---------- 6. Print summary ----------
print(f"\n=== SUMMARY ===")
print(f"Profile           : {PROFILE_ID}")
print(f"Service roles     : {len(service_groups)}")
print(f"Reviews authored  : {len(reviews)}")

conf_counts = Counter()
for a in paper_assignments:
    if not a['is_workshop']:
        conf_counts[a['conference']] += 1
if conf_counts:
    print("\nPapers reviewed per conference (excl. workshops):")
    for conf in sorted(conf_counts):
        print(f"  {conf:30s}  {conf_counts[conf]} papers")
    print(f"  {'TOTAL':30s}  {sum(conf_counts.values())} papers")

print(f"\nWrote to {OUTPUT_DIR}/:")
print("  reviewer_groups.csv")
print("  reviewer_assignments.csv")
print("  reviews_authored.csv")
print("  reviewer_export.json")
