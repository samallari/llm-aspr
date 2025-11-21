"""
Uses OpenReview API to scrape papers, reviews, and decisions from OpenReview to create a structured dataset.
"""
from openreview.api import OpenReviewClient
import json

REQUEST_LIMIT = 5

def get_conference_data(venue_id, year):
    # Initialize client (uses OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD env vars if set)
    client = OpenReviewClient(
        baseurl='https://api2.openreview.net'
    )

    # 1. Get all submissions
    print("Fetching submissions...")
    submissions = list(client.get_all_notes(
        invitation=f'{venue_id}/{year}/Conference/-/Submission'
    ))[:REQUEST_LIMIT]
    print(f"Found {len(submissions)} submissions")

    all_data = []

    for i, submission in enumerate(submissions):
        print(f"Processing paper {i+1}/{len(submissions)}: {submission.id}")

        paper_data = {
            'venue': venue_id,
            'year': year,
            'paper_id': submission.id,
            'title': submission.content.get('title', {}).get('value'),
            'reviews': [],
            'decision': None
        }

        # 2. Get reviews
        reviews = client.get_all_notes(
            forum=submission.id,
            invitation=f'{venue_id}/{year}/Conference/Submission{submission.number}/-/Official_Review'
        )

        for review in reviews:
            content = review.content
            paper_data['reviews'].append({
                'summary': content.get('summary', {}).get('value'),
                'strengths': content.get('strengths', {}).get('value'),
                'weaknesses': content.get('weaknesses', {}).get('value'),
                'questions': content.get('questions', {}).get('value'),
                'rating': content.get('rating', {}).get('value'),
                'confidence': content.get('confidence', {}).get('value')
            })

        # 3. Get decision
        decisions = client.get_all_notes(
            forum=submission.id,
            invitation=f'{venue_id}/{year}/Conference/Submission{submission.number}/-/Decision'
        )

        for decision in decisions:
            paper_data['decision'] = decision.content.get('decision', {}).get('value')
            break

        all_data.append(paper_data)

    return all_data

# Usage
venue_id = "ICLR.cc"
year = "2024"
data = get_conference_data(venue_id, year)

# Save to JSON
with open('conference_data.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"\nTotal papers: {len(data)}")
accepted = sum(1 for p in data if p['decision'] and 'accept' in p['decision'].lower())
rejected = sum(1 for p in data if p['decision'] and 'reject' in p['decision'].lower())
print(f"Accepted: {accepted}")
print(f"Rejected: {rejected}")
