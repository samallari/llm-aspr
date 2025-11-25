"""
Uses OpenReview API to scrape papers, reviews, and decisions from OpenReview to create a structured dataset.
"""
from openreview.api import OpenReviewClient
import json
import time

# limit the number of papers to fetch for dataset creation
PAPER_LIMIT = 50

def get_conference_data(venue_id, year, verbose=True):
    # Initialize client
    client = OpenReviewClient(
        baseurl='https://api2.openreview.net'
    )

    # 1. API call to get all submissions
    print("Fetching submissions...")
    submissions = list(client.get_all_notes(
        invitation=f'{venue_id}/{year}/Conference/-/Submission'
    ))[:PAPER_LIMIT]
    print(f"Found {len(submissions)} submissions")

    all_data = []

    # for each submission, makes 2 API calls: one for reviews, one for decision
    for i, submission in enumerate(submissions):
        if verbose:
            print(f"Processing paper {i+1}/{len(submissions)}: {submission.id}")

        paper_data = {
            'venue': venue_id,
            'year': year,
            'paper_id': submission.id,
            'title': submission.content.get('title', {}).get('value'),
            'abstract': submission.content.get('abstract', {}).get('value'),
            'reviews': [],
            'decision': None
        }

        # 2. API call to get reviews
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

        # 3. API call to get decision
        decisions = client.get_all_notes(
            forum=submission.id,
            invitation=f'{venue_id}/{year}/Conference/Submission{submission.number}/-/Decision'
        )

        for decision in decisions:
            decision_value = decision.content.get('decision', {}).get('value')
            # clean up decision text
            if decision_value and 'Accept' in decision_value:
                paper_data['decision'] = 'Accept'
            else:
                paper_data['decision'] = 'Reject'
            break

        all_data.append(paper_data)
        time.sleep(2)  # Stay under 60 API calls/min rate limit

    return all_data

# Usage
if __name__ == "__main__":
    venue_id = "ICLR.cc"
    year = "2024"
    data = get_conference_data(venue_id, year, verbose=True)

    # Save to JSON
    results_file = 'data/ICLR2024_sample50.json'
    with open(results_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nTotal papers: {len(data)}")
    accepted = sum(1 for p in data if p['decision'] and 'accept' in p['decision'].lower())
    rejected = sum(1 for p in data if p['decision'] and 'reject' in p['decision'].lower())
    print(f"Accepted: {accepted}")
    print(f"Rejected: {rejected}")