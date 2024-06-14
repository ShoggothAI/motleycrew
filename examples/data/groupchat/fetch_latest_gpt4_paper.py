# filename: fetch_latest_gpt4_paper.py
import requests
from datetime import datetime

def fetch_latest_paper():
    # Define the API endpoint
    url = "http://export.arxiv.org/api/query"
    
    # Set the search parameters to find papers related to GPT-4
    params = {
        "search_query": "all:GPT-4",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": 1
    }
    
    # Send a GET request to the API
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        # Parse the response XML
        from xml.etree import ElementTree as ET
        root = ET.fromstring(response.content)
        
        # Navigate to the entry element
        entry = root.find('{http://www.w3.org/2005/Atom}entry')
        if entry is not None:
            # Extract title and summary (abstract)
            title = entry.find('{http://www.w3.org/2005/Atom}title').text
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
            published_date = entry.find('{http://www.w3.org/2005/Atom}published').text
            
            # Convert published date to a readable format
            published_datetime = datetime.strptime(published_date, '%Y-%m-%dT%H:%M:%SZ')
            
            print("Title:", title)
            print("Published Date:", published_datetime.strftime('%Y-%m-%d'))
            print("Abstract:", summary.strip())
        else:
            print("No GPT-4 papers found.")
    else:
        print("Failed to fetch data from arXiv. Status code:", response.status_code)

fetch_latest_paper()