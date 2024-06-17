# filename: fetch_arxiv_gpt4.py
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

def search_arxiv(query):
    url = 'http://export.arxiv.org/api/query?'
    params = {
        'search_query': query,
        'start': 0,
        'max_results': 5,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    query_string = urllib.parse.urlencode(params)
    url += query_string
    with urllib.request.urlopen(url) as response:
        response_text = response.read()
    return response_text

def parse_response(response):
    root = ET.fromstring(response)
    papers = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        published = entry.find('{http://www.w3.org/2005/Atom}published').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        papers.append({'title': title, 'published': published, 'summary': summary})
    return papers

def main():
    query = 'all:"GPT-4"'
    response = search_arxiv(query)
    papers = parse_response(response)
    if papers:
        print("Most Recent Paper on GPT-4:")
        print("Title:", papers[0]['title'])
        print("Published Date:", papers[0]['published'])
        print("Summary:", papers[0]['summary'])
    else:
        print("No papers found.")

if __name__ == '__main__':
    main()