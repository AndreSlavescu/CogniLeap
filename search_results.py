import requests
import os

# api key
metaphor_api_key = os.getenv("METAPHOR_API_KEY")


# search results
def search_results(query: str):
    """
    Returns search results for a query and all important context generated
    """
    url = "https://api.metaphor.systems/search"

    payload = {
        "query": query,
        "useAutoprompt": True,
        "type": "neural",
        "numResults": 3,
        "includeDomains": ["https://github.com/", "https://arxiv.org/"]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": metaphor_api_key,
    }
    response = requests.post(url, json=payload, headers=headers)
    search_response = response.json()

    search_results_array = []

    for result in search_response["results"]:
        title = result["title"]
        source = result["url"]
        score = result["score"]
        search_results_array.append((title, source, score))

    return search_results_array
