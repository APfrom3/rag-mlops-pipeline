import requests
from bs4 import BeautifulSoup
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Motley Fool transcript URLs - these are publicly available
TRANSCRIPTS = [
    {
        "ticker": "GOOGL",
        "company": "Alphabet",
        "year": 2024,
        "quarter": 3,
        "url": "https://www.fool.com/earnings/call-transcripts/2024/10/29/alphabet-goog-q3-2024-earnings-call-transcript/"
    },
    {
        "ticker": "NVDA",
        "company": "NVIDIA",
        "year": 2024,
        "quarter": 3,
        "url": "https://www.fool.com/earnings/call-transcripts/2024/11/20/nvidia-nvda-q3-2024-earnings-call-transcript/"
    },
    {
        "ticker": "META",
        "company": "Meta",
        "year": 2024,
        "quarter": 3,
        "url": "https://www.fool.com/earnings/call-transcripts/2024/10/30/meta-platforms-meta-q3-2024-earnings-call-transcri/"
    },
    {
        "ticker": "AMZN",
        "company": "Amazon",
        "year": 2024,
        "quarter": 3,
        "url": "https://www.fool.com/earnings/call-transcripts/2024/10/31/amazoncom-amzn-q3-2024-earnings-call-transcript/"
    }
]

def scrape_transcript(url: str) -> str:
    """
    Scrapes the transcript text from a Motley Fool earnings call page.
    """
    headers = {
        # We set a User-Agent so the request looks like it's coming
        # from a real browser. Without this many sites block scrapers.
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch page: HTTP {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Motley Fool puts transcript content in <p> tags inside the article body
    article = soup.find("div", class_="article-body")

    if article is None:
        raise Exception("Could not find article body — page structure may have changed")

    paragraphs = article.find_all("p")
    full_text = "\n\n".join([p.get_text() for p in paragraphs])

    return full_text

def fetch_transcripts(output_dir: str = "data/transcripts"):
    """
    Fetches earnings call transcripts and saves them as JSON files locally.
    Skips files that already exist so we don't re-scrape unnecessarily.
    """
    os.makedirs(output_dir, exist_ok=True)

    for entry in TRANSCRIPTS:
        ticker = entry["ticker"]
        output_path = os.path.join(output_dir, f"{ticker}_Q3_2024.json")

        # Skip if we already have it
        if os.path.exists(output_path):
            print(f"Already have {ticker}, skipping.")
            continue

        print(f"Fetching transcript for {ticker}...")
        try:
            text = scrape_transcript(entry["url"])

            with open(output_path, "w") as f:
                json.dump({
                    "ticker": ticker,
                    "company": entry["company"],
                    "year": entry["year"],
                    "quarter": entry["quarter"],
                    "text": text
                }, f, indent=2)

            print(f"  Saved to {output_path}")

        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")

if __name__ == "__main__":
    fetch_transcripts()