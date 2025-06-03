import requests
from bs4 import BeautifulSoup
import sqlite3
from datetime import datetime, timezone
import time

SOURCES = [
    {
        "name": "Le Matin",
        "rss": "https://lematin.ma/rssFeed/4",
        "article_selector": "div.article-content.content"
    },
    {
        "name": "L'Economiste",
        "rss": "https://www.leconomiste.com/rss-leconomiste",
        "article_selector": "div#content_leconomiste"
    }
]

def fetch_full_article(url, selector):
    """
    Fetches the full article text from the news link using the given CSS selector.
    Joins all strings found in the selector, so bold or inline tags do not cause line breaks.
    """
    try:
        resp = requests.get(url, timeout=15)
        soup = BeautifulSoup(resp.content, "html.parser")
        article_div = soup.select_one(selector)
        if not article_div:
            # Try fallback: get all <p> tags as a last resort
            paragraphs = soup.find_all('p')
            if paragraphs:
                return " ".join(p.get_text(strip=True) for p in paragraphs)
            return ""
        # Remove script/style tags
        for tag in article_div(['script', 'style']):
            tag.decompose()
        # Join every string in the content, so inline tags (like <b>) don't add newlines
        article_text = " ".join(article_div.stripped_strings)
        return article_text
    except Exception as e:
        print(f"Error fetching article at {url}: {e}")
        return ""

def parse_rss_item(item, source_name, article_selector):
    """
    Parses an individual RSS <item> tag and fetches its full description.
    """
    title = item.title.text.strip() if item.title else ""
    link = item.link.text.strip() if item.link else ""
    pub_date = item.pubDate.text.strip() if item.pubDate else ""
    # Fetch full description by webscraping
    description = fetch_full_article(link, article_selector) if link else ""
    # Respectful pause to avoid hammering the site
    time.sleep(1)  # 1 second pause between requests
    return {
        "source": source_name,
        "title": title,
        "link": link,
        "date": pub_date,
        "description": description
    }

def fetch_rss_articles(source):
    """
    Fetches articles from the RSS feed of a given source.
    """
    print(f"[INFO] Fetching RSS from {source['name']}...")
    resp = requests.get(source['rss'], timeout=15)
    soup = BeautifulSoup(resp.content, features='xml')
    articles = []
    for item in soup.find_all('item'):
        article = parse_rss_item(item, source['name'], source['article_selector'])
        articles.append(article)
    return articles

def save_to_db(articles, db_name="morocco_financial_news.db"):
    """
    Saves new articles to the database, logs what is added,
    and notifies if nothing was added.
    """
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT,
        title TEXT,
        link TEXT UNIQUE,
        date TEXT,
        description TEXT,
        fetched_at TEXT
    )''')
    added_count = 0
    for art in articles:
        try:
            c.execute(
                "INSERT OR IGNORE INTO news (source, title, link, date, description, fetched_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    art['source'],
                    art['title'],
                    art['link'],
                    art['date'],
                    art['description'],
                    datetime.now(timezone.utc).isoformat()
                )
            )
            if c.rowcount > 0:
                # Something was added
                print(f"[ADDED] {art['source']} | {art['title'][:60]}... | {art['link']}")
                added_count += 1
        except Exception as e:
            print(f"[ERROR] Error inserting article: {e}")
    conn.commit()
    conn.close()
    if added_count == 0:
        print("[INFO] No new articles were added to the database.")
    else:
        print(f"[INFO] {added_count} new article(s) added to the database.")

def main():
    """
    Main execution function.
    """
    all_articles = []
    for source in SOURCES:
        try:
            articles = fetch_rss_articles(source)
            all_articles.extend(articles)
        except Exception as e:
            print(f"[ERROR] Error fetching {source['name']}: {e}")
    print(f"[INFO] Fetched {len(all_articles)} articles in total.")
    save_to_db(all_articles)

if __name__ == "__main__":
    main()