"""
Web scraper to collect Moroccan financial news from specific financial sections
with improved article detection and content filtering.
"""
import datetime
import time
import random
import re
import requests
from bs4 import BeautifulSoup
from newspaper import Article, Config
import pandas as pd
from urllib.parse import urljoin, urlparse
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COMPANIES, RAW_DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("scraper")

# Updated news sources based on supervisor's recommendations
NEWS_SOURCES = [
    {
        "name": "Boursenews",
        "url": "https://boursenews.ma/articles/marches",
        "language": "fr"
    },
    {
        "name": "FLM",
        "url": "https://flm.ma/economie.php",
        "language": "fr"
    },
    {
        "name": "Le Desk",
        "url": "https://ledesk.ma/encontinu/",
        "language": "fr"
    },
    {
        "name": "H24Info",
        "url": "https://www.h24info.ma/economie/",
        "language": "fr"
    },
    {
        "name": "Le Matin",
        "url": "https://lematin.ma/economie",
        "language": "fr"
    },
    {
        "name": "Medias24",
        "url": "https://medias24.com/categorie/leboursier/",
        "language": "fr"
    },
    {
        "name": "Finance News - Financial",
        "url": "https://fnh.ma/articles/actualite-financiere-maroc",
        "language": "fr"
    },
    {
        "name": "Finance News - Economic",
        "url": "https://fnh.ma/articles/actualite-economique",
        "language": "fr"
    }
]

def normalize_text(text):
    """
    Normalize article text by converting it to a single line.
    
    Args:
        text (str): Raw article text with potential line breaks
        
    Returns:
        str: Normalized text as a single line
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Replace all line breaks with spaces
    normalized = re.sub(r'\s*\n\s*', ' ', text)
    
    # Replace multiple spaces with a single space
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove any leading/trailing whitespace
    normalized = normalized.strip()
    
    return normalized

def is_likely_article_url(url, base_url):
    """
    Determine if a URL is likely to be an article based on various heuristics.
    
    Args:
        url (str): URL to check
        base_url (str): Base URL of the news source
        
    Returns:
        bool: True if the URL is likely an article, False otherwise
    """
    # Basic URL validation
    if not url or url == "#" or url == "/" or not isinstance(url, str):
        return False
    
    # Get domain of the base URL
    base_domain = urlparse(base_url).netloc
    
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Check if URL is from the same domain
    if parsed_url.netloc and parsed_url.netloc != base_domain:
        return False
    
    # Skip obvious navigation, category, tag, or utility pages
    skip_patterns = [
        '/tag/', '/category/', '/author/', '/search/', '/contact/', 
        '/about/', '/archives/', '/sitemap/', '/privacy/', '/terms/',
        'javascript:', 'mailto:', 'whatsapp:', 'facebook.com', 'twitter.com',
        'instagram.com', 'youtube.com', 'linkedin.com', '/rss/', '/feed/',
        '/abonnement', '/subscription', '/login', '/register', '/mon-compte',
        '/profile', '/user', '/newsletter', '/flux-rss', '/publicite',
        '/cgu', '/faq', '/aide', '/help', '/mentions-legales', '/confidentialite',
        '/bourse-de-casablanca', '/bourse', '/cotations', '/indices',
        '/activites-royales', '/nation', '/monde', '/societe', '/culture', 
        '/sports', '/videos', '/regions'
    ]
    
    if any(pattern in url.lower() for pattern in skip_patterns):
        return False
    
    # Look for article indicators in the URL
    article_indicators = [
        '/article/', '/news/', '/actualite/', '/economie/', '/finance/', 
        '/bourse/', '/entreprise/', '/marches/', '/banque/', '/industrie/',
        '/agriculture/', '/energie/', '/tourisme/', '/transport/', '/immobilier/',
        '/tech/', '/startup/', '/pme/', '/export/', '/investissement/',
        '.html', '.php?id=', '/20', '/id='
    ]
    
    # URL must contain at least one article indicator
    if not any(indicator in url.lower() for indicator in article_indicators):
        return False
    
    # Ensure URL has path components and is not just domain
    if not parsed_url.path or parsed_url.path == '/':
        return False
    
    # Check for pagination links
    if 'page=' in url or '/page/' in url:
        return False
    
    # Check for minimum path length (articles usually have longer paths)
    if len(parsed_url.path.split('/')) < 3:
        return False
    
    return True

def fetch_with_retry(url, headers, max_retries=3, initial_timeout=15):
    """
    Fetch a URL with retry logic for better reliability.
    
    Args:
        url (str): URL to fetch
        headers (dict): Request headers
        max_retries (int): Maximum number of retry attempts
        initial_timeout (int): Initial timeout in seconds
        
    Returns:
        requests.Response or None: Response object or None if all retries failed
    """
    timeout = initial_timeout
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed for {url}: {e}")
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                # Exponential backoff with jitter
                sleep_time = timeout * (1.5 + random.random() * 0.5)
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                # Increase timeout for next attempt
                timeout = min(timeout * 1.5, 45)  # Cap at 45 seconds
            else:
                logger.error(f"All retries failed for {url}")
                return None
    
    return None

class NewsArticle:
    def __init__(self, title, text, url, date, source):
        self.title = title
        # Normalize text to ensure it's on a single line
        self.text = normalize_text(text)
        self.url = url
        self.date = date
        self.source = source
        self.companies = self._detect_companies()
        
    def _detect_companies(self):
        """Detect which companies are mentioned in the article."""
        mentioned_companies = []
        text_to_search = f"{self.title} {self.text}".lower()
        
        for company in COMPANIES:
            for keyword in company["keywords"]:
                if keyword.lower() in text_to_search:
                    mentioned_companies.append(company["name"])
                    break
        
        return mentioned_companies
    
    def to_dict(self):
        return {
            "title": self.title,
            "text": self.text,
            "url": self.url,
            "date": self.date,
            "source": self.source,
            "companies": ",".join(self.companies),
        }
    
    def is_valid_article(self):
        """
        Check if this is a valid article with meaningful content.
        
        Returns:
            bool: True if the article is valid, False otherwise
        """
        # Article must have a title
        if not self.title or len(self.title) < 10:
            return False
        
        # Article must have text content
        if not self.text or len(self.text) < 100:
            return False
        
        # Skip articles with very short titles that are likely not real articles
        if len(self.title) < 15:
            return False
        
        # Skip articles with common non-article titles
        skip_titles = [
            "accueil", "actualités", "économie", "finance", "bourse", 
            "cotations", "indices", "abonnement", "subscription", "login",
            "découvrez", "vidéo", "société", "culture", "sport", "vidéos", 
            "contact", "à propos", "mentions légales", "cgu", "confidentialité"
        ]
        
        if any(skip.lower() in self.title.lower() for skip in skip_titles):
            return False
        
        # Check if text has sentences (at least 3 periods)
        if self.text.count('.') < 3:
            return False
        
        # Check if text is mostly navigation or menu items
        nav_indicators = ["accueil", "actualités", "économie", "finance", "bourse", 
                         "contact", "à propos", "mentions légales", "cgu", "confidentialité"]
        
        nav_count = sum(1 for indicator in nav_indicators if indicator.lower() in self.text.lower())
        if nav_count > 3:  # If more than 3 navigation terms appear, it's likely a navigation page
            return False
        
        return True

class NewsScraperBase:
    def __init__(self, source_config):
        self.name = source_config["name"]
        self.base_url = source_config["url"]
        self.language = source_config["language"]
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        # Configure newspaper settings
        self.config = Config()
        self.config.request_timeout = 20  # Increased timeout
        self.config.browser_user_agent = self.headers["User-Agent"]
        self.config.number_threads = 1  # Lower thread count to avoid overwhelming sites
    
    def get_article_urls(self):
        """Get article URLs from the source. To be implemented by subclasses."""
        try:
            urls = []
            # Access the specific section page
            response = fetch_with_retry(self.base_url, self.headers)
            if not response:
                return []
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Find all links - focus on those within article containers if possible
            article_containers = soup.select("article, .article, .news-item, .post, .entry, .card")
            
            if article_containers:
                # If we found article containers, extract links from them
                for container in article_containers:
                    links = container.find_all("a", href=True)
                    for link in links:
                        url = link["href"]
                        
                        # Make absolute URL if relative
                        if not url.startswith(('http://', 'https://')):
                            base_domain = re.match(r'(https?://[^/]+)', self.base_url).group(1)
                            if url.startswith('/'):
                                url = base_domain + url
                            else:
                                url = urljoin(self.base_url, url)
                        
                        # Check if URL is likely an article
                        if is_likely_article_url(url, self.base_url) and url not in urls:
                            urls.append(url)
            else:
                # Fallback: look for links that match article patterns
                links = soup.find_all("a", href=True)
                for link in links:
                    url = link["href"]
                    
                    # Make absolute URL if relative
                    if not url.startswith(('http://', 'https://')):
                        base_domain = re.match(r'(https?://[^/]+)', self.base_url).group(1)
                        if url.startswith('/'):
                            url = base_domain + url
                        else:
                            url = urljoin(self.base_url, url)
                    
                    # Check if URL is likely an article
                    if is_likely_article_url(url, self.base_url) and url not in urls:
                        urls.append(url)
            
            # Remove duplicates and limit to 20 articles
            urls = list(set(urls))[:20]
            
            # Log the URLs we're going to process
            for url in urls:
                logger.debug(f"Found potential article URL: {url}")
            
            return urls
        except Exception as e:
            logger.error(f"Error getting article URLs from {self.name}: {e}")
            return []
    
    def parse_article(self, url):
        """Parse an article from the given URL."""
        try:
            # Create article with config that has timeout set
            article = Article(url, language=self.language, config=self.config)
            article.download()
            article.parse()
            
            # Some sources might not have proper date extraction via newspaper3k
            date = article.publish_date
            if not date:
                date = datetime.datetime.now()
            
            # Create NewsArticle object
            news_article = NewsArticle(
                title=article.title,
                text=article.text,
                url=url,
                date=date,
                source=self.name
            )
            
            # Check if the article is valid
            if not news_article.is_valid_article():
                logger.warning(f"Skipping {url} - Not a valid article")
                return None
                
            return news_article
        except Exception as e:
            logger.error(f"Error parsing article {url}: {e}")
            return None

class BoursenewsScraper(NewsScraperBase):
    """Specialized scraper for Boursenews market section"""
    def get_article_urls(self):
        try:
            urls = []
            response = fetch_with_retry(self.base_url, self.headers)
            if not response:
                return []
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Try multiple selectors specific to Boursenews
            article_selectors = [
                ".listing-actualite .item a",
                ".article-preview a",
                ".card-article a",
                ".latest-news a",
                ".featured-articles a",
                ".list-articles a",
                "a.read-more"
            ]
            
            for selector in article_selectors:
                links = soup.select(selector)
                for link in links:
                    if 'href' in link.attrs:
                        url = link['href']
                        if not url.startswith('http'):
                            url = "https://boursenews.ma" + url
                        if url not in urls and is_likely_article_url(url, self.base_url):
                            urls.append(url)
            
            # If no links found with selectors, try a more general approach
            if not urls:
                # Look for any link that contains article in the URL
                all_links = soup.find_all("a", href=True)
                for link in all_links:
                    url = link["href"]
                    if not url.startswith('http'):
                        url = "https://boursenews.ma" + url
                    
                    # Check for article indicators specific to Boursenews
                    if ("/article/" in url or "/actualite/" in url or "/marches/" in url) and url not in urls:
                        urls.append(url)
            
            return list(set(urls))[:20]
        except Exception as e:
            logger.error(f"Error getting article URLs from {self.name}: {e}")
            return []

class FLMScraper(NewsScraperBase):
    """Specialized scraper for FLM economy section"""
    def get_article_urls(self):
        try:
            urls = []
            response = fetch_with_retry(self.base_url, self.headers)
            if not response:
                return []
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # FLM uses a PHP-based structure - we need to handle URLs differently
            # First look for article blocks
            article_blocks = soup.select(".article-block, .news-item, .post-item, .article-item")
            
            if article_blocks:
                for block in article_blocks:
                    links = block.find_all("a", href=True)
                    for link in links:
                        href = link["href"]
                        # Correct URL construction - FLM uses PHP files with query parameters
                        if href.startswith("http"):
                            url = href
                        elif "?" in href:  # PHP file with query parameters
                            # Extract the PHP file name and parameters
                            path_parts = href.split("?")
                            if len(path_parts) > 1:
                                php_file = path_parts[0]
                                params = path_parts[1]
                                url = f"https://flm.ma/{php_file}?{params}"
                            else:
                                url = f"https://flm.ma/{href}"
                        else:
                            # Regular path
                            url = f"https://flm.ma/{href}" if not href.startswith("/") else f"https://flm.ma{href}"
                        
                        if url not in urls:
                            urls.append(url)
            
            # If no article blocks found, try a different approach
            if not urls:
                # Look for links with ID parameters which are common in PHP sites
                php_links = []
                all_links = soup.find_all("a", href=True)
                
                for link in all_links:
                    href = link["href"]
                    
                    # Match PHP files with ID parameters
                    if ".php" in href and ("id=" in href or "article_id=" in href or "news_id=" in href):
                        php_links.append(link)
                
                for link in php_links:
                    href = link["href"]
                    # Correct URL construction for PHP files
                    if href.startswith("http"):
                        url = href
                    elif "?" in href:
                        path_parts = href.split("?")
                        if len(path_parts) > 1:
                            php_file = path_parts[0]
                            params = path_parts[1]
                            url = f"https://flm.ma/{php_file}?{params}"
                        else:
                            url = f"https://flm.ma/{href}"
                    else:
                        url = f"https://flm.ma/{href}" if not href.startswith("/") else f"https://flm.ma{href}"
                    
                    if url not in urls:
                        urls.append(url)
            
            # Third approach - look for specific patterns in all links
            if not urls:
                all_links = soup.find_all("a", href=True)
                for link in all_links:
                    href = link["href"]
                    
                    # Look for specific patterns that suggest article links
                    if (("economie" in href.lower() or "finance" in href.lower() or 
                         "business" in href.lower() or "maroc" in href.lower()) and
                        (".php" in href or ".html" in href)):
                        
                        # Properly format the URL
                        if href.startswith("http"):
                            url = href
                        else:
                            # Handle different path formats
                            if href.startswith("/"):
                                url = f"https://flm.ma{href}"
                            else:
                                url = f"https://flm.ma/{href}"
                        
                        if url not in urls:
                            urls.append(url)
            
            # Check all URLs for validity before returning
            valid_urls = []
            for url in urls:
                # Skip URLs with incorrect domain structure
                if not ("flm.ma" in url or url.startswith("https://flm.ma")):
                    continue
                
                # Fix common URL structure issues
                if "flm.ma" in url and not url.startswith("https://flm.ma"):
                    url = "https://" + url.split("flm.ma")[1]
                
                valid_urls.append(url)
            
            logger.info(f"Found {len(valid_urls)} potential articles on FLM")
            return valid_urls[:20]
        
        except Exception as e:
            logger.error(f"Error getting article URLs from {self.name}: {e}")
            return []

class H24InfoScraper(NewsScraperBase):
    """Specialized scraper for H24Info economy section"""
    def get_article_urls(self):
        try:
            urls = []
            response = fetch_with_retry(self.base_url, self.headers)
            if not response:
                return []
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # H24Info specific article selectors
            article_selectors = [
                ".post-card a",
                ".entry-title a",
                ".economie-article a",
                ".actualite-item a",
                ".post-thumbnail a",
                ".post-title a",
                ".card-content a"
            ]
            
            for selector in article_selectors:
                links = soup.select(selector)
                for link in links:
                    if 'href' in link.attrs:
                        url = link['href']
                        if not url.startswith('http'):
                            url = "https://www.h24info.ma" + url
                        if url not in urls and is_likely_article_url(url, self.base_url):
                            urls.append(url)
            
            # Try another approach: look for articles in the main content area
            if not urls:
                content_area = soup.select("#main-content, .main-content, .content-area, .site-content")
                if content_area:
                    for container in content_area:
                        links = container.find_all("a", href=True)
                        for link in links:
                            url = link["href"]
                            if not url.startswith('http'):
                                url = "https://www.h24info.ma" + url
                            
                            # H24Info likely uses WordPress, so look for typical WordPress article URLs
                            if "/20" in url and url not in urls:  # URLs typically include year
                                urls.append(url)
            
            # Direct article extraction based on URL patterns
            if not urls:
                all_links = soup.find_all("a", href=True)
                for link in all_links:
                    url = link["href"]
                    if not url.startswith('http'):
                        url = "https://www.h24info.ma" + url
                    
                    # H24Info URL patterns
                    if ("/economie/" in url and url != self.base_url and 
                        not any(skip in url for skip in ['/category/', '/tag/']) and
                        url not in urls):
                        urls.append(url)
            
            return list(set(urls))[:20]
        except Exception as e:
            logger.error(f"Error getting article URLs from {self.name}: {e}")
            return []

class LeMatinScraper(NewsScraperBase):
    """Specialized scraper for Le Matin economy section"""
    def get_article_urls(self):
        try:
            urls = []
            response = fetch_with_retry(self.base_url, self.headers)
            if not response:
                return []
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Look for article containers
            article_containers = soup.select(".post, .article-item, .list-item, .card-article")
            
            if article_containers:
                for container in article_containers:
                    link = container.find("a", href=True)
                    if link:
                        url = link["href"]
                        if not url.startswith('http'):
                            url = "https://lematin.ma" + url
                        
                        # Only consider URLs with article-like paths
                        if "/article/" in url or url.endswith(".html") or "/economie/" in url:
                            urls.append(url)
            
            # If not found yet, try looking for article headlines
            if not urls:
                headlines = soup.select("h2.post-title, h3.entry-title, .post-heading, .article-title")
                for headline in headlines:
                    link = headline.find("a", href=True)
                    if link:
                        url = link["href"]
                        if not url.startswith('http'):
                            url = "https://lematin.ma" + url
                        
                        # Skip non-article links
                        if "/economie/" in url and is_likely_article_url(url, self.base_url):
                            urls.append(url)
            
            # Final fallback - look for specific patterns in links
            if not urls:
                links = soup.find_all("a", href=True)
                for link in links:
                    url = link["href"]
                    if not url.startswith('http'):
                        url = "https://lematin.ma" + url
                    
                    # Look for specific patterns in URL that indicate articles
                    if ("/economie/" in url and 
                        not any(skip in url for skip in ['/abonnement', '/videos', '/bourse']) and
                        len(url.split('/')) >= 5):  # URL must have enough segments to be an article
                        urls.append(url)
            
            return list(set(urls))[:20]
        except Exception as e:
            logger.error(f"Error getting article URLs from {self.name}: {e}")
            return []

class FinanceNewsScraper(NewsScraperBase):
    """Specialized scraper for Finance News sections"""
    def get_article_urls(self):
        try:
            urls = []
            response = fetch_with_retry(self.base_url, self.headers, max_retries=4)
            if not response:
                return []
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Find article links - they are typically in article cards
            article_links = soup.select(".card__link, .card-content a, .article-card a, .card a")
            
            for link in article_links:
                url = link["href"]
                if not url.startswith('http'):
                    url = "https://fnh.ma" + url
                
                # Only add URLs that look like articles
                if "/article/" in url and url not in urls:
                    urls.append(url)
            
            # If no article links found, fallback to generic approach
            if not urls:
                links = soup.find_all("a", href=True)
                for link in links:
                    url = link["href"]
                    if "/article/" in url and url not in urls:
                        if not url.startswith('http'):
                            url = "https://fnh.ma" + url
                        urls.append(url)
            
            return list(set(urls))[:20]
        except Exception as e:
            logger.error(f"Error getting article URLs from {self.name}: {e}")
            return []

class NewsScraperFactory:
    @staticmethod
    def get_scraper(source_config):
        name = source_config["name"]
        url = source_config["url"]
        
        if "boursenews.ma" in url:
            return BoursenewsScraper(source_config)
        elif "flm.ma" in url:
            return FLMScraper(source_config)
        elif "h24info.ma" in url:
            return H24InfoScraper(source_config)
        elif "lematin.ma" in url:
            return LeMatinScraper(source_config)
        elif "fnh.ma" in url:
            return FinanceNewsScraper(source_config)
        else:
            # Use base class for other sources for now
            return NewsScraperBase(source_config)

def collect_news():
    """Collect news from all configured sources."""
    all_articles = []
    
    for source in NEWS_SOURCES:
        logger.info(f"Collecting news from {source['name']}...")
        scraper = NewsScraperFactory.get_scraper(source)
        
        article_urls = scraper.get_article_urls()
        logger.info(f"Found {len(article_urls)} potential articles from {source['name']}")
        
        article_count = 0
        for url in article_urls:
            # Add random delay to avoid overloading the server
            time.sleep(random.uniform(1, 3))
            article = scraper.parse_article(url)
            if article:
                all_articles.append(article.to_dict())
                article_count += 1
        
        logger.info(f"Successfully parsed {article_count} valid articles from {source['name']}")
    
    # Save collected articles to a single file that gets appended to
    if all_articles:
        # Create a DataFrame with new articles
        new_articles_df = pd.DataFrame(all_articles)
        
        # Define a fixed file path for the single CSV file
        file_path = RAW_DATA_DIR / "moroccan_financial_news.csv"
        
        # Check if the file already exists
        if os.path.exists(file_path):
            # Read existing data
            existing_df = pd.read_csv(file_path)
            
            # Filter out duplicates (based on URL)
            existing_urls = set(existing_df['url']) if 'url' in existing_df.columns else set()
            new_articles_df = new_articles_df[~new_articles_df['url'].isin(existing_urls)]
            
            if len(new_articles_df) > 0:
                # Fix the FutureWarning by ensuring consistent dtypes before concatenation
                # First, ensure all columns from both dataframes are present in each
                for col in existing_df.columns:
                    if col not in new_articles_df.columns:
                        new_articles_df[col] = None
                
                for col in new_articles_df.columns:
                    if col not in existing_df.columns:
                        existing_df[col] = None
                
                # Ensure column order is the same
                new_articles_df = new_articles_df[existing_df.columns]
                
                # Now concatenate with consistent dtypes
                combined_df = pd.concat([existing_df, new_articles_df], ignore_index=True)
                combined_df.to_csv(file_path, index=False)
                logger.info(f"Added {len(new_articles_df)} new articles to {file_path}")
            else:
                logger.info("No new articles to add.")
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Save as new file
            new_articles_df.to_csv(file_path, index=False)
            logger.info(f"Created new file with {len(new_articles_df)} articles at {file_path}")
    else:
        logger.warning("No articles were collected.")

if __name__ == "__main__":
    collect_news()