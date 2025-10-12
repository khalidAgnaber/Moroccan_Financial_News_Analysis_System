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
import hashlib
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Setup directories and logging
RAW_DATA_DIR = Path("/Users/mac/Desktop/Moroccan_Financial_News_Analysis_System/data")
RAW_DATA_DIR.mkdir(exist_ok=True)

# where we actually store processed / labeled data
PROCESSED_DIR = RAW_DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LABELED_FILE = PROCESSED_DIR / "labeled_news_multi.csv"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('scraper.log'), logging.StreamHandler()]
)
logger = logging.getLogger("perfect_financial_scraper")

# Configure HTTP session with retries
session = requests.Session()
retries = Retry(total=5, backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))
session.mount("https://", HTTPAdapter(max_retries=retries))

# Data sources configuration
FINANCIAL_NEWS_SOURCES = [
    {
        "name": "Boursenews",
        "url": "https://boursenews.ma/articles/marches",
        "language": "fr",
        "finance_sections": ["/marches/", "/actualite-financiere/", "/bourse/"],
        "non_finance_sections": ["/sport/", "/societe/", "/politique/"]
    },
    {
        "name": "FLM",
        "url": "https://flm.ma/economie.php",
        "language": "fr",
        "finance_sections": ["/economie.php", "/finance.php", "/business.php"],
        "non_finance_sections": ["/sports.php", "/culture.php", "/politique.php"]
    },
    {
        "name": "Le Desk",
        "url": "https://ledesk.ma/thematiques/business/",
        "language": "fr",
        "finance_sections": ["/business/", "/economie/", "/finances/"],
        "non_finance_sections": ["/sport/", "/culture/", "/politique/"]
    },
    {
        "name": "H24Info",
        "url": "https://www.h24info.ma/economie/",
        "language": "fr",
        "finance_sections": ["/economie/", "/finance/", "/entreprises/"],
        "non_finance_sections": ["/sport/", "/maroc/", "/monde/"]
    },
    {
        "name": "Le Matin",
        "url": "https://lematin.ma/economie",
        "language": "fr",
        "finance_sections": ["/economie/", "/finance/", "/bourse/"],
        "non_finance_sections": ["/sport/", "/culture/", "/societe/"]
    },
    {
        "name": "Medias24",
        "url": "https://medias24.com/categorie/economie/",
        "language": "fr",
        "finance_sections": ["/economie/", "/finance/", "/leboursier/"],
        "non_finance_sections": ["/politique/", "/societe/", "/sport/"]
    },
    {
        "name": "Finance News - Financial",
        "url": "https://fnh.ma/articles/actualite-financiere-maroc",
        "language": "fr",
        "finance_sections": ["/actualite-financiere-maroc", "/bourse/", "/finance/"],
        "non_finance_sections": ["/opinions/", "/politique/", "/societe/"]
    },
    {
        "name": "Finance News - Economic",
        "url": "https://fnh.ma/articles/actualite-economique",
        "language": "fr",
        "finance_sections": ["/actualite-economique", "/economie/", "/entreprises/"],
        "non_finance_sections": ["/opinions/", "/politique/", "/societe/"]
    }
]

BRANCH_KEYWORDS = {
    "corporate finance": [
        'fusion', 'acquisition', 'rachat', 'cession', 'prise de participation', 'holding',
        'augmentation de capital', 'émission d\'actions', 'ipo', 'introduction en bourse',
        'offre publique', 'opa', 'ope', 'privatisation', 'nationalisation',
        'fusion-acquisition', 'spin-off', 'scission', 'offre publique de retrait', 'opr',
        'financement d\'entreprise', 'crédit corporate', 'prêt syndiqué', 'obligations corporates',
        'restructuration financière', 'refinancement', 'leveraged buyout', 'lbo',
        'corporate banking', 'banque d\'affaires', 'conseil en fusion',
        'conseil d\'administration', 'assemblée générale', 'gouvernance d\'entreprise',
        'résultats consolidés', 'bénéfice net', 'chiffre d\'affaires consolidé',
        'ebitda', 'cash-flow', 'endettement net', 'ratio financier', 'dividende',   "rachat", "acquisition", "cession", "fusion", "fusion-acquisition", "opa", "ope",
        "offre publique", "augmentation de capital", "levée de fonds", "levée", "ipo",
        "introduction en bourse", "émission d'actions", "émission d'obligations", "obligations",
        "prêt syndiqué", "syndiqué", "financement", "financement d'entreprise", "refinancement",
        "restructuration financière", "rachat", "takeover", "buyout", "lbo", "mbo", "mbi",
        "offre de rachat", "cession d'actifs", "cession de participation", "prise de participation",
        "capitalisation", "valorisation", "rating", "note de crédit", "coté en bourse",
        "consolidé", "résultat net", "chiffre d'affaires", "bénéfice", "ebitda", "masse salariale",
        "investisseur institutionnel", "actionnaire", "gouvernance", "assemblée générale",
        "dividende", "dividendes", "rapport financier", "conseil d'administration"
    ],
    "asset management": [
        'gestion d\'actifs', 'asset management', 'gestion de portefeuille', 'fonds d\'investissement',
        'sicav', 'fcp', 'opcvm', 'etf', 'tracker', 'fonds indiciel',
        'gestion collective', 'gestion institutionnelle', 'gestion privée', 'allocation d\'actifs',
        'diversification', 'gestion active', 'gestion passive', 'benchmark',
        'frais de gestion', 'frais de performance', 'assurance vie', 'pierre-papier', 'gestion de patrimoine',
        'performance des fonds', 'rendement des placements', 'value at risk', 'var', 'sharpe ratio',
        'volatilité', 'monétaire', 'alternatif', 'hedge fund',
        'fonds obligataire', 'fonds actions', 'fonds mixte', 'fonds immobilier', 'opc', 'organismes de placement collectif', "gestion d'actifs", "gestion d actifs", "asset management", "gestion de portefeuille",
        "fonds", "fonds d'investissement", "fonds immobilier", "sicav", "fcp", "opcvm",
        "etf", "tracker", "gestion collective", "gestion institutionnelle", "gestion privée",
        "allocation d'actifs", "performance des fonds", "frais de gestion", "frais de performance",
        "assurance vie", "gestion de patrimoine", "gestion active", "gestion passive", "benchmark",
        "rendement des fonds", "alpha", "tracking error", "var", "sharpe", "rendement"
    ],
    "private equity": [
        'capital investissement', 'private equity', 'capital-risque', 'venture capital',
        'capital développement', 'capital transmission', 'buy-out', 'mbo', 'mbi',
        'fonds de capital investissement', 'société de capital risque', 'scr', 'fpci', 'fcpr',
        'levée de fonds', 'amorçage', 'préamorçage', 'seed capital', 'early stage', 'growth capital', 'expansion capital',
        'capital retournement', 'turnaround', 'distressed', 'mezzanine',
        'limited partner', 'general partner', 'carried interest', 'management fee',
        'exit strategy', 'ipo exit', 'stratégie de sortie', 'secondary buyout',
        'due diligence', 'deal sourcing', 'portfolio company', 'tour de table',  "capital investissement", "private equity", "capital-risque", "venture capital",
        "levée de fonds", "tour de table", "buyout", "buy-out", "mbo", "mbi", "secondary",
        "capital développement", "capital transmission", "investisseur en capital", "firmes de private equity",
        "limited partner", "general partner", "carried interest", "management fee", "exit",
        "harvest", "portfolio company", "due diligence", "deal sourcing"
    ],
    "brokerage&custody": [
        'courtage', 'brokerage', 'négociation de titres', 'exécution d\'ordres',
        'market making', 'teneur de marché', 'spread bid-ask', 'commission de courtage',
        'plateforme de trading', 'trading algorithmique', 'high frequency trading',
        'conservation de titres', 'custody', 'dépositaire', 'garde de titres',
        'clearing', 'règlement-livraison', 'post-marché',
        'maroclear', 'dépositaire central', 'compte titre', 'compte espèces',
        'services aux investisseurs', 'prime brokerage', 'securities lending',
        'repo', 'pension livrée', 'collatéral', 'margining', 'corporate actions',
        'proxy voting', 'income collection', "courtage", "courtier", "brokerage", "négociation d'ordres", "exécution d'ordres",
        "market making", "teneur de marché", "marcheur", "spread", "commission de courtage",
        "plateforme de trading", "trading algorithmique", "high frequency trading",
        "conservation de titres", "custody", "dépositaire", "garde de titres",
        "clearing", "règlement-livraison", "post-marché", "maroclear", "compte titre",
        "prime brokerage", "securities lending", "repo", "pension livrée", "collatéral", "margin"
    ],
    "real estate": [
        'foncier', 'promotion immobilière', 'développement immobilier',
        'investissement immobilier', 'investissement locatif', 'fonds immobilier', 'foncière',
        'reit', 'sci', 'société civile immobilière', 'scpi', 'siic', 'opci', 'défiscalisation immobilière',
        'résidentiel', 'bureaux', 'retail',
        'centres commerciaux', 'entrepôts', 'hôtellerie', 'résidences services',
        'crédit immobilier', 'hypothèque', 'refinancement immobilier',
        'titrisation immobilière', 'mortgage', 'prêt immobilier',
        'prix immobilier', 'marché immobilier', 'transaction immobilière',
        'mètre carré', 'm²', 'surface', 'rendement locatif', 'capitalisation immobilière',
        'loyer', 'loyers', 'loi Pinel', 'loi Malraux', "immobilier", "foncier", "promotion immobilière", "développement immobilier",
        "investissement immobilier", "investissement locatif", "fonds immobilier",
        "foncière", "reit", "sci", "scpi", "siic", "opci", "titrisation immobilière",
        "résidentiel", "bureaux", "retail", "centres commerciaux", "entrepôts", "prix immobilier",
        "mètre carré", "m²", "loyer", "loyers", "rendement locatif", "hypothèque", "crédit immobilier"
    ]
}

FINANCIAL_TERMS = [
    'bourse', 'action', 'obligation', 'titre', 'indice', 'masi', 'madex', 'casablanca',
    'dividende', 'capitalisation', 'investissement', 'placement', 'portefeuille',
    'fonds', 'sicav', 'opcvm', 'etf', 'rendement', 'performance',
    'banque', 'crédit', 'prêt', 'taux', 'intérêt', 'financement', 'bancaire',
    'corporate', 'fusion', 'acquisition', 'augmentation de capital', 'émission',
    'gestion d\'actifs', 'asset management', 'gestion de portefeuille', 'allocation',
    'diversification', 'benchmark', 'volatilité', 'sharpe',
    'capital investissement', 'capital risque', 'venture capital',
    'buy-out', 'exit', 'portfolio', 'due diligence',
    'courtage', 'brokerage', 'négociation', 'trading', 'custody', 'conservation',
    'clearing', 'compensation', 'maroclear', 'teneur de marché',
    'immobilier', 'foncier', 'promotion immobilière', 'investissement immobilier',
    'crédit immobilier', 'prix immobilier', 'transaction immobilière',
    'résultat', 'bénéfice', 'profit', 'chiffre d\'affaires', 'ebitda', 'cash-flow',
    'endettement', 'ratio', 'marge', 'croissance', 'valorisation',
    'bam', 'bank al-maghrib', 'ammc', 'attijariwafa', 'bmce', 'cih', 'cdg',
    'banque populaire', 'crédit agricole', 'bank of africa',
    'dirham', 'mdh', 'mmdh', 'millions', 'devise', 'change'
]

NON_FINANCIAL_TERMS = [
    # Sports
    'football', 'match', 'coupe du monde', 'mondial', 'championnat', 'tournoi',
    'stade', 'arbitre', 'basket', 'tennis', 'handball', 'volleyball', 'athlétisme',
    'natation', 'cyclisme', 'marathon',
    # Politics
    'élection', 'vote', 'parlement', 'scrutin', 'ministre', 'gouvernement', 'coalition', 'majorité',
    'sénat', 'assemblée',
    # Culture/Media
    'cinéma', 'film', 'acteur', 'musique', 'concert', 'festival', 'série', 'spectacle',
    'télévision', 'célébrité', 'oscar', 'cannes',
    # Health
    'santé', 'hôpital', 'médecin', 'patient', 'maladie', 'traitement', 'vaccin',
    'épidémie', 'pandémie', 'covid', 'urgences médicales',
    # Crime/Law
    'crime', 'police', 'tribunal', 'prison', 'drogue', 'trafic', 'saisie', 'trafic',
    # Education
    'école', 'université', 'étudiant', 'enseignement', 'professeur', 'académie', 'diplôme',
    'examen', 'concours',
    # Weather/Nature
    'météo', 'tempête', 'neige', 'inondation', 'sécheresse', 'incendie', 'séisme',
    'tremblement', 'feu de forêt', 'orage',
    'accident', 'route', 'autoroute', 'circulation', 'embouteillage',
    'permis', 'radar', 'amende',
    # Technology (broadly non-financial)
    'internet', 'application', 'smartphone', 'ordinateur', 'logiciel', 'innovation',
    # Social Media / Influenceurs
    'influenceur', 'influenceuse', 'blogueur', 'youtube', 'tiktok'
]

FINANCIAL_CONTEXT_INDICATORS = [
    r'\b\d+(\.\d+)?\s*(milliards?|millions?|mds|mdh|mmdh|k€|k\$)\b',
    r'\b\d+(\.\d+)?\s*(dh|mad|dirhams?|euros?|dollars?|€|\$|usd|eur)\b',
    r'\b(croissance|hausse|baisse|progression|chute|recul|rendement|taux)\s+(de\s+)?\d+(\.\d+)?\s*(%|pour\s?cent)\b',
    r'\b(ratio|marge|ebitda|roe|roa|price to earnings)\b',
    r'\b(capitalisation|valorisation|endettement|liquidité|solvabilité)\b',
    r'\b(cours|cotation|indice|bourse)\b.*\d+(\.\d+)?'
]

def normalize_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def has_financial_context(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in FINANCIAL_CONTEXT_INDICATORS)

def classify_branches(title: str, text: str) -> list:
    """Classify an article into financial branches by matching whole-word keywords."""
    combined = (title + " " + text).lower()
    branches = []
    for branch, keywords in BRANCH_KEYWORDS.items():
        # Count whole-word keyword matches
        matches = sum(bool(re.search(r'\b' + re.escape(kw.lower()) + r'\b', combined)) for kw in keywords)
        if matches >= 1:
            if branch == "real estate":
                # Require specific real-estate terms or numeric context
                re_terms = [
                    'investissement immobilier', 'crédit immobilier', 'fonds immobilier',
                    'prix immobilier', 'marché immobilier', 'rendement locatif',
                    'transaction immobilière', 'valorisation immobilière', 'reit',
                    'promotion immobilière', 'défiscalisation'
                ]
                if any(re.search(r'\b' + term + r'\b', combined) for term in re_terms) or has_financial_context(combined):
                    branches.append(branch)
            else:
                branches.append(branch)
    return branches

def is_relevant_financial_article(title: str, text: str, url: str, source_config: dict) -> bool:
    """Determine if an article is finance-related and fits one of the branches."""
    if not title or not text:
        return False
    # Skip if URL in a known non-finance section
    if any(sec in url.lower() for sec in source_config.get("non_finance_sections", [])):
        return False
    combined = (title + " " + text).lower()
    # Exclusion patterns (sports, politics, etc.)
    exclusion_patterns = [
        r'(match|rencontre|finale|demi-finale|quart de finale)\s+(de|du)\s+(football|basket|tennis)',
        r'(coupe du monde|mondial)\b',
        r'(élection|campagne électorale|scrutin|vote|parlement)',
        r'(festival|concert|spectacle|cinéma|film|série)',
        r'(arrestation|saisie|trafic|drogue|police|tribunal)\b',
        r'(météo|tempête|inondation|incendie|accident)\b'
    ]
    for patt in exclusion_patterns:
        if re.search(patt, combined, re.IGNORECASE):
            return False
    # Count finance vs non-finance terms
    financial_count = sum(term in combined for term in FINANCIAL_TERMS)
    non_financial_count = sum(term in combined for term in NON_FINANCIAL_TERMS)
    branches = classify_branches(title, text)
    has_context = has_financial_context(combined)
    relevant = False
    if branches and (financial_count >= 2 or has_context):
        relevant = True
    # If too many non-finance terms, reject
    if relevant and non_financial_count > financial_count:
        relevant = False
    return relevant

def is_likely_article_url(url: str, source_config: dict) -> bool:
    """Check if a URL looks like an article link (not navigation or external)."""
    if not url or url in ("#", "/"):
        return False
    base_domain = urlparse(source_config["url"]).netloc
    parsed_url = urlparse(url)
    # Enforce same domain
    if parsed_url.netloc and parsed_url.netloc != base_domain:
        return False
    # Skip known patterns
    skip_patterns = [
        '/tag/', '/author/', '/search/', '/contact/', '/about/', '/archives/',
        '/sitemap/', '/privacy/', '/terms/', 'javascript:', 'mailto:',
        '/rss/', '/feed/', '/login', '/register', '/mon-compte', '/profile',
        '/newsletter', '/publicite', '/cgu', '/mentions-legales', '/confidentialite',
        '/cotations', '/indices', '/activites-royales', '/nation', '/monde',
        '/societe', '/culture', '/sports', '/sport', '/videos', '/regions', '/politique'
    ]
    for p in skip_patterns:
        if p in url.lower():
            return False
    # Article indicators
    article_indicators = [
        '/article/', '/news/', '/actualite/', '/economie/', '/finance/', '/bourse/',
        '/entreprise/', '/marches/', '/banque/', '/industrie/', '/immobilier/',
        '/energie/', '/tourisme/', '/tech/', '/startup/', '/pme/', '/investissement/',
        '.html', '.php?id=', '/20'
    ]
    article_indicators += source_config.get("finance_sections", [])
    if not any(ind in url.lower() for ind in article_indicators):
        return False
    # Exclude pagination or root
    if 'page=' in url or '/page/' in url:
        return False
    if not parsed_url.path or parsed_url.path == '/':
        return False
    return True

def fetch_with_retry(url: str, headers: dict, max_retries: int = 3, timeout: int = 15):
    """Fetch a URL with retries using the configured session."""
    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait = timeout * (1.5 + random.random()*0.5)
                logger.warning(f"Retry {attempt+1} for {url} after {wait:.1f}s due to {e}")
                time.sleep(wait)
                timeout = min(timeout*1.5, 60)
            else:
                logger.error(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                return None
    return None

def generate_content_hash(title: str, text: str) -> str:
    content = f"{title}::{text[:1000]}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()

class FinancialNewsArticle:
    def __init__(self, title, text, url, date, source):
        self.title = title
        self.text = normalize_text(text)
        self.url = url
        self.date = date
        self.source = source
        self.content_hash = generate_content_hash(title, self.text)
        self.branches = classify_branches(title, text)

    def to_dict(self):
        return {
            "title": self.title,
            "text": self.text,
            "url": self.url,
            "date": self.date,
            "source": self.source,
            "content_hash": self.content_hash,
            "scraped_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "branch": "; ".join(self.branches)
        }

    def is_valid_article(self):
        if not self.title or len(self.title) < 10:
            return False
        if not self.text or len(self.text) < 100:
            return False
        if len(self.title) < 15:
            return False
        skip_titles = ["accueil", "actualités", "économie", "finance", "bourse",
                       "cotations", "indices", "abonnement", "login", "découvrez",
                       "vidéo", "société", "culture", "sport", "contact",
                       "à propos", "mentions légales", "cgu", "confidentialité"]
        for skip in skip_titles:
            if skip.lower() in self.title.lower():
                return False
        if self.text.count('.') < 3:
            return False
        nav_indicators = ["accueil", "actualités", "economie", "finance", "contact",
                          "mentions légales", "cgu", "confidentialité"]
        if sum(1 for term in nav_indicators if term in self.text.lower()) > 3:
            return False
        if not self.branches:
            return False
        return True

class FinancialNewsScraper:
    def __init__(self, source_config):
        self.name = source_config["name"]
        self.base_url = source_config["url"]
        self.language = source_config["language"]
        self.source_config = source_config
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept-Language": "fr-FR,fr;q=0.9"
        }
        self.config = Config()
        self.config.request_timeout = 20
        self.config.browser_user_agent = self.headers["User-Agent"]
        self.config.number_threads = 1
        self.start_time = time.time()
        self.articles_processed = 0
        self.articles_accepted = 0
        self.articles_rejected = 0
        self.branch_stats = {branch: 0 for branch in BRANCH_KEYWORDS.keys()}

    def get_article_urls(self):
        try:
            urls = set()
            response = fetch_with_retry(self.base_url, self.headers)
            if not response:
                return []
            soup = BeautifulSoup(response.content, "html.parser")
            anchors = soup.find_all("a", href=True)
            for anchor in anchors:
                href = anchor.get("href")
                if not href:
                    continue
                url = urljoin(self.base_url, href)
                if is_likely_article_url(url, self.source_config):
                    urls.add(url)
            urls = list(urls)
            for url in urls:
                logger.debug(f"Found URL: {url}")
            return urls
        except Exception as e:
            logger.error(f"Error getting URLs from {self.name}: {e}")
            return []

    def parse_article(self, url: str):
        try:
            self.articles_processed += 1
            article = Article(url, language=self.language, config=self.config)
            article.download()
            article.parse()
            date = article.publish_date or datetime.datetime.now()
            news_article = FinancialNewsArticle(
                title=article.title,
                text=article.text,
                url=url,
                date=date,
                source=self.name
            )
            if not news_article.is_valid_article():
                self.articles_rejected += 1
                logger.debug(f"Invalid article (structure): {news_article.title[:30]}")
                return None
            if not is_relevant_financial_article(news_article.title, news_article.text, url, self.source_config):
                self.articles_rejected += 1
                logger.debug(f"Not finance-relevant: {news_article.title[:30]}")
                return None
            for branch in news_article.branches:
                self.branch_stats[branch] += 1
            self.articles_accepted += 1
            return news_article
        except Exception as e:
            self.articles_rejected += 1
            logger.error(f"Error parsing {url}: {e}")
            return None

    def report_performance(self):
        elapsed = time.time() - self.start_time
        return {
            "source": self.name,
            "processed": self.articles_processed,
            "accepted": self.articles_accepted,
            "rejected": self.articles_rejected,
            "acceptance_rate": f"{(self.articles_accepted / max(1, self.articles_processed)) * 100:.1f}%",
            "elapsed_time": f"{elapsed:.2f}s",
            "branch_stats": self.branch_stats.copy()
        }

def load_existing_articles():
    path = LABELED_FILE
    if path.exists():
        try:
            df = pd.read_csv(path)
            urls = set(df['url']) if 'url' in df.columns else set()
            hashes = set(df['content_hash']) if 'content_hash' in df.columns else set()
            return urls, hashes, df
        except Exception as e:
            logger.error(f"Error loading existing CSV ({path}): {e}")
    return set(), set(), pd.DataFrame()

def collect_perfect_financial_news():
    existing_urls, existing_hashes, existing_df = load_existing_articles()
    logger.info(f"Existing articles: {len(existing_urls)} URLs, {len(existing_hashes)} hashes")
    all_new = []
    performance = []
    total_branch_stats = {branch: 0 for branch in BRANCH_KEYWORDS.keys()}

    for config in FINANCIAL_NEWS_SOURCES:
        logger.info(f"Scraping source: {config['name']}")
        scraper = FinancialNewsScraper(config)
        urls = scraper.get_article_urls()
        new_urls = [u for u in urls if u not in existing_urls]
        logger.info(f"Found {len(urls)} URLs, {len(new_urls)} new")
        for url in new_urls:
            time.sleep(random.uniform(1, 2))
            article = scraper.parse_article(url)
            if article:
                if article.content_hash not in existing_hashes:
                    all_new.append(article.to_dict())
                    existing_hashes.add(article.content_hash)
                    for b in article.branches:
                        total_branch_stats[b] += 1
                else:
                    logger.info(f"Duplicate content (skipped): {article.title[:50]}")
        performance.append(scraper.report_performance())

    # Combine and save results
    if all_new:
        new_df = pd.DataFrame(all_new)
        file_path = LABELED_FILE
        logger.info(f"Preparing to write {file_path.resolve()}")
        if not existing_df.empty:
            backup_path = PROCESSED_DIR / f"labeled_news_multi_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            try:
                existing_df.to_csv(backup_path, index=False)
                logger.info(f"Backed up existing data to {backup_path.name}")
            except Exception as e:
                logger.error(f"Failed to write backup to {backup_path}: {e}")
            combined = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
            combined.drop_duplicates(subset=['url', 'content_hash'], inplace=True)
            try:
                combined.to_csv(file_path, index=False)
                added = len(combined) - len(existing_df)
                logger.info(f"Appended {added} new articles to {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to write combined file {file_path}: {e}")
        else:
            try:
                new_df.to_csv(file_path, index=False)
                logger.info(f"Created {file_path.name} with {len(new_df)} articles")
            except Exception as e:
                logger.error(f"Failed to write file {file_path}: {e}")
    else:
        logger.info("No new financial articles to add.")

    # Summary logging
    total_processed = sum(p["processed"] for p in performance)
    total_accepted = sum(p["accepted"] for p in performance)
    logger.info(f"Scraping complete. Total processed: {total_processed}, accepted: {total_accepted}")
    for branch, count in total_branch_stats.items():
        if count:
            logger.info(f"  Branch '{branch}': {count} articles")
    for p in performance:
        logger.info(f"{p['source']}: processed={p['processed']} accepted={p['accepted']} ({p['acceptance_rate']})")

    return len(all_new)

if __name__ == "__main__":
    logger.info(f"Starting scraper at {datetime.datetime.now().isoformat()}")
    try:
        logger.info(f"Target output file: {LABELED_FILE.resolve()}")
        new_count = collect_perfect_financial_news()
        logger.info(f"Scraper finished. Added {new_count} new articles.")
    except Exception as e:
        logger.error(f"Scraper error: {e}", exc_info=True)
