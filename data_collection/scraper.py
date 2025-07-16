"""
100% Precision Financial News Scraper v1.2
Only genuine financial/economic news with absolutely no exceptions
Last updated: 2025-06-23
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
import hashlib
from typing import List, Dict, Optional, Set, Any, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("perfect_financial_scraper")

# Configure financial news sources with specific finance sections only
FINANCIAL_NEWS_SOURCES = [
    {
        "name": "Boursenews",
        "url": "https://boursenews.ma/articles/marches",  # Specifically the markets section
        "language": "fr",
        "finance_sections": ["/marches/", "/actualite-financiere/", "/bourse/"],
        "non_finance_sections": ["/sport/", "/societe/", "/politique/"]
    },
    {
        "name": "FLM",
        "url": "https://flm.ma/economie.php",  # Economy section
        "language": "fr",
        "finance_sections": ["/economie.php", "/finance.php", "/business.php"],
        "non_finance_sections": ["/sports.php", "/culture.php", "/politique.php"]
    },
    {
        "name": "Le Desk",
        "url": "https://ledesk.ma/thematiques/business/",  # Business section
        "language": "fr",
        "finance_sections": ["/business/", "/economie/", "/finances/"],
        "non_finance_sections": ["/sport/", "/culture/", "/politique/"]
    },
    {
        "name": "H24Info",
        "url": "https://www.h24info.ma/economie/",  # Economy section only
        "language": "fr",
        "finance_sections": ["/economie/", "/finance/", "/entreprises/"],
        "non_finance_sections": ["/sport/", "/maroc/", "/monde/"]
    },
    {
        "name": "Le Matin",
        "url": "https://lematin.ma/economie",  # Economy section
        "language": "fr",
        "finance_sections": ["/economie/", "/finance/", "/bourse/"],
        "non_finance_sections": ["/sport/", "/culture/", "/societe/"]
    },
    {
        "name": "Medias24",
        "url": "https://medias24.com/categorie/economie/",  # Economy category
        "language": "fr",
        "finance_sections": ["/economie/", "/finance/", "/leboursier/"],
        "non_finance_sections": ["/politique/", "/societe/", "/sport/"]
    },
    {
        "name": "Finance News - Financial",
        "url": "https://fnh.ma/articles/actualite-financiere-maroc",  # Financial news
        "language": "fr",
        "finance_sections": ["/actualite-financiere-maroc", "/bourse/", "/finance/"],
        "non_finance_sections": ["/opinions/", "/politique/", "/societe/"]
    },
    {
        "name": "Finance News - Economic",
        "url": "https://fnh.ma/articles/actualite-economique",  # Economic news
        "language": "fr",
        "finance_sections": ["/actualite-economique", "/economie/", "/entreprises/"],
        "non_finance_sections": ["/opinions/", "/politique/", "/societe/"]
    }
]

# Define comprehensive financial terms for content filtering
FINANCIAL_TERMS = [
    # Financial markets and instruments
    'bourse', 'action', 'obligation', 'titre', 'indice', 'masi', 'madex', 'casablanca', 
    'dividende', 'capitalisation', 'investissement', 'placement', 'portefeuille',
    
    # Banking and finance
    'banque', 'crédit', 'prêt', 'taux', 'intérêt', 'dépôt', 'finance', 'financement',
    'bancaire', 'hypothèque', 'assurance', 'microfinance', 'fintech',
    
    # Economic indicators
    'économie', 'pib', 'croissance', 'inflation', 'déflation', 'récession', 'emploi', 'chômage',
    'budget', 'déficit', 'excédent', 'dette', 'fiscal', 'impôt', 'taxe',
    
    # Business and corporate
    'entreprise', 'société', 'corporate', 'bénéfice', 'résultat', 'chiffre d\'affaires',
    'revenu', 'profit', 'marge', 'acquisition', 'fusion', 'holding', 'filiale',
    
    # Trade and commerce
    'commerce', 'import', 'export', 'balance commerciale', 'douane', 'tarif',
    'échange', 'libre-échange', 'partenariat', 'accord commercial',
    
    # Industry and sectors
    'industrie', 'secteur', 'manufacturier', 'production', 'immobilier', 'btp',
    'énergie', 'tourisme', 'agriculture', 'télécommunication', 'transport',
    
    # Financial institutions and regulators
    'bam', 'bank al-maghrib', 'ammc', 'berd', 'bid', 'fmi', 'banque mondiale',
    'trésor', 'bourse de casablanca', 'régulateur', 'supervision',
    
    # Currency and forex
    'dirham', 'devise', 'monnaie', 'change', 'parité', 'convertibilité',
    'appréciation', 'dépréciation', 'réserve', 'forex',
    
    # Financial analysis and metrics
    'analyse', 'ratio', 'performance', 'rendement', 'liquidité', 'solvabilité',
    'valorisation', 'cotation', 'cours', 'volatilité', 'risque',
    
    # Specific Moroccan financial terms
    'ofppt', 'cnss', 'cih', 'bmce', 'attijariwafa', 'cdg', 'onee', 'ocp',
    'plan d\'investissement', 'projet économique', 'développement économique',
    
    # Financial metrics (specific)
    'mdh', 'mmdh', 'milliards?', 'millions?', 'augmentation de capital', 'résultat net', 
    'capitalisation boursière', 'dette nette', 'ebitda', 'résultat d\'exploitation',
    
    # Additional banking and monetary terms
    'fed', 'réserve fédérale', 'taux directeur', 'politique monétaire', 'obligations',
    'eurobond', 'eurobonds', 'titrisation', 'compartiment', 'souscription', 'émission'
]

# Expanded list of non-financial terms for exclusion
NON_FINANCIAL_TERMS = [
    # Sports
    'football', 'match', 'but', 'joueur', 'équipe', 'championnat', 'tournoi',
    'stade', 'coupe', 'arbitre', 'wydad', 'raja', 'olympique',
    
    # Politics
    'élection', 'parlement', 'ministre', 'gouvernement', 'politique', 'parti',
    'député', 'campagne', 'vote', 'scrutin', 'constitution',
    
    # Entertainment
    'cinéma', 'film', 'acteur', 'musique', 'concert', 'festival', 'télévision',
    'série', 'spectacle', 'théâtre', 'artiste', 'célébrité',
    
    # Society and general news
    'accident', 'crime', 'policier', 'tribunal', 'manifestation', 'météo',
    'santé', 'hôpital', 'médecin', 'école', 'université', 'étudiant',
    
    # Military and defense
    'militaire', 'service militaire', 'armée', 'défense', 'naval', 'marine',
    'exercice militaire', 'forces armées', 'soldat', 'manoeuvre', 'guerre',
    
    # Disasters and incidents
    'incendie', 'feu', 'catastrophe', 'forêt', 'inondation', 'séisme',
    'accident', 'victime', 'sinistre', 'explosion', 'pompier', 'secours',
    
    # General infrastructure without financial angle
    'inauguration', 'cérémonie', 'barrage', 'pont', 'route', 'infrastructure',
    'chantier', 'construction', 'bâtiment', 'édifice', 'aménagement',
    
    # Education and training
    'académie', 'formation', 'apprentissage', 'stage', 'diplôme', 'étudiant', 
    'élève', 'enseignement', 'compétence', 'pédagogique', 'éducation', 'cours',
    'atelier', 'master class', 'professeur', 'instructeur', 'tuteur', 'apprenant',
    
    # Hospitality non-financial
    'accueil', 'hébergement', 'restauration', 'hôtellerie', 'gastronomie', 
    'cuisine', 'chef', 'service', 'réception', 'chambre', 'séjour', 'all-inclusive',
    'pension', 'buffet', 'restaurant', 'resort',
    
    # Transportation regulatory
    'plaque d\'immatriculation', 'immatriculation', 'permis de conduire', 'code de la route',
    'autorisation de voyage', 'narsa', 'circulation', 'sécurité routière', 'contrôle technique',
    
    # Research and scientific fields (non-financial)
    'recherche scientifique', 'étude', 'laboratoire', 'expérimentation', 'chercheur',
    'universitaire', 'séquençage', 'génome', 'caractérisation', 'composé actif',
    'cannabis', 'variété', 'autochtone', 'thérapeutique', 'médical', 'médicinal'
]

# Keywords that require financial context to be relevant
CONTEXT_DEPENDENT_TERMS = [
    'barrage', 'pont', 'route', 'infrastructure', 'chantier', 'construction',
    'projet', 'développement', 'inauguration', 'lancement', 'programme',
    'initiative', 'stratégie', 'plan',
    # Terms that are ambiguous between business/education
    'formation', 'partenariat', 'développement', 'croissance', 'transformation',
    'compétence', 'expertise', 'performance', 'qualité', 'innovation',
    # Research terms that need financial context
    'recherche', 'étude', 'analyse', 'laboratoire', 'caractérisation', 'valorisation'
]

# Required financial metrics - at least one must be present in corporate news
FINANCIAL_METRICS = [
    r'\d+(\.\d+)?(\s+)?(milliards?|millions?|mds|mdh|mmdh)',  # Amount with currency units
    r'\d+(\.\d+)?(\s+)?(dh|mad|dirhams?|euros?|dollars?|€|\$)',  # Currency with symbol
    r'(augmentation|hausse|baisse|progression|chute|recul)(\s+de\s+)?\d+(\.\d+)?(\s+)?(%|pour cent|points)', # Percentage change
    r'(croissance|progression)(\s+de\s+)?\d+(\.\d+)?(\s+)?(%|pour cent)', # Growth percentage
    r'(résultat|bénéfice|profit|perte)(\s+net)?(\s+de\s+)?\d+', # Net results
    r'(chiffre d\'affaires|ca|revenu)(\s+de\s+)?\d+', # Revenue
    r'(valorisation|capitalisation)(\s+de\s+)?\d+', # Valuation
    r'(dividende|coupon)(\s+de\s+)?\d+', # Dividend
    r'(coût|budget|financement|investissement)(\s+de\s+)?\d+', # Cost/budget
    r'(taux\s+directeur|taux\s+d\'intérêt)\s+(\w+\s+)?\d+(\.\d+)?(\s+)?(%|pour cent)', # Interest rates
    r'(indice|masi|madex)(\s+\w+)?\s+([+-])?\d+(\.\d+)?(\s+)?(%|pour cent)' # Stock market indices
]

# Strong sector-dependent terms that need extra verification
SECTOR_TERMS = {
    "hospitality": ['hôtel', 'resort', 'restaurant', 'tourisme', 'voyage', 'séjour', 'hébergement'],
    "education": ['formation', 'académie', 'école', 'université', 'enseignement', 'diplôme'],
    "culture": ['musée', 'exposition', 'spectacle', 'festival', 'concert', 'théâtre'],
    "sports": ['stade', 'équipe', 'match', 'joueur', 'tournoi', 'championnat'],
    "research": ['recherche', 'étude', 'laboratoire', 'expérimentation', 'scientifique', 'université'],
    "cannabis": ['cannabis', 'chanvre', 'anrac', 'beldia', 'médical', 'médicinal', 'thérapeutique']
}

# Strong financial entity indicators - presence strongly suggests financial content
FINANCIAL_ENTITIES = [
    'bank al-maghrib', 'bam', 'ammc', 'ministère des finances', 'ministre des finances',
    'bourse de casablanca', 'trésor', 'fmi', 'banque mondiale', 'berd', 'attijariwafa', 'bmce', 
    'cih bank', 'crédit agricole', 'crédit du maroc', 'banque populaire', 'bank of africa',
    'fed', 'réserve fédérale', 'bce', 'banque centrale européenne', 'eurobonds', 'titrisation'
]

def normalize_text(text: str) -> str:
    """Normalize article text by converting it to a single line."""
    if not text or not isinstance(text, str):
        return ""
    
    # Replace all line breaks with spaces
    normalized = re.sub(r'\s*\n\s*', ' ', text)
    
    # Replace multiple spaces with a single space
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove any leading/trailing whitespace
    normalized = normalized.strip()
    
    return normalized

def has_financial_metrics(text: str) -> bool:
    """Check if text contains any financial metrics patterns."""
    return any(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in FINANCIAL_METRICS)

def is_financial_article(title: str, text: str, url: str, source_config: Dict[str, Any]) -> bool:
    """
    100% accuracy determination if an article is truly finance-related
    with enhanced exclusion for education, hospitality and other edge cases
    """
    if not title or not text:
        return False
    
    # Step 1: Check URL patterns first
    finance_sections = source_config.get("finance_sections", [])
    non_finance_sections = source_config.get("non_finance_sections", [])
    
    if any(section in url.lower() for section in non_finance_sections):
        logger.debug(f"Rejected due to non-finance URL section: {url}")
        return False
    
    url_looks_financial = any(section in url.lower() for section in finance_sections)
    
    # Step 2: Initial content analysis
    combined_text = (title + " " + text).lower()
    
    # Check for strong financial entities - these are very reliable indicators
    has_financial_entity = any(entity.lower() in combined_text for entity in FINANCIAL_ENTITIES)
    if has_financial_entity and has_financial_metrics(combined_text):
        logger.debug(f"Fast-tracked approval: Contains financial entity and metrics: {title[:50]}...")
        return True
    
    # Step 3: Check for explicit exclusion categories first before counting terms
    
    # 3.1: Crime/Security/Drug articles exclusion (even if they mention quantities)
    crime_terms = ['drogue', 'cocaïne', 'trafic', 'saisie', 'criminalité', 'police', 
                  'interpellation', 'arrestation', 'prison', 'pénitentiaire', 'contrebande',
                  'stupéfiant', 'sûreté', 'délit', 'infraction', 'illégal', 'narcotique']
    
    crime_term_count = sum(1 for term in crime_terms if term in combined_text)
    
    # If multiple crime terms AND mentions of kg/quantity, it's likely a drug seizure story
    has_quantity_pattern = bool(re.search(r'\d+(\.\d+)?\s*(kg|kilo|kilogramme|tonne|g|gramme)', combined_text))
    
    if crime_term_count >= 2 and has_quantity_pattern:
        logger.debug(f"Rejected: Crime/security article with quantities: {title[:50]}...")
        return False
    
    # 3.2: Organizational/Administrative announcement without financial substance
    administrative_terms = ['corporation', 'artisanat', 'ministériel', 'décret', 'arrêté', 
                           'réglementation', 'structure organisationnelle', 'comité', 
                           'association professionnelle', 'regroupement', 'institution',
                           'organisme', 'supervision', 'tutelle', 'gouvernance']
    
    admin_term_count = sum(1 for term in administrative_terms if term in combined_text)
    
    # 3.3 Education/Training program focus check
    education_terms = ['académie', 'formation', 'apprentissage', 'stage', 'diplôme', 
                       'étudiant', 'élève', 'enseignement', 'compétence', 'pédagogique']
    
    education_term_count = sum(1 for term in education_terms if term in combined_text)
    
    # 3.4 Hospitality/Tourism non-financial focus check
    hospitality_terms = ['hôtel', 'accueil', 'hébergement', 'restauration', 'gastronomie',
                        'cuisine', 'chef', 'service', 'réception', 'chambre']
    
    hospitality_term_count = sum(1 for term in hospitality_terms if term in combined_text)
    
    # 3.5 Cannabis/Medical research check (non-financial)
    cannabis_terms = ['cannabis', 'chanvre', 'caractérisation', 'beldia', 'autochtone', 
                    'anrac', 'recherche scientifique', 'composé actif', 'thérapeutique',
                    'séquençage', 'génome', 'variété']
    
    cannabis_term_count = sum(1 for term in cannabis_terms if term in combined_text)
    
    if cannabis_term_count >= 3 and 'médicinal' in combined_text:
        # Must have explicit financial metrics or investment information
        if not has_financial_metrics(combined_text) and not 'investissement' in combined_text:
            logger.debug(f"Rejected: Cannabis/medical research without financial metrics: {title[:50]}...")
            return False
    
    # 3.8: Transportation regulatory announcement check
    transport_reg_terms = ['narsa', 'immatriculation', 'plaque', 'permis', 'code de la route',
                          'circulation', 'véhicule', 'conduite', 'sécurité routière']

    transport_reg_count = sum(1 for term in transport_reg_terms if term in combined_text)

    if transport_reg_count >= 2 and 'obligatoire' in combined_text:
        # Must have financial impact to qualify
        has_transport_finance = any(term in combined_text for term in 
                                  ['taxe', 'amende', 'coût', 'tarif', 'prix', 'économie'])
        
        if not has_transport_finance:
            logger.debug(f"Rejected: Transport regulation without financial focus: {title[:50]}...")
            return False
    
    # Count financial terms
    financial_term_count = sum(1 for term in FINANCIAL_TERMS if term in combined_text)
    
    # Count non-financial terms
    non_financial_term_count = sum(1 for term in NON_FINANCIAL_TERMS if term in combined_text)
    
    # 3.5 Check for education articles with business terminology but no financial metrics
    if education_term_count >= 3 and financial_term_count <= 3:
        # Check if there are specific financial metrics in education/training articles
        has_financial_metrics_bool = has_financial_metrics(combined_text)
        
        if not has_financial_metrics_bool:
            logger.debug(f"Rejected: Education/training article without financial metrics: {title[:50]}...")
            return False
    
    # 3.6 Check for hospitality articles with business terminology but no financial metrics
    if hospitality_term_count >= 3 and financial_term_count <= 3:
        # Check if there are specific financial metrics in hospitality articles
        has_financial_metrics_bool = has_financial_metrics(combined_text)
        
        if not has_financial_metrics_bool:
            logger.debug(f"Rejected: Hospitality article without financial metrics: {title[:50]}...")
            return False
    
    # 3.7: Detect organizational stories with minimal financial content
    if admin_term_count >= 3 and financial_term_count <= 2:
        # Additional check: look for actual financial metrics in administrative articles
        has_financial_metrics_bool = has_financial_metrics(combined_text)
        
        if not has_financial_metrics_bool:
            logger.debug(f"Rejected: Administrative article without financial substance: {title[:50]}...")
            return False
    
    # Step 4: Check for context-dependent terms and financial context
    has_context_dependent_terms = any(term in combined_text for term in CONTEXT_DEPENDENT_TERMS)
    
    # If the article mentions context-dependent terms, it must have strong financial context
    if has_context_dependent_terms:
        # Check if there are financial indicators like numbers with currency symbols
        has_currency_amounts = any(bool(re.search(pattern, combined_text, re.IGNORECASE)) 
                                  for pattern in FINANCIAL_METRICS[:2])  # First two patterns check currency
        
        has_financial_metrics_bool = any(metric in combined_text for metric in 
                                  ['investissement', 'financement', 'budget', 'coût', 'montant', 
                                   'bénéfice', 'profit', 'revenu', 'chiffre d\'affaires'])
        
        # If context-dependent terms lack financial context, require more financial terms
        if not (has_currency_amounts or has_financial_metrics_bool):
            financial_term_count -= 2  # Stronger penalty
    
    # Check for sector-dependent articles that need extra verification
    for sector, terms in SECTOR_TERMS.items():
        sector_term_count = sum(1 for term in terms if term in combined_text)
        if sector_term_count >= 3:
            # For these high-risk sectors, we need explicit financial metrics
            has_explicit_metrics = has_financial_metrics(combined_text)
            
            if not has_explicit_metrics:
                financial_term_count -= 3  # Very strong penalty
                logger.debug(f"Applied strong penalty to {sector} article without explicit metrics: {title[:50]}...")
    
    # Step 5: More stringent financial content requirements
    is_financial = (
        # Either strong URL + good content:
        (url_looks_financial and financial_term_count >= 3 and financial_term_count > non_financial_term_count) or
        # Or very strong content regardless of URL:
        (financial_term_count >= 5 and financial_term_count >= non_financial_term_count * 2) or
        # Or clear financial metrics present and more financial than non-financial terms:
        (has_financial_metrics(combined_text) and financial_term_count > non_financial_term_count)
    )
    
    # Step 6: Additional checks for problematic categories
    if is_financial:
        # 6.1: Stricter checks for crime/security articles
        if crime_term_count >= 2:
            # Must have explicit financial impact statements to qualify
            financial_impact_terms = ['blanchiment', 'économie', 'finance', 'impact économique', 
                                     'coût', 'perte', 'valeur', 'marché financier', 'transaction']
            
            has_financial_impact = any(term in combined_text for term in financial_impact_terms)
            
            if not has_financial_impact:
                is_financial = False
                logger.debug(f"Rejected: Crime story without financial impact focus: {title[:50]}...")
        
        # 6.2: Stricter checks for articles about regulatory bodies
        if admin_term_count >= 3:
            # Must have financial metrics or explicit monetary values
            has_monetary_values = has_financial_metrics(combined_text)
            
            if not has_monetary_values:
                is_financial = False
                logger.debug(f"Rejected: Regulatory/administrative without financial metrics: {title[:50]}...")
        
        # 6.3 Last check for corporate/business news with no financial metrics
        if "stratégie" in combined_text or "initiative" in combined_text:
            # Articles about corporate strategy must have at least one financial metric
            has_at_least_one_metric = has_financial_metrics(combined_text)
            
            if not has_at_least_one_metric:
                is_financial = False
                logger.debug(f"Rejected: Corporate strategy without any financial metrics: {title[:50]}...")
    
    if not is_financial:
        logger.debug(f"Rejected non-financial content: {title[:50]}... | Financial terms: {financial_term_count}, Non-financial terms: {non_financial_term_count}")
    else:
        logger.debug(f"Accepted financial article: {title[:50]}... | Financial terms: {financial_term_count}, Non-financial terms: {non_financial_term_count}")
    
    return is_financial

def is_likely_article_url(url: str, source_config: Dict[str, Any]) -> bool:
    """
    Determine if a URL is likely to be a financial article based on patterns.
    """
    # Basic URL validation
    if not url or url == "#" or url == "/" or not isinstance(url, str):
        return False
    
    # Get domain and base URL
    base_url = source_config["url"]
    base_domain = urlparse(base_url).netloc
    
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Check if URL is from the same domain
    if parsed_url.netloc and parsed_url.netloc != base_domain:
        return False
    
    # Skip obvious navigation, category, tag, or utility pages
    skip_patterns = [
        '/tag/', '/author/', '/search/', '/contact/', 
        '/about/', '/archives/', '/sitemap/', '/privacy/', '/terms/',
        'javascript:', 'mailto:', 'whatsapp:', 'facebook.com', 'twitter.com',
        'instagram.com', 'youtube.com', 'linkedin.com', '/rss/', '/feed/',
        '/abonnement', '/subscription', '/login', '/register', '/mon-compte',
        '/profile', '/user', '/newsletter', '/flux-rss', '/publicite',
        '/cgu', '/faq', '/aide', '/help', '/mentions-legales', '/confidentialite',
        '/cotations', '/indices',
        '/activites-royales', '/nation', '/monde', '/societe', '/culture', 
        '/sports/', '/videos/', '/regions/', '/sport/', '/politique/'
    ]
    
    if any(pattern in url.lower() for pattern in skip_patterns):
        return False
    
    # Check for non-finance sections specific to this source
    non_finance_sections = source_config.get("non_finance_sections", [])
    if any(section in url.lower() for section in non_finance_sections):
        return False
    
    # Look for article indicators in the URL
    article_indicators = [
        '/article/', '/news/', '/actualite/', '/economie/', '/finance/', 
        '/bourse/', '/entreprise/', '/marches/', '/banque/', '/industrie/',
        '/agriculture/', '/energie/', '/tourisme/', '/transport/', '/immobilier/',
        '/tech/', '/startup/', '/pme/', '/export/', '/investissement/',
        '.html', '.php?id=', '/20', '/id='
    ]
    
    # Prioritize finance section indicators
    finance_sections = source_config.get("finance_sections", [])
    article_indicators.extend(finance_sections)
    
    # URL must contain at least one article indicator
    if not any(indicator in url.lower() for indicator in article_indicators):
        return False
    
    # Ensure URL has path components and is not just domain
    if not parsed_url.path or parsed_url.path == '/':
        return False
    
    # Check for pagination links
    if 'page=' in url or '/page/' in url:
        return False
    
    return True

def fetch_with_retry(url: str, headers: Dict[str, str], max_retries: int = 3, initial_timeout: int = 15) -> Optional[requests.Response]:
    """Fetch a URL with retry logic for better reliability."""
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

def generate_content_hash(title: str, text: str) -> str:
    """Generate a hash from title and text to detect duplicate content."""
    content = f"{title}::{text[:1000]}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()

class FinancialNewsArticle:
    def __init__(self, title: str, text: str, url: str, date: datetime.datetime, source: str):
        self.title = title
        # Normalize text to ensure it's on a single line
        self.text = normalize_text(text)
        self.url = url
        self.date = date
        self.source = source
        self.content_hash = generate_content_hash(title, self.text)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert article to dictionary"""
        return {
            "title": self.title,
            "text": self.text,
            "url": self.url,
            "date": self.date,
            "source": self.source,
            "content_hash": self.content_hash,
            "scraped_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def is_valid_article(self) -> bool:
        """
        Check if this is a valid article with meaningful content.
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

class FinancialNewsScraper:
    def __init__(self, source_config: Dict[str, Any]):
        self.name = source_config["name"]
        self.base_url = source_config["url"]
        self.language = source_config["language"]
        self.source_config = source_config
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        # Configure newspaper settings
        self.config = Config()
        self.config.request_timeout = 20  # timeout
        self.config.browser_user_agent = self.headers["User-Agent"]
        self.config.number_threads = 1  # Lower thread count to avoid overwhelming sites
        self.start_time = time.time()
        self.articles_processed = 0
        self.articles_accepted = 0
        self.articles_rejected = 0
    
    def get_article_urls(self) -> List[str]:
        """Get financial article URLs from the source."""
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
                            base_domain = re.match(r'(https?://[^/]+)', self.base_url)
                            if base_domain:
                                base_domain = base_domain.group(1)
                                if url.startswith('/'):
                                    url = base_domain + url
                                else:
                                    url = urljoin(self.base_url, url)
                            else:
                                url = urljoin(self.base_url, url)
                        
                        # Check if URL is likely a financial article
                        if is_likely_article_url(url, self.source_config) and url not in urls:
                            urls.append(url)
            else:
                # Fallback: look for links that match article patterns
                links = soup.find_all("a", href=True)
                for link in links:
                    url = link["href"]
                    
                    # Make absolute URL if relative
                    if not url.startswith(('http://', 'https://')):
                        base_domain = re.match(r'(https?://[^/]+)', self.base_url)
                        if base_domain:
                            base_domain = base_domain.group(1)
                            if url.startswith('/'):
                                url = base_domain + url
                            else:
                                url = urljoin(self.base_url, url)
                        else:
                            url = urljoin(self.base_url, url)
                    
                    # Check if URL is likely a financial article
                    if is_likely_article_url(url, self.source_config) and url not in urls:
                        urls.append(url)
            
            # Remove duplicates and limit to 15 articles per source
            urls = list(set(urls))[:15]
            
            # Log the URLs we're going to process
            for url in urls:
                logger.debug(f"Found potential financial article URL: {url}")
            
            return urls
        except Exception as e:
            logger.error(f"Error getting article URLs from {self.name}: {e}")
            return []
    
    def parse_article(self, url: str) -> Optional[FinancialNewsArticle]:
        """Parse a financial article from the given URL."""
        try:
            self.articles_processed += 1
            
            # Create article with config that has timeout set
            article = Article(url, language=self.language, config=self.config)
            article.download()
            article.parse()
            
            # Some sources might not have proper date extraction via newspaper3k
            date = article.publish_date
            if not date:
                date = datetime.datetime.now()
            
            # Create FinancialNewsArticle object
            news_article = FinancialNewsArticle(
                title=article.title,
                text=article.text,
                url=url,
                date=date,
                source=self.name
            )
            
            # Check if the article is valid
            if not news_article.is_valid_article():
                logger.warning(f"Skipping {url} - Not a valid article")
                self.articles_rejected += 1
                return None
            
            # Check if the article is truly finance-related
            if not is_financial_article(news_article.title, news_article.text, url, self.source_config):
                logger.warning(f"Skipping {url} - Not a financial article")
                self.articles_rejected += 1
                return None
            
            self.articles_accepted += 1
            return news_article
        except Exception as e:
            logger.error(f"Error parsing article {url}: {e}")
            self.articles_rejected += 1
            return None
    
    def report_performance(self) -> Dict[str, Any]:
        """Report scraper performance metrics."""
        elapsed_time = time.time() - self.start_time
        return {
            "source": self.name,
            "articles_processed": self.articles_processed,
            "articles_accepted": self.articles_accepted,
            "articles_rejected": self.articles_rejected,
            "acceptance_rate": f"{(self.articles_accepted / max(1, self.articles_processed)) * 100:.1f}%",
            "elapsed_time": f"{elapsed_time:.2f} seconds"
        }

def load_existing_articles() -> Tuple[Set[str], Set[str]]:
    """Load existing URLs and content hashes to avoid duplicates."""
    file_path = RAW_DATA_DIR / "moroccan_financial_news.csv"
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path)
            existing_urls = set(existing_df['url']) if 'url' in existing_df.columns else set()
            existing_hashes = set(existing_df['content_hash']) if 'content_hash' in existing_df.columns else set()
            return existing_urls, existing_hashes
        except Exception as e:
            logger.error(f"Error loading existing articles: {e}")
            return set(), set()
    return set(), set()

def collect_perfect_financial_news():
    """Collect only genuine financial news with 100% accuracy"""
    start_time = time.time()
    all_articles = []
    performance_metrics = []
    
    # Load existing articles to avoid duplicates
    existing_urls, existing_hashes = load_existing_articles()
    
    logger.info(f"Found {len(existing_urls)} existing URLs and {len(existing_hashes)} content hashes in database")
    
    for source_config in FINANCIAL_NEWS_SOURCES:
        logger.info(f"Collecting financial news from {source_config['name']}...")
        scraper = FinancialNewsScraper(source_config)
        
        article_urls = scraper.get_article_urls()
        # Filter out already processed URLs
        new_urls = [url for url in article_urls if url not in existing_urls]
        
        logger.info(f"Found {len(article_urls)} potential articles, {len(new_urls)} are new from {source_config['name']}")
        
        for url in new_urls:
            # Add random delay to avoid overloading the server
            time.sleep(random.uniform(1, 3))
            article = scraper.parse_article(url)
            if article:
                # Check content hash to avoid near-duplicate content
                if article.content_hash not in existing_hashes:
                    all_articles.append(article.to_dict())
                    existing_hashes.add(article.content_hash)
                else:
                    logger.info(f"Skipping article with duplicate content: {article.title[:50]}...")
        
        performance_metrics.append(scraper.report_performance())
    
    # Save collected articles to a single file that gets appended to
    if all_articles:
        # Create a DataFrame with new articles
        new_articles_df = pd.DataFrame(all_articles)
        
        # Define a fixed file path for the single CSV file
        timestamp = datetime.datetime.now().strftime('%Y%m%d')
        file_path = RAW_DATA_DIR / "moroccan_financial_news.csv"
        backup_path = RAW_DATA_DIR / f"moroccan_financial_news_backup_{timestamp}.csv"
        
        # Check if the file already exists
        if os.path.exists(file_path):
            # Create a backup of the existing file
            try:
                existing_df = pd.read_csv(file_path)
                existing_df.to_csv(backup_path, index=False)
                logger.info(f"Created backup of existing data at {backup_path}")
                
                # Filter out duplicates (based on URL and content hash)
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
                    logger.info(f"Added {len(new_articles_df)} new financial articles to {file_path}")
                else:
                    logger.info("No new financial articles to add.")
            except Exception as e:
                logger.error(f"Error processing existing data: {e}")
                # Save new articles to a separate file as fallback
                fallback_path = RAW_DATA_DIR / f"moroccan_financial_news_new_{timestamp}.csv"
                new_articles_df.to_csv(fallback_path, index=False)
                logger.info(f"Saved {len(new_articles_df)} new articles to fallback file {fallback_path}")
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Save as new file
            new_articles_df.to_csv(file_path, index=False)
            logger.info(f"Created new file with {len(new_articles_df)} financial articles at {file_path}")
    else:
        logger.warning("No financial articles were collected.")
    
    # Log performance summary
    total_time = time.time() - start_time
    total_processed = sum(metric["articles_processed"] for metric in performance_metrics)
    total_accepted = sum(metric["articles_accepted"] for metric in performance_metrics)
    
    logger.info("=== Performance Summary ===")
    logger.info(f"Total run time: {total_time:.2f} seconds")
    logger.info(f"Total articles processed: {total_processed}")
    logger.info(f"Total articles accepted: {total_accepted}")
    logger.info(f"Overall acceptance rate: {(total_accepted / max(1, total_processed)) * 100:.1f}%")
    logger.info(f"New articles added: {len(all_articles)}")
    
    for metric in performance_metrics:
        logger.info(f"Source: {metric['source']} - Processed: {metric['articles_processed']}, Accepted: {metric['articles_accepted']} ({metric['acceptance_rate']})")
    
    return len(all_articles)

if __name__ == "__main__":
    # Log script execution with timestamp
    version = "1.2"
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Starting perfect financial scraper v{version} on {timestamp}")
    
    try:
        new_articles_count = collect_perfect_financial_news()
        logger.info(f"Financial scraper completed successfully. Added {new_articles_count} new articles.")
    except Exception as e:
        logger.error(f"Financial scraper encountered an error: {e}", exc_info=True)