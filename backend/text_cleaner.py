import re
import unicodedata
import string
from typing import List, Optional, Union, Dict, Any

class FinancialTextCleaner:

    def __init__(self, 
                 keep_numbers: bool = True, 
                 keep_pct: bool = True, 
                 keep_monetary: bool = True,
                 lowercase: bool = True,
                 remove_punct: bool = True,
                 remove_special_chars: bool = True,
                 normalize_whitespace: bool = True,
                 preserve_entities: bool = True):

        self.keep_numbers = keep_numbers
        self.keep_pct = keep_pct
        self.keep_monetary = keep_monetary
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.remove_special_chars = remove_special_chars
        self.normalize_whitespace = normalize_whitespace
        self.preserve_entities = preserve_entities
        
        # Compile common patterns for efficiency
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.monetary_pattern = re.compile(r'(\d+(?:[\.,]\d+)?)\s*(?:€|\$|£|MAD|DH|dirhams?|euros?|dollars?|MDH|MMDH)')
        self.pct_pattern = re.compile(r'(\d+(?:[\.,]\d+)?)\s*(?:%|pour cent|points de pourcentage)')
        self.number_pattern = re.compile(r'\b\d+(?:[\.,]\d+)?\b')
        self.whitespace_pattern = re.compile(r'\s+')
        self.punct_pattern = re.compile(f'[{re.escape(string.punctuation)}]')
        
        # Special financial abbreviations to keep
        self.financial_abbr = {
            'pib', 'cac', 'dow', 'nyse', 'nasdaq', 'masi', 'madex', 'ftse', 's&p', 'btp', 'bce', 'fed',
            'bam', 'cih', 'bmce', 'ifrs', 'bvmt', 'anc', 'ammc', 'bnp', 'hsbc', 'lvmh', 'pme', 'pmi',
            'mdh', 'mmdh', 'md', 'mm', 'ifc', 'sfi', 'hcp', 'onee', 'berd'
        }
        
        # Company names and financial entities to preserve
        self.financial_entities = [
            'maroc telecom', 'bank al-maghrib', 'tamwilcom', 'alliances', 'haut-commissariat au plan',
            'cih bank', 'credit agricole', 'attijariwafa', 'bmce capital', 'bourse de casablanca',
            'societe financiere internationale', 'ifc', 'banque europeenne', 'berd', 'tresor', 
            'reserve federale', 'fed', 'banque mondiale', 'fmi', 'onee', 'anrac', 'aradei capital',
            'akdital', 'maroc', 'morocco', 'opep', 'opec', 'wall street journal'
        ]
        
        # Compile entity patterns
        self.entity_patterns = [
            re.compile(rf'\b{re.escape(entity)}\b', re.IGNORECASE) 
            for entity in self.financial_entities
        ]
        
        # List of common French articles with apostrophes to remove
        self.articles_to_remove = [
            "l'", "d'", "s'", "n'", "c'", "j'", "m'", "t'", "qu'", "jusqu'", "lorsqu'", 
            "puisqu'", "quoiqu'", "aujourd'", "presqu'", "quelqu'", "entr'"
        ]
        
        # Apostrophe handling, meaningful phrases to preserve
        self.apostrophe_preservations = {
            "d'affaires": "d_affaires", 
            "l'économie": "l_economie", 
            "l'entreprise": "l_entreprise", 
            "l'investissement": "l_investissement", 
            "d'investissement": "d_investissement",
            "l'industrie": "l_industrie", 
            "d'euros": "d_euros", 
            "l'année": "l_annee", 
            "d'activité": "d_activite", 
            "l'accord": "l_accord", 
            "l'ensemble": "l_ensemble", 
            "l'ONEE": "l_ONEE", 
            "l'IFC": "l_IFC", 
            "l'énergie": "l_energie", 
            "d'émission": "d_emission", 
            "l'horizon": "l_horizon",
            "s'est": "s_est",
            "n'est": "n_est",
            "c'est": "c_est",
            "d'un": "d_un",
            "d'une": "d_une",
            "l'on": "l_on",
            "s'agit": "s_agit",
            "qu'il": "qu_il",
            "qu'elle": "qu_elle",
            "d'autres": "d_autres",
            "l'offre": "l_offre",
            "l'analyse": "l_analyse"
        }
        
        # Compile apostrophe patterns
        self.apostrophe_patterns = [
            re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE) 
            for term in self.apostrophe_preservations.keys()
        ]
        
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        return self.url_pattern.sub(' ', text)
    
    def remove_emails(self, text: str) -> str:
        """Remove emails from text"""
        return self.email_pattern.sub(' ', text)
    
    def normalize_monetary_values(self, text: str) -> str:
        if self.keep_monetary:
            # Standardize million/milliard formatting
            text = re.sub(r'(\d+(?:[\.,]\d+)?)\s*milliards?\s*(?:de)?\s*(?:dirhams?|dh|mad)', r'\1_MMDH', text, flags=re.IGNORECASE)
            text = re.sub(r'(\d+(?:[\.,]\d+)?)\s*millions?\s*(?:de)?\s*(?:dirhams?|dh|mad)', r'\1_MDH', text, flags=re.IGNORECASE)
            
            # Standardize other currency formats
            text = re.sub(r'(\d+(?:[\.,]\d+)?)\s*(?:€|euros?)', r'\1_EUR', text, flags=re.IGNORECASE)
            text = re.sub(r'(\d+(?:[\.,]\d+)?)\s*(?:\$|dollars?|usd)', r'\1_USD', text, flags=re.IGNORECASE)
            text = re.sub(r'(\d+(?:[\.,]\d+)?)\s*(?:dirhams?|dh|mad)', r'\1_MAD', text, flags=re.IGNORECASE)
            
            return text
        else:
            return self.monetary_pattern.sub(' ', text)
    
    def normalize_percentages(self, text: str) -> str:

        if self.keep_pct:
            text = re.sub(r'(\d+(?:[\.,]\d+)?)\s*(?:%|pour cent|points de pourcentage)', r'\1_PCT', text, flags=re.IGNORECASE)
            return text
        else:
            return self.pct_pattern.sub(' ', text)
    
    def handle_numbers(self, text: str) -> str:
 
        if not self.keep_numbers:
            # Keep numbers that are part of monetary or percentage expressions
            if not self.monetary_pattern.search(text) and not self.pct_pattern.search(text):
                text = self.number_pattern.sub(' ', text)
        return text
    
    def normalize_accents(self, text: str) -> str:

        # into the combination of simple ones
        text = unicodedata.normalize('NFD', text)
        
        if self.remove_special_chars:
            # Keep only ASCII characters and whitespace
            text = ''.join(c for c in text if not unicodedata.combining(c))
        
        return text
    
    def handle_punctuation(self, text: str) -> str:
        if self.remove_punct:
            # Replace punctuation with spaces
            text = self.punct_pattern.sub(' ', text)
        return text
    
    def normalize_case(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()
        return text
    
    def clean_whitespace(self, text: str) -> str:
        if self.normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text).strip()
        return text
    
    def preserve_financial_abbreviations(self, text: str) -> str:
        # This function ensures that abbreviations like MDH, MMDH, etc are preserved
        for abbr in self.financial_abbr:
            pattern = re.compile(rf'\b{re.escape(abbr)}\b', re.IGNORECASE)
            text = pattern.sub(lambda m: m.group(0).lower(), text)
        return text
    
    def handle_apostrophes(self, text: str) -> str:
        # Remove common French articles with apostrophes
        for article in self.articles_to_remove:
            text = re.sub(rf'\b{re.escape(article)}\b', ' ', text)
        
        # Preserve specific terms with apostrophes by replacing with underscore
        for i, pattern in enumerate(self.apostrophe_patterns):
            term = list(self.apostrophe_preservations.keys())[i]
            replacement = self.apostrophe_preservations[term]
            text = pattern.sub(replacement, text)
        
        # Replace remaining apostrophes with space
        text = re.sub(r'(\w)\'(\w)', r'\1 \2', text)
        
        return text
    
    def preserve_named_entities(self, text: str) -> str:
        if self.preserve_entities:
            # Replace spaces in financial entities with underscores
            for pattern in self.entity_patterns:
                text = pattern.sub(lambda m: m.group(0).replace(" ", "_"), text)
        
        return text
    
    def clean(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        
        # Apply all cleaning steps in sequence
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        
        # Apply case normalization early
        text = self.normalize_case(text)
        
        # Handle apostrophes before entity preservation
        text = self.handle_apostrophes(text)
        
        # Preserve named entities before other transformations
        text = self.preserve_named_entities(text)
        
        # Normalize values
        text = self.normalize_monetary_values(text)
        text = self.normalize_percentages(text)
        text = self.handle_numbers(text)
        
        # Normalize accents and handle punctuation
        text = self.normalize_accents(text)
        text = self.handle_punctuation(text)
        
        # Preserve financial abbreviations
        text = self.preserve_financial_abbreviations(text)
        
        # Final whitespace cleaning
        text = self.clean_whitespace(text)
        
        return text