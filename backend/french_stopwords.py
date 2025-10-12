def get_french_stopwords():
# Core French stopwords
    FRENCH_STOPWORDS = {
        "a", "à", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en", "et", 
        "eux", "il", "ils", "je", "j'ai", "j'", "la", "le", "les", "leur", "lui", "ma", "mais", "me", 
        "même", "mes", "moi", "mon", "nos", "notre", "nous", "on", "ou", "par", "pas", "pour", "qu", 
        "que", "qui", "s", "sa", "se", "si", "son", "sur", "ta", "te", "tes", "toi", "ton", "tu", "un", 
        "une", "votre", "vous", "c", "d", "j", "l", "m", "n", "s", "t", "y", "été", "étée", "étées", 
        "étés", "étant", "étante", "étants", "étantes", "suis", "es", "est", "sommes", "êtes", "sont", 
        "serai", "seras", "sera", "serons", "serez", "seront", "serais", "serait", "serions", "seriez", 
        "seraient", "étais", "était", "étions", "étiez", "étaient", "fus", "fut", "fûmes", "fûtes", 
        "furent", "sois", "soit", "soyons", "soyez", "soient", "fusse", "fusses", "fût", "fussions", 
        "fussiez", "fussent", "avoir", "ayant", "ayante", "ayantes", "ayants", "eu", "eue", "eues", 
        "eus", "ai", "as", "avons", "avez", "ont", "aurai", "auras", "aura", "aurons", "aurez", 
        "auront", "aurais", "aurait", "aurions", "auriez", "auraient", "avais", "avait", "avions", 
        "aviez", "avaient", "eut", "eûmes", "eûtes", "eurent", "aie", "aies", "ait", "ayons", "ayez", 
        "aient", "eusse", "eusses", "eût", "eussions", "eussiez", "eussent", "ceci", "cela", "celà", 
        "cet", "cette", "ici", "ils", "les", "leurs", "quel", "quels", "quelle", "quelles", "sans", 
        "soi", "ai", "as", "ah", "ainsi", "alors", "au", "aucun", "aucune", "aujourd", "aujourd'hui", 
        "auquel", "aussi", "autre", "autres", "aux", "auxquelles", "auxquels", "après", "assez", 
        "beaucoup", "bien", "car", "c'est", "ceci", "cela", "celle", "celles", "celui", "cependant", 
        "certaines", "certains", "chacun", "chacune", "chaque", "chez", "ci", "combien", "comme", 
        "comment", "concernant", "contre", "d'après", "d'un", "d'une", "dans", "davantage", "de", 
        "dehors", "delà", "depuis", "dernier", "dernière", "des", "dès", "désormais", "desquelles", 
        "desquels", "dessous", "dessus", "devant", "devers", "devra", "divers", "diverses", "doit", 
        "donc", "dont", "du", "duquel", "durant", "dès", "déjà", "elle", "elles", "en", "encore", 
        "enfin", "entre", "envers", "environ", "es", "est", "et", "etc", "etre", "être", "eu", "eux", 
        "excepté", "hormis", "hors", "hélas", "hui", "huit", "huitième", "il", "ils", "importe", 
        "je", "jusqu", "jusque", "la", "laquelle", "le", "lequel", "les", "lesquelles", "lesquels", 
        "leur", "leurs", "lors", "lorsque", "lui", "là", "ma", "mais", "malgré", "me", "merci", 
        "mes", "mien", "mienne", "miennes", "miens", "moi", "moins", "mon", "moyennant", "même", 
        "mêmes", "n'avait", "n'y", "ne", "néanmoins", "ni", "non", "nos", "notamment", "notre", 
        "nous", "néanmoins", "nôtre", "nôtres", "n'a", "n'est", "n'ont", "n'était",
        "ouf", "oh", "ou", "où", "par", "parmi", "partant", "pas", "passé", "pendant", "peu", 
        "plus", "plusieurs", "pour", "pourquoi", "près", "puisque", "qu", "quand", "que", "quel", 
        "quelle", "quelles", "quelque", "quelques", "quelqu'un", "quels", "qui", "quoi", "quoique", 
        "revoici", "revoilà", "s'il", "sa", "sans", "sauf", "se", "selon", "seront", "ses", "si", 
        "sien", "sienne", "siennes", "siens", "sinon", "soi", "soit", "son", "sont", "sous", "suivant", 
        "sur", "ta", "tandis", "tant", "tes", "toi", "ton", "toujours", "tous", "tout", "toute", 
        "toutes", "très", "trop", "tu", "un", "une", "unes", "uns", "vers", "via", "voici", "voilà", 
        "vos", "votre", "vous", "vu", "vôtre", "vôtres", "y",
        
        # Add these exact forms as they appear in the top features
        # Important: These must match exactly how they appear in the tokens
        "l'", "d'", "s'", "n'", "c'", "j'", "m'", "t'", "qu'",
        
        # Add variants of these forms to cover all bases
        "l", "d", "s", "n", "c", "j", "m", "t", "qu",
        
        # Add alternate forms that might be created during preprocessing
        "l_", "d_", "s_", "n_", "c_", "j_", "m_", "t_", "qu_"
    }
    
    # Numbers and common dates in French
    NUMBERS_DATES = {
        "zero", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
        "onze", "douze", "treize", "quatorze", "quinze", "seize", "vingt", "trente", "quarante",
        "cinquante", "soixante", "cent", "mille", "million", "milliard", "premier", "première",
        "second", "seconde", "deuxième", "troisième", "quatrième", "cinquième", "janvier", "février",
        "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", 
        "décembre", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"
    }
    
    # Common financial terms that should not be removed as stopwords
    FINANCIAL_TERMS_TO_KEEP = {
        "banque", "bourse", "action", "dividende", "obligation", "titre", "finance", "économie",
        "marché", "investissement", "entreprise", "société", "croissance", "inflation", "taux",
        "crédit", "prêt", "dette", "budget", "déficit", "bénéfice", "résultat", "impôt", "taxe",
        "commerce", "exportation", "importation", "industrie", "production", "énergie", "tourisme",
        "immobilier", "transport", "agricole", "assurance", "capitalisation", "investisseur",
        "actionnaire", "rendement", "liquidité", "financement", "placement", "portefeuille",
        "profit", "perte", "rentabilité", "hausse", "baisse", "acquisition", "fusion", "holding",
        "filiale", "développement", "stratégie", "performance", "risque", "valorisation"
    }
    
    # French stopwords specific to news articles
    NEWS_STOPWORDS = {
        "afin", "alors", "après", "aujourd'hui", "auprès", "avant", "avoir", "c'est", "c'est-à-dire",
        "cependant", "concernant", "d'après", "d'autres", "d'entre", "dans", "depuis", "dernier",
        "dernière", "donc", "durant", "effet", "également", "entre", "environ", "faire", "fait",
        "fois", "grâce", "indique", "lors", "notamment", "nouveau", "nouvelle", "parce", "part",
        "particulier", "particulière", "pendant", "permet", "permettre", "plusieurs", "plutôt",
        "précise", "près", "prochaine", "prochain", "propos", "puis", "selon", "sera", "seront",
        "seulement", "suite", "toujours", "toutefois", "travers", "très", "vient"
    }
    
    # Financial document specific words that don't add meaning
    FINANCIAL_DOC_STOPWORDS = {
        "communiqué", "presse", "annonce", "annoncé", "déclaré", "publié", "rapport", "indiqué",
        "précisé", "souligné", "ajouté", "expliqué", "affirmé", "confirmé", "dit", "déclaration",
        "information", "informations", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
        "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
    }
    
    # Create the comprehensive stopwords set
    all_stopwords = FRENCH_STOPWORDS.union(NUMBERS_DATES).union(NEWS_STOPWORDS).union(FINANCIAL_DOC_STOPWORDS)
    
    # Remove financial terms that should be kept
    final_stopwords = all_stopwords - FINANCIAL_TERMS_TO_KEEP
    
    # Add lemmatized versions of stopwords to handle preprocessing consistency
    lemmatized_stopwords = set()
    for word in final_stopwords:
        # Add base form and common lemmatized forms
        lemmatized_stopwords.add(word.lower())
        # Add common lemmatized endings for french words
        if len(word) > 4:
            lemmatized_stopwords.add(word.lower().rstrip('es'))
            lemmatized_stopwords.add(word.lower().rstrip('s'))
            lemmatized_stopwords.add(word.lower().rstrip('e'))
    
    return lemmatized_stopwords

FRENCH_STOPWORDS = get_french_stopwords()