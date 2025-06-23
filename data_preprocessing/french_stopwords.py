"""
French stopwords list for text preprocessing.
"""

FRENCH_STOPWORDS = [
    "a", "à", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en", "et", "eux",
    "il", "ils", "je", "j'", "j'ai", "la", "le", "les", "leur", "lui", "ma", "mais", "me", "même",
    "mes", "moi", "mon", "ni", "notre", "nous", "on", "ou", "par", "pas", "pour", "qu'", "que", "qui",
    "s'", "sa", "se", "si", "son", "sur", "ta", "te", "tes", "toi", "ton", "tu", "un", "une", "votre",
    "vous", "c'", "d'", "j'", "l'", "m'", "n'", "s'", "t'", "y", "été", "étée", "étées", "étés", "étant",
    "suis", "es", "est", "sommes", "êtes", "sont", "serai", "seras", "sera", "serons", "serez", "seront",
    "serais", "serait", "serions", "seriez", "seraient", "étais", "était", "étions", "étiez", "étaient",
    "fus", "fut", "fûmes", "fûtes", "furent", "sois", "soit", "soyons", "soyez", "soient", "fusse",
    "fusses", "fût", "fussions", "fussiez", "fussent", "avoir", "ayant", "eu", "eue", "eues", "eus",
    "ai", "as", "avons", "avez", "ont", "aurai", "auras", "aura", "aurons", "aurez", "auront", "aurais",
    "aurait", "aurions", "auriez", "auraient", "avais", "avait", "avions", "aviez", "avaient", "eut",
    "eûmes", "eûtes", "eurent", "aie", "aies", "ait", "ayons", "ayez", "aient", "eusse", "eusses", "eût",
    "eussions", "eussiez", "eussent", "ceci", "celà", "cet", "cette", "ici", "ils", "les", "leurs",
    "quel", "quels", "quelle", "quelles", "sans", "soi"
]

# Extended list for financial context
FINANCIAL_STOPWORDS = [
    "banque", "finances", "économie", "marché", "bourse", "actions", "dividendes", "investissement",
    "investisseur", "capital", "rendement", "croissance", "baisse", "hausse", "euro", "euros", "dirham",
    "dh", "mad", "million", "milliard", "pourcent", "pourcentage", "trimestre", "semestre", "annuel",
    "annuellement", "fiscal", "fiscale", "impôt", "impôts", "taxe", "taxes", "budget", "déficit", "dette",
    "profit", "bénéfice", "perte", "revenu", "chiffre", "affaires", "vente", "ventes", "entreprise", 
    "société", "groupe", "filiale", "actionnaire", "actionnaires", "conseil", "administration", "directeur",
    "président", "ceo", "pib", "inflation", "taux", "intérêt", "cours", "clôture", "ouverture", "transaction"
]

# Combine the standard stopwords with financial stopwords
COMBINED_STOPWORDS = FRENCH_STOPWORDS + FINANCIAL_STOPWORDS