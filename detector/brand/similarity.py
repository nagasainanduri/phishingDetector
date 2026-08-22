import idna

# A minimal mapping of common cyrillic/greek homoglyphs to latin characters
HOMOGLYPH_MAP = {
    'а': 'a', 'с': 'c', 'е': 'e', 'о': 'o', 'р': 'p', 'х': 'x', 'у': 'y',
    'і': 'i', 'ј': 'j', 'ѕ': 's', 'ԁ': 'd', 'ԛ': 'q', 'ԝ': 'w'
}

def decode_punycode(domain: str) -> str:
    """Decodes xn-- domains to their unicode representation."""
    try:
        # idna.decode handles subdomains as well
        return idna.decode(domain)
    except Exception:
        return domain

def normalize_homoglyphs(text: str) -> str:
    """Normalizes known homoglyphs to standard ASCII."""
    result = []
    for char in text:
        result.append(HOMOGLYPH_MAP.get(char, char))
    return ''.join(result)

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates the Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
        
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def compute_similarity(s1: str, s2: str) -> float:
    """
    Returns a similarity score between 0.0 and 1.0.
    1.0 means exactly identical.
    """
    if not s1 and not s2:
        return 1.0
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len)
