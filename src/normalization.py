"""
Text Normalization & Tokenization Module
Amazon ML Challenge 2026 - Business Entity Resolution

This module provides reusable, robust text normalization and tokenization
routines for business entity resolution.

Engineering Rationale & Noise Patterns Handled:
1. Unicode NFKD Diacritic Stripping:
   - In our EDA, we observed ~15-19% non-ASCII text in Sources 2 & 3, including French
     accents (e.g. 'Café' -> 'cafe', 'Cobayes & Frères' -> 'cobayes and freres') in
     the test set. Folding Latin diacritics to plain ASCII significantly boosts recall
     across sources where one contributor stripped accents and another preserved them.
   - Non-Latin scripts (Devanagari, Bengali, etc.) are safely preserved unaltered without
     corruption or stripping of necessary phonetic matras/vowels.
2. Canonical Legal Suffix Standardization:
   - Real-world business records arrive with diverse representations of company structures
     (e.g., 'Private Limited', 'Pvt Ltd', 'Pvt. Ltd.', 'Ltd.', 'Limited', 'LLC', 'L.L.C.',
     'Corporation', 'Corp.', 'SARL', 'S.A.R.L.'). Canonicalizing them ensures similarity
     metrics score core company names rather than legal syntax discrepancies.
3. Address Abbreviation & Component Normalization:
   - Addresses feature variations like 'Rd.' vs 'Road', 'Ave' vs 'Avenue', 'N.' vs 'North',
     'Apt' vs 'Apartment'. Word-boundary-aware standardization normalizes these tokens.
   - Comma-to-space replacement ensures concatenated tokens (e.g., 'Thane,Maharashtra')
     split cleanly into 'thane maharashtra' without glued tokens.
4. Graceful Empty & Null Handling:
   - Handles the 2.6% - 3.4% empty/suspicious addresses identified during EDA without
     throwing exceptions or producing 'nan' tokens.
"""

import re
import unicodedata
from typing import List, Optional

# ---------------------------------------------------------------------------
# Precompiled Regex Patterns for Legal Suffix Normalization
# ---------------------------------------------------------------------------

# Multi-word combinations must be matched first before single-word components
LEGAL_SUFFIX_PATTERNS = [
    # Public Limited Company / PLC
    (re.compile(r"\b(?:public\s+limited\s+company|p\.?\s*l\.?\s*c\.?)\b", re.IGNORECASE), "plc"),
    # Private Limited / Pvt Ltd
    (re.compile(r"\b(?:private\s+limited|pvt\.?\s*ltd\.?|pvt\.?\s*limited|private\s+ltd\.?)\b", re.IGNORECASE), "pvt ltd"),
    # Company Limited / Co Ltd
    (re.compile(r"\b(?:company\s+limited|co\.?\s*ltd\.?|co\.?\s*limited)\b", re.IGNORECASE), "co ltd"),
    # LLC / L.L.C.
    (re.compile(r"\b(?:l\.?\s*l\.?\s*c\.?)\b", re.IGNORECASE), "llc"),
    # LLP / L.L.P.
    (re.compile(r"\b(?:l\.?\s*l\.?\s*p\.?)\b", re.IGNORECASE), "llp"),
    # French SARL / S.A.R.L.
    (re.compile(r"\b(?:s\.?\s*a\.?\s*r\.?\s*l\.?)\b", re.IGNORECASE), "sarl"),
    # French SAS / S.A.S.
    (re.compile(r"\b(?:s\.?\s*a\.?\s*s\.?)\b", re.IGNORECASE), "sas"),
    # French SCI / S.C.I.
    (re.compile(r"\b(?:s\.?\s*c\.?\s*i\.?)\b", re.IGNORECASE), "sci"),
    # French SA / S.A.
    (re.compile(r"\b(?:s\.?\s*a\.?)\b", re.IGNORECASE), "sa"),
    # Incorporated / Inc
    (re.compile(r"\b(?:incorporated|inc\.?)\b", re.IGNORECASE), "inc"),
    # Corporation / Corp
    (re.compile(r"\b(?:corporation|corp\.?)\b", re.IGNORECASE), "corp"),
    # Company / Co.
    (re.compile(r"\b(?:company|co\.)\b", re.IGNORECASE), "co"),
    # Private / Pvt
    (re.compile(r"\b(?:private|pvt\.?)\b", re.IGNORECASE), "pvt"),
    # Limited / Ltd
    (re.compile(r"\b(?:limited|ltd\.?)\b", re.IGNORECASE), "ltd"),
]

# ---------------------------------------------------------------------------
# Precompiled Regex Patterns for Address Abbreviation Normalization
# ---------------------------------------------------------------------------

# Use word boundaries (\b) to avoid mangling names or words containing these letters
ADDRESS_ABBR_PATTERNS = [
    # Thoroughfares
    (re.compile(r"\b(?:road|rd\.?)\b", re.IGNORECASE), "rd"),
    (re.compile(r"\b(?:street|st\.?)\b", re.IGNORECASE), "st"),
    (re.compile(r"\b(?:avenue|ave\.?)\b", re.IGNORECASE), "ave"),
    (re.compile(r"\b(?:drive|dr\.?)\b", re.IGNORECASE), "dr"),
    (re.compile(r"\b(?:boulevard|blvd\.?)\b", re.IGNORECASE), "blvd"),
    (re.compile(r"\b(?:lane|ln\.?)\b", re.IGNORECASE), "ln"),
    # Units
    (re.compile(r"\b(?:apartment|apt\.?)\b", re.IGNORECASE), "apt"),
    (re.compile(r"\b(?:suite|ste\.?)\b", re.IGNORECASE), "ste"),
    # Cardinal Directions (carefully bounded with word boundaries)
    (re.compile(r"\b(?:north|n\.)\b", re.IGNORECASE), "n"),
    (re.compile(r"\b(?:south|s\.)\b", re.IGNORECASE), "s"),
    (re.compile(r"\b(?:east|e\.)\b", re.IGNORECASE), "e"),
    (re.compile(r"\b(?:west|w\.)\b", re.IGNORECASE), "w"),
]

# Punctuation & Formatting Regexes
AMPERSAND_PATTERN = re.compile(r"\s*&\s*")
APOSTROPHE_PATTERN = re.compile(r"['’`]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def remove_diacritics(text: str) -> str:
    """
    Unicode NFKD normalization followed by selective removal of combining diacritical marks.
    Converts accented Latin characters like 'é', 'à', 'ç' to base ASCII equivalents 'e', 'a', 'c'.
    Non-Latin scripts (Devanagari, Bengali, etc.) are preserved unaltered without stripping
    essential vowel signs/matras, and recomposed to NFC standard.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    # Strip combining diacritical marks in Latin combining ranges (U+0300 - U+036F, U+1AB0 - U+1AFF, U+1DC0 - U+1DFF)
    filtered = "".join(
        c for c in nfkd
        if not (0x0300 <= ord(c) <= 0x036F or 0x1AB0 <= ord(c) <= 0x1AFF or 0x1DC0 <= ord(c) <= 0x1DFF)
    )
    return unicodedata.normalize("NFC", filtered)


def replace_punctuation_with_space(text: str) -> str:
    """
    Replaces Unicode punctuation (P*) and symbols (S*) with spaces while preserving
    letters (L*), numbers (N*), and non-Latin vowel marks (M*).
    """
    return "".join(" " if unicodedata.category(c).startswith(("P", "S")) else c for c in text)


def normalize_name(name: Optional[str]) -> str:
    """
    Normalizes a business name string into a canonical, cleaned form.

    Steps:
    1. Graceful empty/None handling.
    2. Unicode NFKD normalization and Latin diacritic removal (folding accents to ASCII,
       preserving non-Latin scripts like Devanagari).
    3. Lowercase conversion.
    4. Standardize '&' and ' and ' to 'and'.
    5. Canonicalize common legal suffixes (e.g. Pvt Ltd, LLC, Inc, Corp, SARL).
    6. Strip apostrophes without adding spaces (e.g., "Orelee's" -> "orelees").
    7. Replace remaining punctuation with spaces (preserves letters and digits across all alphabets).
    8. Collapse multiple whitespace and strip.

    Parameters
    ----------
    name : str or None
        Raw business name.

    Returns
    -------
    str
        Normalized canonical business name.
    """
    if not name or not isinstance(name, str):
        return ""

    text = name.strip()
    if not text:
        return ""

    # 1. Unicode diacritic stripping (accents -> ascii; non-Latin preserved)
    text = remove_diacritics(text)

    # 2. Lowercase
    text = text.lower()

    # 3. Standardize ampersands to 'and'
    text = AMPERSAND_PATTERN.sub(" and ", text)

    # 4. Standardize legal suffixes
    for pattern, replacement in LEGAL_SUFFIX_PATTERNS:
        text = pattern.sub(replacement, text)

    # 5. Remove apostrophes (e.g. "Orelee's" -> "orelees", "d'Ivoire" -> "divoire")
    text = APOSTROPHE_PATTERN.sub("", text)

    # 6. Replace punctuation with space
    text = replace_punctuation_with_space(text)

    # 7. Collapse whitespace and strip
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


def normalize_address(address: Optional[str]) -> str:
    """
    Normalizes a business address string into a canonical, cleaned form.

    Steps:
    1. Graceful empty/None handling (handles the 2.6-3.4% empty addresses in EDA).
    2. Unicode NFKD normalization and Latin diacritic removal.
    3. Lowercase conversion.
    4. Standardize address abbreviations (rd, st, ave, dr, blvd, ln, apt, ste, n, s, e, w).
    5. Replace commas and punctuation with spaces to avoid concatenating distinct tokens.
    6. Collapse multiple whitespace and strip.

    Parameters
    ----------
    address : str or None
        Raw business address.

    Returns
    -------
    str
        Normalized canonical business address.
    """
    if not address or not isinstance(address, str):
        return ""

    text = address.strip()
    if not text:
        return ""

    # 1. Unicode diacritic stripping
    text = remove_diacritics(text)

    # 2. Lowercase
    text = text.lower()

    # 3. Standardize address abbreviations
    for pattern, replacement in ADDRESS_ABBR_PATTERNS:
        text = pattern.sub(replacement, text)

    # 4. Replace commas and punctuation with spaces (preserves digits and word tokens)
    text = replace_punctuation_with_space(text)

    # 5. Collapse whitespace and strip
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


def tokenize(text: Optional[str]) -> List[str]:
    """
    Tokenizes a normalized string into non-empty whitespace-separated tokens.

    Parameters
    ----------
    text : str or None
        Text to tokenize (typically output of normalize_name or normalize_address).

    Returns
    -------
    list of str
        List of non-empty tokens.
    """
    if not text or not isinstance(text, str):
        return []
    return text.split()
