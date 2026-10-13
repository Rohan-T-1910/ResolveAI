"""Named Entity Recognition and Domain Entity Extraction module.
Demonstrates:
- Standard Named Entity Recognition (PERSON, ORG, GPE, DATE, MONEY) via spaCy
- Domain-specific regex extraction (ORDER_ID, TICKET_ID, PHONE_NUMBER, EMAIL, AMOUNT)
- Clear architectural separation between statistical NER and rule-based pattern matching
"""

import re
from typing import List, Dict, Any


class EntityExtractor:
    """Extracts both standard linguistic entities (spaCy) and e-commerce domain entities."""

    def __init__(self):
        self.nlp = None
        self._load_spacy()

        # Domain regex patterns
        self.patterns = {
            "ORDER_ID": re.compile(
                r"(?:#|order\s*(?:id|#|no\.?|number)?\s*[:#-]?\s*)([A-Za-z]{2,4}\d{4,8})\b|\b(ORD\d{4,8})\b",
                re.IGNORECASE,
            ),
            "TICKET_ID": re.compile(
                r"\b(?:CMP|TCK|TICKET|CASE)[-#]?\d{3,7}\b",
                re.IGNORECASE,
            ),
            "EMAIL": re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
            ),
            "PHONE_NUMBER": re.compile(
                r"(?:\+?91[\-\s]?)?[6-9]\d{9}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"
            ),
            "AMOUNT": re.compile(
                r"(?:₹|Rs\.?|INR|\$)\s*[\d,]+(?:\.\d{1,2})?",
                re.IGNORECASE,
            ),
        }

    def _load_spacy(self):
        """Loads spaCy model safely with fallback."""
        try:
            import spacy

            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None

    def extract_domain_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extracts structured e-commerce domain entities using regex rules."""
        domain_entities = []
        if not text:
            return domain_entities

        # 1. ORDER_ID
        for match in self.patterns["ORDER_ID"].finditer(text):
            val = match.group(1) or match.group(2) or match.group(0).strip("#")
            domain_entities.append({
                "text": val.strip(),
                "label": "ORDER_ID",
                "source": "Domain Rule-Based (Regex)",
                "start": match.start(),
                "end": match.end(),
                "description": "Unique e-commerce order identifier",
            })

        # 2. TICKET_ID
        for match in self.patterns["TICKET_ID"].finditer(text):
            domain_entities.append({
                "text": match.group(0),
                "label": "TICKET_ID",
                "source": "Domain Rule-Based (Regex)",
                "start": match.start(),
                "end": match.end(),
                "description": "Customer support ticket identifier",
            })

        # 3. EMAIL
        for match in self.patterns["EMAIL"].finditer(text):
            domain_entities.append({
                "text": match.group(0),
                "label": "EMAIL",
                "source": "Domain Rule-Based (Regex)",
                "start": match.start(),
                "end": match.end(),
                "description": "Customer email contact",
            })

        # 4. PHONE_NUMBER
        for match in self.patterns["PHONE_NUMBER"].finditer(text):
            domain_entities.append({
                "text": match.group(0),
                "label": "PHONE_NUMBER",
                "source": "Domain Rule-Based (Regex)",
                "start": match.start(),
                "end": match.end(),
                "description": "Customer telephone number",
            })

        # 5. AMOUNT
        for match in self.patterns["AMOUNT"].finditer(text):
            domain_entities.append({
                "text": match.group(0),
                "label": "AMOUNT",
                "source": "Domain Rule-Based (Regex)",
                "start": match.start(),
                "end": match.end(),
                "description": "Financial transaction or price value",
            })

        # De-duplicate matches covering the exact same span
        unique = []
        seen_spans = set()
        for ent in domain_entities:
            key = (ent["label"], ent["text"])
            if key not in seen_spans:
                seen_spans.add(key)
                unique.append(ent)

        return unique

    def extract_standard_ner(self, text: str) -> List[Dict[str, Any]]:
        """Extracts standard named entities using spaCy linguistic model."""
        standard_entities = []
        if not text or not self.nlp:
            return standard_entities

        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                standard_entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "source": "Statistical NER (spaCy)",
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "description": self._describe_spacy_label(ent.label_),
                })
        except Exception:
            pass

        return standard_entities

    def extract_all(self, text: str) -> Dict[str, Any]:
        """Extracts both domain and standard entities, returning combined & partitioned results."""
        domain_ents = self.extract_domain_entities(text)
        standard_ents = self.extract_standard_ner(text)

        all_ents = domain_ents + standard_ents

        return {
            "all_entities": all_ents,
            "domain_entities": domain_ents,
            "standard_entities": standard_ents,
            "total_count": len(all_ents),
        }

    def _describe_spacy_label(self, label: str) -> str:
        descriptions = {
            "PERSON": "People, including fictional",
            "ORG": "Companies, agencies, institutions",
            "GPE": "Countries, cities, states",
            "LOC": "Non-GPE locations, mountain ranges, water bodies",
            "DATE": "Absolute or relative dates or periods",
            "TIME": "Times smaller than a day",
            "MONEY": "Monetary values, including unit",
            "PERCENT": "Percentage (% sign or 'percent')",
            "PRODUCT": "Objects, vehicles, foods, etc. (not services)",
            "CARDINAL": "Numerals that do not fall under another type",
            "ORDINAL": "First, second, third, etc.",
        }
        return descriptions.get(label, f"Named Entity ({label})")
