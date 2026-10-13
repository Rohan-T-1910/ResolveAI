"""Transparent Rule-Based Priority Detection Engine.
Calculates priority levels: Low, Medium, High, Critical
Provides explicit, auditable reasons for priority assignment based on:
- Security / fraud / unauthorized transaction signals
- Strong negative sentiment & frustration intensity
- Delivery failure and courier unreachability
- Financial loss / refund delay breaches
- Urgent temporal indicators (ASAP, immediately, emergency)
"""

import re
from typing import Dict, List, Any


class PriorityEngine:
    """Evaluates complaint urgency using transparent heuristic rule signals."""

    def __init__(self):
        # Critical urgency keywords
        self.critical_keywords = [
            "fraud", "unauthorized", "unauthorised", "hacked", "stolen", "scam",
            "identity theft", "compromised", "police", "legal action", "cyber crime"
        ]

        # High urgency keywords
        self.high_urgency_keywords = [
            "immediately", "urgent", "urgently", "emergency", "asap", "unacceptable",
            "frustrated", "frustrating", "terrible", "worst", "harassment", "sue",
            "no response", "refused", "lost package", "broken seal", "expired"
        ]

        # Temporal / SLA breach patterns
        self.sla_breach_pattern = re.compile(
            r"(\b(?:over|past|more than)\s+\d+\s+days\b|\b\d+\s+days\s+(?:late|ago|delayed)\b|\btwo weeks\b|\bthree times\b)",
            re.IGNORECASE,
        )

    def determine_priority(
        self,
        text: str,
        category: str = "Other",
        sentiment: str = "Neutral",
        compound_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Calculates priority level and returns transparent justifications."""
        t = (text or "").lower()
        reasons: List[str] = []
        score = 0

        # 1. Critical Security / Fraud Signals
        found_critical = [w for w in self.critical_keywords if w in t]
        if found_critical:
            score += 80
            reasons.append(f"Security or fraud signal detected: '{', '.join(found_critical)}'.")

        # 2. Sentiment Intensity Signals
        if compound_score <= -0.45:
            score += 35
            reasons.append(f"Strong negative sentiment detected (valence compound: {compound_score:+.2f}).")
        elif compound_score <= -0.15 or sentiment == "Negative":
            score += 20
            reasons.append(f"Negative sentiment detected (valence compound: {compound_score:+.2f}).")

        # 3. Explicit Urgent Language
        found_urgency = [w for w in self.high_urgency_keywords if w in t]
        if found_urgency:
            score += 25
            reasons.append(f"High urgency / frustration terminology: '{', '.join(found_urgency[:3])}'.")

        # 4. SLA Breaches / Repeated Failure Signals
        sla_matches = self.sla_breach_pattern.findall(t)
        if sla_matches:
            score += 25
            reasons.append(f"Service Level Agreement (SLA) breach indicator: '{sla_matches[0]}'.")

        # 5. Category-Specific Risk Rules
        if category == "Payment Issue" and any(w in t for w in ["deducted", "double", "charged twice", "unauthorized"]):
            score += 30
            reasons.append("Unresolved monetary debit or duplicate payment transaction.")
        elif category == "Refund Issue" and any(w in t for w in ["days", "waiting", "promised", "not credited"]):
            score += 25
            reasons.append("Unfulfilled refund promise exceeding expected processing timeline.")
        elif category == "Delivery Issue" and any(w in t for w in ["not arrived", "no response", "failed"]):
            score += 20
            reasons.append("Delivery failure with communication breakdown.")
        elif category == "Account Issue" and any(w in t for w in ["locked", "cannot log in", "otp"]):
            score += 25
            reasons.append("Account accessibility lockout preventing user platform access.")

        # Determine Priority Level based on transparent score thresholds
        if score >= 70 or any("fraud" in r or "Security" in r for r in reasons):
            priority = "Critical"
        elif score >= 40:
            priority = "High"
        elif score >= 20:
            priority = "Medium"
        else:
            priority = "Low"
            if not reasons:
                reasons.append("Standard inquiry without urgent escalation indicators.")

        return {
            "priority": priority,
            "priority_score": score,
            "reasons": reasons,
            "disclaimer": (
                "Priority is assigned by deterministic rule heuristics analyzing sentiment, "
                "keywords, and category risk; it is not clinically or commercially validated."
            ),
        }
