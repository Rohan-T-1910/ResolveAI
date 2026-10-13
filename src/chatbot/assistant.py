"""Complaint-Aware Interactive Assistant module.
Demonstrates:
- Local conversational Q&A over current complaint analysis
- Deterministic intent matching without external paid LLM APIs
- Precise answers grounded directly in the active NLP analysis object
"""

from typing import Dict, Any, List


class ComplaintAssistant:
    """Answers support agent questions regarding the active complaint analysis."""

    def __init__(self):
        pass

    def answer_question(self, question: str, analysis: Dict[str, Any]) -> str:
        """Answers agent queries grounded in the current analysis context."""
        q = (question or "").strip().lower()
        if not q:
            return "Please type a question about this complaint (e.g., 'What is the issue?', 'Why is priority high?')."

        if not analysis:
            return (
                "No complaint has been analyzed yet. Please analyze a customer complaint first "
                "or load one from the history."
            )

        # 1. Category / Issue questions
        if any(w in q for w in ["what is the issue", "what is the complaint", "category", "what issue", "problem"]):
            cat = analysis.get("category", "Unspecified")
            conf = analysis.get("classification", {}).get("confidence_percentage", "")
            method = analysis.get("classification", {}).get("method", "Classifier")
            return f"The complaint is classified as a **{cat}** ({conf}% confidence via {method})."

        # 2. Priority questions
        if any(w in q for w in ["why", "priority", "urgency", "urgent"]):
            priority = analysis.get("priority", "Normal")
            reasons = analysis.get("priority_reasons", [])
            reasons_bullet = "\n".join(f"- {r}" for r in reasons) if reasons else "- Standard baseline priority."
            return (
                f"The priority level is **{priority}**.\n\n"
                f"**Specific signals detected:**\n{reasons_bullet}"
            )

        # 3. Department / Routing questions
        if any(w in q for w in ["department", "who should handle", "route", "team", "assign"]):
            dept = analysis.get("department", "Customer Support")
            return f"The recommended handling department is **{dept}**."

        # 4. Action questions
        if any(w in q for w in ["action", "what should i do", "next step", "procedure", "how to resolve"]):
            action = analysis.get("recommended_action", "Review ticket details.")
            return f"**Recommended Action for Agent:**\n{action}"

        # 5. Customer response questions
        if any(w in q for w in ["response", "reply", "what should we send", "customer reply", "message"]):
            resp = analysis.get("suggested_response", "")
            return f"**Suggested Customer Response Draft:**\n\n\"{resp}\""

        # 6. Sentiment questions
        if any(w in q for w in ["sentiment", "tone", "emotion", "feeling", "angry", "happy"]):
            sent = analysis.get("sentiment", "Neutral")
            score = analysis.get("sentiment_score", 0.0)
            expl = analysis.get("sentiment_details", {}).get("explanation", "")
            return f"The detected sentiment is **{sent}** (compound score: `{score:+.2f}`).\n{expl}"

        # 7. Summary questions
        if any(w in q for w in ["summary", "summarize", "brief", "overview", "tldr"]):
            summary = analysis.get("summary", "No summary available.")
            return f"**Extractive Complaint Summary:**\n\"{summary}\""

        # 8. Entity questions
        if any(w in q for w in ["entity", "entities", "order id", "ticket id", "phone", "email", "amount"]):
            entities = analysis.get("entities", [])
            if not entities:
                return "No structured entities (such as Order ID, Ticket ID, Amount, or Email) were detected in this complaint."
            ent_list = "\n".join(f"- **{e['label']}**: `{e['text']}` ({e.get('source', 'Detected')})" for e in entities)
            return f"**Detected Entities:**\n{ent_list}"

        # 9. Greeting / Help
        if any(w in q for w in ["hello", "hi", "hey", "help"]):
            return (
                "Hello! I am your ResolveAI Complaint Assistant. You can ask me:\n"
                "- *What is the issue?*\n"
                "- *Why is this complaint high priority?*\n"
                "- *Which department should handle this?*\n"
                "- *What action should be taken?*\n"
                "- *What entities were detected?*\n"
                "- *What is the customer sentiment?*\n"
                "- *What response should we send?*"
            )

        # Fallback
        return (
            f"I can answer questions regarding this complaint's category, priority reasons, "
            f"sentiment, detected entities, summarization, routing department, or suggested response. "
            f"Try asking: *'Why is the priority {analysis.get('priority', 'assigned')}?'* or *'What is the recommended action?'*"
        )
