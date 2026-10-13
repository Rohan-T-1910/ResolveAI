"""Resolution and Response Recommendation Engine.
Produces:
- Recommended handling department
- Recommended actionable workflow steps for support agents
- Suggested customer response draft using deterministic professional templates
- Incorporates detected entities (Order ID, Amount, Email) responsibly
- Strict policy: NEVER claims an action has already been completed.
"""

from typing import Dict, Any, List
from src.config import CATEGORY_DEPARTMENT_MAP


class ResolutionEngine:
    """Generates support routing and deterministic customer response recommendations."""

    def __init__(self):
        self.actions = {
            "Delivery Issue": (
                "Verify shipment tracking status with the logistics partner. "
                "Contact the local delivery hub to check dispatch logs, confirm customer delivery address, "
                "and issue an updated delivery estimate within 24 hours."
            ),
            "Payment Issue": (
                "Cross-check payment gateway transaction reference against merchant ledger. "
                "Confirm whether debit was captured or pending. If duplicate or unauthorized, "
                "freeze the transaction and initiate banking reconciliation."
            ),
            "Refund Issue": (
                "Check return inspection approval status and payment reversal SLA. "
                "Verify bank account/UPI destination details and coordinate with the finance team "
                "to ensure credit processing timeline is communicated."
            ),
            "Product Issue": (
                "Review uploaded photos or customer defect report. "
                "Verify product warranty and seller return policy. "
                "Arrange reverse pickup for inspection and schedule replacement dispatch or store credit."
            ),
            "Account Issue": (
                "Review recent login attempts, IP geo-location, and password reset logs. "
                "If unauthorized activity is suspected, temporarily secure the account, "
                "verify customer identity through registered secondary channels, and trigger secure reset."
            ),
            "Technical Issue": (
                "Log error codes and browser/app version details into engineering ticketing. "
                "Test checkout/search repro steps on staging environment and provide customer with "
                "workaround steps or platform status update."
            ),
            "Cancellation": (
                "Check fulfillment status in warehouse management system. "
                "If order has not shipped, trigger immediate order cancellation. "
                "If already in transit, initiate return-to-origin with courier partner."
            ),
            "Other": (
                "Review customer inquiry details, categorize request under appropriate support queue, "
                "and assign to customer service specialist for personalized follow-up."
            ),
        }

        self.response_templates = {
            "Delivery Issue": (
                "Dear Customer, we sincerely apologize for the delay regarding your shipment{order_ref}. "
                "Our logistics operations team is currently checking the transit status with our delivery partner. "
                "We are actively tracking your package and will follow up with an updated delivery estimate shortly."
            ),
            "Payment Issue": (
                "Dear Customer, thank you for alerting us regarding your payment concern{order_ref}{amount_ref}. "
                "Our billing team is currently verifying the transaction logs with our payment gateway partner. "
                "If a duplicate charge or failed transaction is confirmed, the reversal process will be initiated in accordance with banking guidelines."
            ),
            "Refund Issue": (
                "Dear Customer, we understand your concern regarding the pending refund{order_ref}{amount_ref}. "
                "Our accounts receivable team is reviewing the return inspection records and refund queue. "
                "We are working to ensure your refund status is updated and will share confirmation details as soon as verified."
            ),
            "Product Issue": (
                "Dear Customer, we are very sorry to hear that the item you received{order_ref} did not meet quality standards. "
                "Our quality assurance desk has been notified. We recommend initiating a replacement or return request, "
                "and our team will assist you with reverse pickup and resolution."
            ),
            "Account Issue": (
                "Dear Customer, we take account security and access issues very seriously. "
                "Our customer security team is investigating the access difficulty on your account. "
                "For your safety, we recommend not sharing your verification codes, and a specialist will contact you with secure recovery instructions."
            ),
            "Technical Issue": (
                "Dear Customer, thank you for reporting this technical difficulty. "
                "Our engineering team has been notified of the error and is investigating the platform behavior. "
                "We appreciate your patience while we work to resolve this issue and restore seamless service."
            ),
            "Cancellation": (
                "Dear Customer, we have received your cancellation request{order_ref}. "
                "Our fulfillment team is verifying the dispatch status with our warehouse. "
                "We will confirm whether the order can be stopped immediately or routed back upon delivery."
            ),
            "Other": (
                "Dear Customer, thank you for reaching out to ResolveAI customer support. "
                "We have logged your query and forwarded it to the relevant support specialist. "
                "Our team will review your details and get back to you with further information."
            ),
        }

        self.positive_response_templates = {
            "Delivery Issue": (
                "Dear Customer, thank you so much for your positive feedback regarding your delivery{order_ref}! "
                "We are thrilled to know your shipment arrived promptly and in great condition. "
                "We truly appreciate your support and look forward to serving you again."
            ),
            "Product Issue": (
                "Dear Customer, thank you so much for sharing your wonderful feedback{order_ref}! "
                "We are delighted to hear that you are pleased with the quality of your item. "
                "Thank you for shopping with us!"
            ),
            "Refund Issue": (
                "Dear Customer, thank you for your feedback{order_ref}. "
                "We are glad to know that your refund process was completed smoothly to your satisfaction. "
                "Thank you for your patience and for choosing us."
            ),
            "Payment Issue": (
                "Dear Customer, thank you for your positive feedback{order_ref}. "
                "We are pleased to hear that your transaction was handled seamlessly. "
                "Thank you for your trust and continued support."
            ),
            "Cancellation": (
                "Dear Customer, thank you for your feedback{order_ref}. "
                "We are glad our support team could assist you promptly with your order request. "
                "We look forward to serving you again in the future."
            ),
            "Account Issue": (
                "Dear Customer, thank you for your kind feedback! "
                "We are delighted that your account assistance was resolved smoothly and securely. "
                "Please reach out anytime you need support."
            ),
            "Technical Issue": (
                "Dear Customer, thank you for your feedback! "
                "We are glad our platform provided you with a smooth and responsive experience. "
                "Thank you for being a valued user."
            ),
            "Default": (
                "Dear Customer, thank you so much for your kind words and positive feedback{order_ref}! "
                "We are delighted to hear about your great experience with our service. "
                "We truly appreciate your support and look forward to serving you again."
            ),
        }

    def generate_recommendation(
        self,
        category: str,
        priority: str,
        entities: List[Dict[str, Any]] = None,
        sentiment: str = "Neutral",
    ) -> Dict[str, Any]:
        """Generates resolution department, action items, and suggested response draft."""
        department = CATEGORY_DEPARTMENT_MAP.get(category, "General Customer Support Services")

        # Extract entity references for personalized draft
        order_ref = ""
        amount_ref = ""

        if entities:
            order_ids = [e["text"] for e in entities if e["label"] == "ORDER_ID"]
            amounts = [e["text"] for e in entities if e["label"] == "AMOUNT"]
            if order_ids:
                order_ref = f" (Order #{order_ids[0].strip('#')})"
            if amounts:
                amount_ref = f" of {amounts[0]}"

        # Handle positive comments/feedback separately
        if (sentiment or "").strip().lower() == "positive":
            action = "No action required. Customer provided positive feedback or appreciation."
            positive_template = self.positive_response_templates.get(
                category, self.positive_response_templates["Default"]
            )
            suggested_response = positive_template.format(order_ref=order_ref, amount_ref=amount_ref)
            agent_notes = [
                "POSITIVE FEEDBACK: No resolution action or escalation required. Ticket can be closed as resolved."
            ]
            policy_advisory = "Feedback recorded. Share customer appreciation with the respective team."
        else:
            action = self.actions.get(category, self.actions["Other"])
            template = self.response_templates.get(category, self.response_templates["Other"])
            suggested_response = template.format(order_ref=order_ref, amount_ref=amount_ref)

            # Priority-based agent workflow notes
            agent_notes = []
            if priority == "Critical":
                agent_notes.append("URGENT SLA: 1-hour first-contact target. Immediate manager escalation required.")
            elif priority == "High":
                agent_notes.append("HIGH PRIORITY: 4-hour SLA target. Prioritize in queue.")
            else:
                agent_notes.append("STANDARD SLA: 24-hour standard resolution window.")

            policy_advisory = (
                "Agent Reminder: Never confirm financial reversals or completed deliveries "
                "until verified in primary backend enterprise systems."
            )

        return {
            "department": department,
            "recommended_action": action,
            "suggested_response": suggested_response,
            "agent_notes": agent_notes,
            "policy_advisory": policy_advisory,
        }
