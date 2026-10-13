"""End-to-End Pipeline and Assistant Tests.
Validates 5 complete real-world e-commerce complaint analysis flows.
"""

from src.pipeline import ResolvePipeline
from src.chatbot.assistant import ComplaintAssistant


def test_flow_1_delayed_delivery():
    pipeline = ResolvePipeline()
    assistant = ComplaintAssistant()

    complaint = (
        "My order #ORD45821 was supposed to arrive yesterday but it still hasn't arrived. "
        "I tried contacting the delivery partner but there is no response. This is extremely frustrating."
    )
    analysis = pipeline.analyze_complaint(complaint, complaint_id="TEST-FLOW-1")

    assert analysis["category"] == "Delivery Issue"
    assert analysis["sentiment"] == "Negative"
    assert analysis["priority"] in ["High", "Critical"]
    assert any(e["label"] == "ORDER_ID" and "ORD45821" in e["text"] for e in analysis["entities"])
    assert "Logistics" in analysis["department"]
    assert len(analysis["summary"]) > 0

    # Assistant query
    ans_issue = assistant.answer_question("What is the issue?", analysis)
    assert "Delivery Issue" in ans_issue

    ans_prio = assistant.answer_question("Why is the priority high?", analysis)
    assert "priority" in ans_prio.lower()


def test_flow_2_unauthorized_charge():
    pipeline = ResolvePipeline()
    assistant = ComplaintAssistant()

    complaint = (
        "Someone made an unauthorized transaction of ₹4,999 on my account. "
        "I received an SMS notification with order ORD89124 which I never placed! "
        "Block my account and refund the money immediately. This is fraud!"
    )
    analysis = pipeline.analyze_complaint(complaint, complaint_id="TEST-FLOW-2")

    assert analysis["priority"] == "Critical"
    assert any(e["label"] == "AMOUNT" for e in analysis["entities"])
    assert any("fraud" in r.lower() or "security" in r.lower() for r in analysis["priority_reasons"])

    ans_dept = assistant.answer_question("Which department should handle this?", analysis)
    assert analysis["department"] in ans_dept


def test_flow_3_damaged_product():
    pipeline = ResolvePipeline()
    assistant = ComplaintAssistant()

    complaint = (
        "The wireless headphones I received under order ORD33012 have a broken headband and "
        "the right earbud is not working at all. The package arrived torn. "
        "I request a replacement or full refund."
    )
    analysis = pipeline.analyze_complaint(complaint, complaint_id="TEST-FLOW-3")

    assert analysis["category"] in ["Product Issue", "Refund Issue"]
    assert any(e["label"] == "ORDER_ID" for e in analysis["entities"])

    ans_action = assistant.answer_question("What action should be taken?", analysis)
    assert "Action" in ans_action


def test_flow_4_refund_sla_breach():
    pipeline = ResolvePipeline()
    assistant = ComplaintAssistant()

    complaint = (
        "I cancelled order #ORD77142 over 12 days ago. Customer support promised my refund of "
        "₹2,450 within 5-7 business days, but nothing has credited to my bank account. "
        "Please check with your finance department."
    )
    analysis = pipeline.analyze_complaint(complaint, complaint_id="TEST-FLOW-4")

    assert analysis["category"] == "Refund Issue"
    assert analysis["priority"] in ["High", "Critical"]
    assert "Finance" in analysis["department"]

    ans_reply = assistant.answer_question("What response should we send?", analysis)
    assert "Customer" in ans_reply


def test_flow_5_account_locked():
    pipeline = ResolvePipeline()
    assistant = ComplaintAssistant()

    complaint = (
        "My account linked to rohan.support@example.com is locked and I cannot log in even "
        "after multiple password reset attempts. The reset OTP is never received on my phone +91-9876543210."
    )
    analysis = pipeline.analyze_complaint(complaint, complaint_id="TEST-FLOW-5")

    assert analysis["category"] == "Account Issue"
    assert any(e["label"] == "EMAIL" for e in analysis["entities"])
    assert any(e["label"] == "PHONE_NUMBER" for e in analysis["entities"])
    assert "Security" in analysis["department"] or "Account" in analysis["department"]
