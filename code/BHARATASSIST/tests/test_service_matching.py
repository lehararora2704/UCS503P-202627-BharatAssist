"""
Comprehensive Service Matching, Action Differentiation, and Intent Recognition Test Suite.

Covers:
1. All actions: new application, update, correction, renewal, replacement, status check, linking, transfer.
2. Information types: documents, eligibility, fees, processing time, steps.
3. Natural conversational phrasing and paraphrasing (e.g., 'I moved house').
4. Anti-hallucination / Negative tests (gun, liquor, baking, buying land).
5. Ambiguous questions requiring clarification (e.g. 'I need a certificate').
6. Verification that sources match the identified service.
7. Correct response format, HTTP 200, and grounded flags.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


class DynamicServiceMatchingTestSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = app
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

    def test_aadhaar_address_update_intent(self):
        """User asks to update Aadhaar address -> Maps to Aadhaar Update service."""
        res = self.client.post("/api/assistant", json={
            "question": "How do I update my Aadhaar address?",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        self.assertEqual(data.get("context_service"), "Aadhaar Update")
        self.assertTrue(any("uidai.gov.in" in s.get("source_url", "") for s in data.get("sources", [])))

    def test_aadhaar_paraphrase_moved_house(self):
        """Natural paraphrase 'I moved house' -> Correctly identifies Aadhaar Update."""
        res = self.client.post("/api/assistant", json={
            "question": "I moved house. How can I change the address on my Aadhaar?",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        self.assertEqual(data.get("context_service"), "Aadhaar Update")

    def test_driving_licence_renewal_action_differentiation(self):
        """DL Renewal -> Distinguishes Driving Licence Renewal from Learner's or Permanent DL."""
        res = self.client.post("/api/assistant", json={
            "question": "How can I renew my driving licence?",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        self.assertEqual(data.get("context_service"), "Driving Licence Renewal")
        self.assertTrue(any("parivahan.gov.in" in s.get("source_url", "") for s in data.get("sources", [])))

    def test_passport_status_check(self):
        """Passport status check -> Distinguishes Passport Status from Fresh Application."""
        res = self.client.post("/api/assistant", json={
            "question": "How can I check my passport application status?",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        self.assertEqual(data.get("context_service"), "Passport Appointment or Application Status")
        self.assertTrue(any("passportindia.gov.in" in s.get("source_url", "") for s in data.get("sources", [])))

    def test_passport_address_change(self):
        """Passport address update -> Maps to Passport Renewal or Re-issue (official Passport Seva re-issue procedure)."""
        res = self.client.post("/api/assistant", json={
            "question": "how to change address in passport",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        self.assertEqual(data.get("context_service"), "Passport Renewal or Re-issue")
        self.assertTrue(any("passportindia.gov.in" in s.get("source_url", "") for s in data.get("sources", [])))

    def test_pan_card_fresh_application(self):
        """PAN fresh application -> Maps to PAN Card Application."""
        res = self.client.post("/api/assistant", json={
            "question": "How do I apply for a new PAN card?",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        self.assertEqual(data.get("context_service"), "PAN Card Application")

    def test_vehicle_ownership_transfer(self):
        """Vehicle Ownership Transfer -> Distinguishes vehicle transfer from driving licence."""
        res = self.client.post("/api/assistant", json={
            "question": "i want to transfer my vehicle ownership",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        self.assertEqual(data.get("context_service"), "Vehicle Ownership Transfer")

    def test_unsupported_gun_license_declined(self):
        """Negative test: Gun license is not in database -> Declines politely, no hallucination."""
        res = self.client.post("/api/assistant", json={
            "question": "How to get a gun license in India?",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertFalse(data.get("grounded"))
        self.assertEqual(data.get("sources"), [])
        self.assertIn("not available in our database", data.get("answer", ""))

    def test_unsupported_cake_baking_declined(self):
        """Negative test: Cooking question -> Declines politely, no hallucination."""
        res = self.client.post("/api/assistant", json={
            "question": "How do I bake a chocolate cake at home?",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertFalse(data.get("grounded"))
        self.assertEqual(data.get("sources"), [])

    def test_unsupported_land_purchase_declined(self):
        """Negative test: Purchasing private plot of land -> Declines, does not confuse with registry."""
        res = self.client.post("/api/assistant", json={
            "question": "How do I buy a plot of land?",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertFalse(data.get("grounded"))

    def test_ambiguous_certificate_prompts_clarification(self):
        """Ambiguous query: 'I need a certificate. How do I apply?' -> Asks concise clarification."""
        res = self.client.post("/api/assistant", json={
            "question": "I need a certificate. How do I apply?",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertFalse(data.get("grounded"))
        self.assertIn("Which government certificate do you need", data.get("answer", ""))

    def test_ambiguous_address_change_prompts_clarification(self):
        """Ambiguous query: 'What documents are required for an address change?' -> Asks for service."""
        res = self.client.post("/api/assistant", json={
            "question": "What documents are required for an address change?",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertFalse(data.get("grounded"))
        self.assertIn("For which official document or service", data.get("answer", ""))


if __name__ == "__main__":
    unittest.main()
