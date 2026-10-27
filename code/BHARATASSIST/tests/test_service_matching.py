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

    def test_arms_gun_license_supported(self):
        """Arms / Gun license is supported via official MHA NDAL-ALIS portal."""
        res = self.client.post("/api/assistant", json={
            "question": "How to get a gun license in India?",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        self.assertEqual(data.get("context_service"), "Arms / Gun Licence Application (NDAL-ALIS)")

    def test_unsupported_rocket_declined(self):
        """Negative test: Explosive/rocket question -> Declines politely, no hallucination."""
        res = self.client.post("/api/assistant", json={
            "question": "How to make an explosive missile?",
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


    def test_conversational_context_follow_up_and_switching(self):
        """
        Verify multi-turn flow:
        1. Query "passport renew" -> Matches Passport Renewal or Re-issue.
        2. Follow-up "fees" (with context_service) -> Retains Passport Renewal and answers fees.
        3. Follow-up "documents" (with context_service) -> Retains Passport Renewal and answers documents.
        4. New service query "How do I renew driving licence?" -> Cleanly transitions context to Driving Licence Renewal.
        """
        # Step 1: Initial query
        res1 = self.client.post("/api/assistant", json={
            "question": "passport renew",
            "language": "English"
        })
        self.assertEqual(res1.status_code, 200)
        data1 = res1.get_json()
        self.assertTrue(data1.get("success"))
        self.assertTrue(data1.get("grounded"))
        self.assertEqual(data1.get("context_service"), "Passport Renewal or Re-issue")
        active_svc = data1.get("context_service")

        # Step 2: Follow-up "fees"
        res2 = self.client.post("/api/assistant", json={
            "question": "fees",
            "language": "English",
            "context_service": active_svc
        })
        self.assertEqual(res2.status_code, 200)
        data2 = res2.get_json()
        self.assertTrue(data2.get("success"))
        self.assertTrue(data2.get("grounded"))
        self.assertEqual(data2.get("context_service"), "Passport Renewal or Re-issue")
        self.assertTrue(
            any(kw in data2.get("answer", "").lower() for kw in ["fee", "1,500", "1500", "rs", "inr", "passport"]),
            f"Answer was: {data2.get('answer')}"
        )

        # Step 3: Follow-up "what are the documents required"
        res3 = self.client.post("/api/assistant", json={
            "question": "what are the documents required",
            "language": "English",
            "context_service": active_svc
        })
        self.assertEqual(res3.status_code, 200)
        data3 = res3.get_json()
        self.assertTrue(data3.get("success"))
        self.assertTrue(data3.get("grounded"))
        self.assertEqual(data3.get("context_service"), "Passport Renewal or Re-issue")

        # Step 4: Context switch to a completely new service
        res4 = self.client.post("/api/assistant", json={
            "question": "How do I renew driving licence?",
            "language": "English",
            "context_service": active_svc
        })
        self.assertEqual(res4.status_code, 200)
        data4 = res4.get_json()
        self.assertTrue(data4.get("success"))
        self.assertTrue(data4.get("grounded"))
        self.assertEqual(data4.get("context_service"), "Driving Licence Renewal")

    def test_category_scheme_discovery_farmers(self):
        """User asks 'which schemes for farmers' -> Returns list of farmer welfare schemes."""
        res = self.client.post("/api/assistant", json={
            "question": "which schemes for farmers",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        ans = data.get("answer", "")
        self.assertIn("Government Schemes & Services for Farmers", ans)
        self.assertTrue("PM-KISAN" in ans or "Kisan" in ans)

    def test_category_scheme_discovery_employment(self):
        """User asks 'employment related services' -> Returns list of employment & labour schemes."""
        res = self.client.post("/api/assistant", json={
            "question": "what are the schemes for employment",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        ans = data.get("answer", "")
        self.assertIn("Employment, Skill & Labour Welfare Schemes", ans)
        self.assertTrue("e-Shram" in ans or "Employment" in ans or "EPFO" in ans)



    def test_duplicate_pan_card_intent(self):
        """Duplicate PAN Card request -> Maps to PAN Card Correction or Reprint, not fresh application."""
        res = self.client.post("/api/assistant", json={
            "question": "How to get a duplicate PAN card?",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        self.assertEqual(data.get("context_service"), "PAN Card Correction or Reprint")
        self.assertNotEqual(data.get("context_service"), "PAN Card Application")

    def test_name_correction_pan_card(self):
        """Name correction in PAN Card -> Maps to PAN Card Correction or Reprint, not fresh application."""
        res = self.client.post("/api/assistant", json={
            "question": "How do I correct my name in PAN card?",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        self.assertEqual(data.get("context_service"), "PAN Card Correction or Reprint")
        self.assertNotEqual(data.get("context_service"), "PAN Card Application")

    def test_aadhar_apply_vs_update_discrimination(self):
        """'aadhar card apply' -> Maps to Aadhaar Enrolment (Fresh), not Aadhaar Update."""
        res = self.client.post("/api/assistant", json={
            "question": "aadhar card apply",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        self.assertEqual(data.get("context_service"), "Aadhaar Enrolment")
        self.assertNotEqual(data.get("context_service"), "Aadhaar Update")

    def test_liquor_licence_discrimination_from_driving_licence(self):
        """Liquor Licence inquiry -> Returns Liquor Licence, never confuses with Driving Licence."""
        res = self.client.post("/api/assistant", json={
            "question": "How do I apply for a liquor licence?",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        self.assertEqual(data.get("context_service"), "Liquor Licence (Commercial / Bar / Restaurant / Retail Shop)")
        self.assertNotIn("driving", data.get("answer", "").lower())
        self.assertNotIn("parivahan", str(data.get("sources", [])).lower())

    def test_vehicle_rc_duplicate_intent(self):
        """Duplicate Vehicle RC -> Maps to Vehicle Registration Certificate Duplicate."""
        res = self.client.post("/api/assistant", json={
            "question": "How to get duplicate RC book for vehicle?",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        self.assertEqual(data.get("context_service"), "Vehicle Registration Certificate Duplicate")
        self.assertNotEqual(data.get("context_service"), "Driving Licence Duplicate")

    def test_grievance_consumer_complaint(self):
        """Consumer complaint filing -> Maps to Consumer Complaint e-Daakhil."""
        res = self.client.post("/api/assistant", json={
            "question": "how to file a consumer complaint against a shop",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        self.assertEqual(data.get("context_service"), "Consumer Complaint e-Daakhil")

    def test_grievance_cyber_crime(self):
        """Cyber crime reporting -> Maps to Cyber Crime Complaint."""
        res = self.client.post("/api/assistant", json={
            "question": "how to file a cyber crime complaint",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        self.assertEqual(data.get("context_service"), "Cyber Crime Complaint")

    def test_aspect_only_fees_concise_response(self):
        """Aspect-only inquiry: 'What is the fee for PAN card?' -> Returns concise fee response without dumping full steps."""
        res = self.client.post("/api/assistant", json={
            "question": "What is the fee for PAN card?",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        self.assertEqual(data.get("context_service"), "PAN Card Application")
        self.assertIn("Government Fee", data.get("answer"))
        self.assertNotIn("### 📋 Step-by-Step", data.get("answer"))

    def test_state_specific_inquiry_delhi(self):
        """State-specific question: 'How do I apply for a ration card in Delhi?' -> Mentions Delhi jurisdiction."""
        res = self.client.post("/api/assistant", json={
            "question": "How do I apply for a ration card in Delhi?",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        self.assertEqual(data.get("context_service"), "Ration Card (New Application)")
        self.assertIn("Delhi", data.get("answer"))

    def test_undocumented_action_limitation_warning(self):
        """Undocumented action: 'How to renew birth certificate?' -> Explains action limitation clearly."""
        res = self.client.post("/api/assistant", json={
            "question": "How to renew birth certificate?",
            "language": "English"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("grounded"))
        self.assertEqual(data.get("context_service"), "Birth Certificate")
        self.assertIn("Action Note", data.get("answer"))
        self.assertIn("crsorgi.gov.in", data.get("answer"))


if __name__ == "__main__":
    unittest.main()

