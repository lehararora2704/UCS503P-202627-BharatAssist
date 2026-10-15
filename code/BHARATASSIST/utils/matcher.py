"""
Intent Recognition and Dynamic Service Matching Engine for BharatAssist.

Features:
1. Dynamic Entity & Synonym Recognition across all 215 catalog services.
2. Action Classification: new_application, update, correction, renewal, replacement,
   status_check, cancellation, linking, transfer.
3. Information Type Classification: eligibility, documents, fees, steps, processing_time,
   general_info.
4. Out-of-Scope / Negative Pattern Detection (gun, liquor, cooking, general tech, plot purchase).
5. Ambiguity Detection (requests asking for generic "card", "certificate", "address change" without specific service).
6. Multi-stage Hybrid Scoring (Keyword entity matching + Action alignment + ChromaDB semantic retrieval).
7. Grounded Content Verification (checks if the matching service actually provides steps/docs/fees for the requested action).
"""

import re
from typing import Dict, List, Optional, Tuple, Any


ACTION_KEYWORDS = {
    "update": [
        "update", "change", "modify", "edit", "shift", "moved", "moving",
        "changed", "updating", "changing", "address change", "update address",
        "change address", "mobile update", "phone update"
    ],
    "correction": [
        "correction", "correct", "fix", "rectify", "mistake", "spelling",
        "wrong", "name change", "dob correction", "date of birth correction"
    ],
    "renewal": [
        "renew", "renewal", "renewing", "expire", "expired", "re-issue",
        "reissue", "extend", "validity"
    ],
    "replacement": [
        "duplicate", "lost", "damaged", "replacement", "replace",
        "stolen", "missing", "misplaced", "reprint"
    ],
    "status_check": [
        "status", "track", "tracking", "check status", "application status",
        "where is my", "dispatch status", "appointment status"
    ],
    "linking": [
        "link", "linking", "linked", "seed", "seeding", "connect"
    ],
    "transfer": [
        "transfer", "ownership transfer", "change of ownership"
    ],
    "new_application": [
        "apply", "application", "new", "fresh", "register", "registration",
        "enrol", "enrolment", "make", "create", "issue", "get a new"
    ]
}

INFO_REQUEST_PATTERNS = {
    "documents": [
        "document", "documents", "docs", "paperwork", "proof", "proofs",
        "id proof", "address proof", "what do i need to submit", "required documents"
    ],
    "fees": [
        "fee", "fees", "cost", "price", "how much", "charges", "payment",
        "rate", "free"
    ],
    "processing_time": [
        "time", "processing time", "duration", "how long", "days", "timeline",
        "when will i get", "tat"
    ],
    "eligibility": [
        "eligible", "eligibility", "who can apply", "age limit", "criteria",
        "condition", "qualify", "income limit"
    ],
    "steps": [
        "how to", "procedure", "process", "steps", "guide", "where do i apply",
        "how can i", "how do i", "portal", "way to"
    ]
}

UNSUPPORTED_DOMAINS = [
    r"\b(?:gun|arms|weapon|pistol|rifle|firearm)\s*(?:license|licence|permit)?\b",
    r"\b(?:liquor|wine|beer|alcohol|bar)\s*(?:shop|license|licence|permit|theka)\b",
    r"\b(?:cake|pizza|bake|cook|recipe|baking|cooking|food preparation)\b",
    r"\b(?:rocket|bomb|explosive|missile)\b",
    r"\b(?:phone to buy|best mobile|smartphone review|laptop to buy)\b",
    r"\b(?:buy|purchase)\s+(?:a\s+)?(?:plot|land|flat|house|property)\b",  # Land purchase vs land records/registration
]

GENERIC_AMBIGUOUS_PATTERNS = [
    r"^(?:how do i|how to|i want to|can i)\s+(?:update|change)\s+my\s+card\??$",
    r"^(?:i need|how to get|how do i apply for)\s+a\s+certificate\??$",
    r"^(?:i need to|how do i|how to)\s+(?:change|update)\s+my\s+address\??$",
    r"^(?:how much does it cost|what is the fee|how much is the fee)\??$",
    r"^(?:how long will it take|what is the processing time)\??$",
]

# Canonical service anchors mapping common phrases/abbreviations to primary keywords
CORE_SERVICE_SYNONYMS = {
    "aadhaar": ["aadhaar", "aadhar", "adhar", "uidai", "uid"],
    "pan": ["pan", "pan card", "pancard", "nsdl", "utiitsl", "incometax"],
    "passport": ["passport", "tatkal", "passport seva", "psk"],
    "driving_licence": ["driving licence", "driving license", "dl", "driver licence", "driver license", "learner licence", "learners licence", "learning licence"],
    "voter": ["voter", "voter id", "epic", "election card", "nvsp", "eci"],
    "ration": ["ration card", "rationcard", "ration", "pds", "nfsa"],
    "vehicle": ["vehicle", "rc", "registration certificate", "car registration", "bike registration", "vahan"],
    "pf": ["pf", "epf", "epfo", "provident fund", "uan"],
    "esic": ["esic", "esi", "employee state insurance"],
    "ayushman": ["ayushman", "pmjay", "pm-jay", "golden card", "health card"],
    "pmkisan": ["pm kisan", "pm-kisan", "pmkisan", "kisan samman"],
    "eshram": ["e-shram", "eshram", "shram card"],
    "birth_cert": ["birth certificate", "birth cert"],
    "death_cert": ["death certificate", "death cert"],
    "marriage_cert": ["marriage certificate", "marriage registration"],
    "caste_cert": ["caste certificate", "community certificate", "obc certificate", "sc certificate", "st certificate"],
    "income_cert": ["income certificate"],
    "domicile_cert": ["domicile certificate", "residence certificate", "prc"],
    "scholarship": ["scholarship", "nsp", "national scholarship"],
    "udyam": ["udyam", "msme registration"],
    "gst": ["gst", "gstin", "goods and services tax registration"],
    "fssai": ["fssai", "food licence", "food license", "foscos"],
    "fir": ["fir", "police complaint", "lost report"],
    "property_reg": ["property registration", "registry", "sale deed"],
    "land_record": ["land record", "jamabandi", "khata", "khasra", "ror", "land mutation"],
}


def is_unsupported_query(text: str) -> bool:
    """Detect queries explicitly outside any Indian civic/government services."""
    q = text.lower().strip()
    for pattern in UNSUPPORTED_DOMAINS:
        if re.search(pattern, q):
            return True
    return False


def is_generic_ambiguous(text: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if user query lacks the specific service name or outcome,
    requiring polite clarification rather than wild guessing.
    """
    q = text.lower().strip()
    clean_q = re.sub(r"[^\w\s]", "", q)

    if clean_q in ["i need a certificate how do i apply", "how do i apply for a certificate", "i need a certificate"]:
        return True, "Which government certificate do you need (for example: Birth, Death, Caste, Income, or Domicile Certificate), and for which State or Union Territory?"

    if clean_q in ["how do i update my card", "i want to update my card", "update my card"]:
        return True, "Which card would you like to update (for example: Aadhaar Card, PAN Card, Ration Card, or Voter ID) and what detail needs updating?"

    # Check if user asks about address update/change without naming any document/service
    if any(p in clean_q for p in ["change my address", "update my address", "address change", "change address", "update address"]):
        # Check if any service keyword is present
        has_service_anchor = any(
            any(syn in clean_q for syn in syn_list)
            for syn_list in CORE_SERVICE_SYNONYMS.values()
        )
        if not has_service_anchor:
            return True, "For which official document or service would you like to update your address (such as Aadhaar, Driving Licence, Voter ID, or Vehicle RC)?"

    return False, None


def detect_actions(text: str) -> List[str]:
    """Identify requested actions in user prompt."""
    q = text.lower()
    detected = []
    for action, keywords in ACTION_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", q):
                detected.append(action)
                break
    return detected


def detect_info_request(text: str) -> List[str]:
    """Identify what specific aspect of the service user wants to know."""
    q = text.lower()
    detected = []
    for info_type, keywords in INFO_REQUEST_PATTERNS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", q):
                detected.append(info_type)
                break
    return detected


def extract_matched_service_intent(
    query: str,
    services_catalog: List[Dict[str, Any]],
    chroma_chunks: List[Dict[str, Any]],
    context_service: Optional[str] = None
) -> Dict[str, Any]:
    """
    Scalable multi-stage service and intent matcher:
    1. Lexical and entity relevance scoring across the 215 records.
    2. Action congruence: prioritizes services specifically designed for that action.
    3. Integration with vector retrieval scores.
    4. Conversational context awareness: seamlessly retains active service context
       for follow-up questions (fees, documents, eligibility, steps, processing time)
       while switching cleanly when a new service is explicitly queried.
    5. Guardrails for unsupported / out-of-scope / ambiguous queries.
    """
    clean_query = query.strip()
    q_lower = clean_query.lower()

    # 1. Check unsupported domains
    if is_unsupported_query(q_lower):
        return {
            "status": "unsupported",
            "service": None,
            "action": None,
            "message": "This request is not covered by any Indian government civic service in our database."
        }

    # 2. Check ambiguous queries needing clarification (e.g. generic 'I need a certificate', 'how to change address')
    # If the user asks a completely generic open-ended question without specifying which certificate or service,
    # and did not explicitly say "for this" / "for that", clarify.
    is_amb, clarification_msg = is_generic_ambiguous(q_lower)
    if is_amb:
        return {
            "status": "clarification_needed",
            "service": None,
            "action": None,
            "message": clarification_msg
        }

    # 3. Extract entities present in query
    user_entities = set()
    for canonical, syn_list in CORE_SERVICE_SYNONYMS.items():
        if any(re.search(rf"\b{re.escape(syn)}\b", q_lower) for syn in syn_list):
            user_entities.add(canonical)

    # 4. Conversational follow-up resolution:
    # If no new entity is introduced and an active context_service is present,
    # check if this is a follow-up inquiry (e.g., fees, docs, steps, eligibility, time)
    # or general inquiry referring back to the current context.
    active_ctx_svc = None
    if context_service:
        ctx_name_clean = context_service.strip().lower()
        for s in services_catalog:
            if s.get("name", "").strip().lower() == ctx_name_clean:
                active_ctx_svc = s
                break

    detected_actions = detect_actions(q_lower)
    detected_info = detect_info_request(q_lower)

    if not user_entities and active_ctx_svc:
        # Check if query is asking for information, action, or referring by pronoun/implicit reference
        is_follow_up_aspect = bool(detected_info or detected_actions)
        is_implicit_ref = any(re.search(rf"\b{w}\b", q_lower) for w in ["it", "this", "that", "same", "charges", "cost", "fee", "fees", "paper", "papers", "detail", "details"])
        # Short query check (e.g. "fees", "how much", "documents needed", "how long", "what next")
        is_short_prompt = len(q_lower.split()) <= 6

        if is_follow_up_aspect or is_implicit_ref or is_short_prompt:
            return {
                "status": "matched",
                "service": active_ctx_svc,
                "score": 5.0,
                "vector_score": 0.5,
                "detected_actions": detected_actions,
                "detected_info": detected_info
            }

    # 5. Map query to service candidate scores
    scored_candidates = []
    
    # Pre-map Chroma results for vector score blending
    chroma_score_map = {}
    for chunk in chroma_chunks:
        c_name = chunk.get("name", "").strip().lower()
        if c_name:
            chroma_score_map[c_name] = chunk.get("score", 0.0)

    # Tokenize query words
    query_tokens = set(re.findall(r"\w+", q_lower))

    # Extract entities present in query
    user_entities = set()
    for canonical, syn_list in CORE_SERVICE_SYNONYMS.items():
        if any(re.search(rf"\b{re.escape(syn)}\b", q_lower) for syn in syn_list):
            user_entities.add(canonical)

    for svc in services_catalog:
        name = svc.get("name", "")
        name_lower = name.lower()
        name_tokens = set(re.findall(r"\w+", name_lower))

        # Check entity domains for this service
        svc_entities = set()
        for canonical, syn_list in CORE_SERVICE_SYNONYMS.items():
            if any(re.search(rf"\b{re.escape(syn)}\b", name_lower) for syn in syn_list):
                svc_entities.add(canonical)

        # STRICT ENTITY ISOLATION:
        # If user explicitly specified a service entity (e.g. 'passport'), do not match services belonging to different entities (e.g. 'driving_licence' or 'vehicle')
        if user_entities:
            if svc_entities and user_entities.isdisjoint(svc_entities):
                continue

        score = 0.0

        # Exact / substring match on full service name
        if name_lower in q_lower:
            score += 4.0
        elif len(name_tokens.intersection(query_tokens)) > 0:
            # Word overlap with name
            overlap = name_tokens.intersection(query_tokens)
            # Penalize generic stop words
            significant_overlap = [w for w in overlap if w not in {"and", "or", "for", "the", "in", "of", "to", "card", "service", "registration", "application"}]
            score += len(significant_overlap) * 1.5

        # Check domain synonyms & entity overlap
        if user_entities and not user_entities.isdisjoint(svc_entities):
            score += 6.0
        else:
            for canonical, syn_list in CORE_SERVICE_SYNONYMS.items():
                user_has_syn = any(re.search(rf"\b{re.escape(syn)}\b", q_lower) for syn in syn_list)
                svc_has_syn = any(re.search(rf"\b{re.escape(syn)}\b", name_lower) for syn in syn_list)
                if user_has_syn and svc_has_syn:
                    score += 3.0

        # Action congruence bonus
        # If user asked for "update" / "change" / "address", and service name has "Update", "Change", "Re-issue", "Shift", boost!
        if any(act in ["update", "correction"] for act in detected_actions):
            if any(w in name_lower for w in ["update", "correction", "change", "reprint", "modification", "re-issue", "reissue", "shift"]):
                score += 3.0
            elif "application (fresh)" in name_lower or "fresh" in name_lower:
                score -= 2.0  # Penalize fresh application when user specifically asked for correction/update

        if "renewal" in detected_actions:
            if "renewal" in name_lower or "re-issue" in name_lower:
                score += 3.0
            elif "learner" in name_lower and "driving licence" in q_lower:
                score -= 1.5

        if "status_check" in detected_actions:
            if "status" in name_lower or "track" in name_lower:
                score += 3.0

        if "replacement" in detected_actions:
            if "duplicate" in name_lower or "replacement" in name_lower or "pvc" in name_lower:
                score += 3.0

        if "transfer" in detected_actions:
            if "transfer" in name_lower:
                score += 3.5

        if "linking" in detected_actions:
            if "link" in name_lower:
                score += 3.5

        # Address specific boosting
        if any(w in q_lower for w in ["address", "residence", "shift", "relocate"]):
            if any(w in name_lower for w in ["address", "shift"]):
                score += 3.0

        # Blend with Chroma vector score if available
        v_score = chroma_score_map.get(name_lower, 0.0)
        score += (v_score * 3.0)

        if score > 1.5:
            scored_candidates.append({
                "service": svc,
                "score": score,
                "vector_score": v_score,
                "name": name
            })

    if not scored_candidates:
        return {
            "status": "not_found",
            "service": None,
            "action": detected_actions,
            "message": "I am sorry, this information is not available in our database. May I help you with anything else?"
        }

    # Sort descending by score
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    best = scored_candidates[0]

    # Verify minimum confidence
    if best["score"] < 2.0 and best["vector_score"] < 0.35:
        return {
            "status": "not_found",
            "service": None,
            "action": detected_actions,
            "message": "I am sorry, this information is not available in our database. May I help you with anything else?"
        }

    matched_svc = best["service"]
    return {
        "status": "matched",
        "service": matched_svc,
        "score": best["score"],
        "vector_score": best["vector_score"],
        "detected_actions": detected_actions,
        "detected_info": detected_info
    }


def validate_grounded_evidence(service: Dict[str, Any], requested_info: List[str]) -> Dict[str, Any]:
    """
    Check if the database actually contains sufficient verified fields
    for the user's specific information request (fees, documents, eligibility, steps).
    Prevents fabricating missing details.
    """
    missing_fields = []
    
    if "fees" in requested_info:
        fees = str(service.get("fees") or "").strip()
        if not fees or fees.lower() in ["not specified", "unverified", "none", "n/a"]:
            missing_fields.append("fees")
            
    if "documents" in requested_info:
        docs = str(service.get("documents_required") or "").strip()
        if not docs or docs.lower() in ["not specified", "unverified", "none", "n/a"]:
            missing_fields.append("documents")

    if "processing_time" in requested_info:
        ptime = str(service.get("processing_time") or "").strip()
        if not ptime or ptime.lower() in ["not specified", "unverified", "none", "n/a"]:
            missing_fields.append("processing_time")

    if "eligibility" in requested_info:
        elig = str(service.get("eligibility") or "").strip()
        if not elig or elig.lower() in ["not specified", "unverified", "none", "n/a"]:
            missing_fields.append("eligibility")

    return {
        "is_complete": len(missing_fields) == 0,
        "missing_fields": missing_fields
    }

