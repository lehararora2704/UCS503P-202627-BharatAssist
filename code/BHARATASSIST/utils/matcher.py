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
        "change address", "mobile update", "phone update", "अपडेट", "बदलना", "बदलाव", "सुधार", "ਅਪਡੇਟ", "ਬਦਲਣਾ"
    ],
    "correction": [
        "correction", "correct", "fix", "rectify", "mistake", "spelling",
        "wrong", "name change", "dob correction", "date of birth correction", "संशोधन", "ਸੁਧਾਰ"
    ],
    "renewal": [
        "renew", "renewal", "renewing", "expire", "expired", "re-issue",
        "reissue", "extend", "validity", "नवीनीकरण", "ਰੀਨਿਊ"
    ],
    "replacement": [
        "duplicate", "lost", "damaged", "replacement", "replace",
        "stolen", "missing", "misplaced", "reprint", "pvc", "pvc card", "डुप्लीकेट", "ਗੁੰਮ"
    ],
    "grievance": [
        "grievance", "complaint", "complain", "report fraud", "report problem",
        "consumer court", "consumer forum", "redressal", "cpgrams", "cyber crime",
        "cyber fraud", "fir", "police complaint", "शिकायत", "ਸਿਕਾਇਤ"
    ],
    "status_check": [
        "status", "track", "tracking", "check status", "application status",
        "where is my", "dispatch status", "appointment status", "स्थिति", "ਸਟੇਟਸ"
    ],
    "linking": [
        "link", "linking", "linked", "seed", "seeding", "connect", "जोड़ना", "ਲਿੰਕ"
    ],
    "transfer": [
        "transfer", "ownership transfer", "change of ownership", "तबादला", "ਟ੍ਰਾਂਸਫਰ"
    ],
    "new_application": [
        "apply", "application", "new", "fresh", "register", "registration",
        "enrol", "enrolment", "make", "create", "issue", "get a new",
        "banwana", "banaye", "banvana", "karna hai", "karwana", "apply karna",
        "अप्लाई", "आवेदन", "बनवाना", "बनवाएं", "नया", "रजिस्ट्रेशन", "ਅਪਲਾਈ", "ਬਣਵਾਉਣਾ", "ਨਵਾਂ", "ਬਣਾਉਣਾ"
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
    "aadhaar": ["aadhaar", "aadhar", "adhar", "uidai", "uid", "आधार", "ਆਧਾਰ", "आधार कार्ड", "ਆਧਾਰ ਕਾਰਡ"],
    "pan": ["pan", "pan card", "pancard", "nsdl", "utiitsl", "incometax", "पैन", "पैन कार्ड", "ਪੈਨ ਕਾਰਡ"],
    "passport": ["passport", "tatkal", "passport seva", "psk", "पासपोर्ट", "ਪਾਸਪੋਰਟ"],
    "driving_licence": ["driving licence", "driving license", "dl", "driver licence", "driver license", "learner licence", "learners licence", "learning licence", "ड्राइविंग लाइसेंस", "ਡਰਾਈਵਿੰਗ ਲਾਇਸੈਂਸ", "लाइसेंस", "ਲਾਇਸੈਂਸ"],
    "liquor_license": ["liquor license", "liquor licence", "liquor permit", "bar license", "bar licence", "alcohol permit", "excise license", "excise licence", "beer license", "wine license", "party liquor permit", "occasional liquor permit", "commercial liquor", "theka", "liquor", "alcohol licence", "alcohol license", "beer bar", "शराब", "शराब लाइसेंस", "ठेका", "दारू", "ਸ਼ਰਾਬ ਲਾਇਸੈਂਸ"],
    "arms_license": ["arms license", "arms licence", "gun license", "gun licence", "pistol license", "weapon license", "rifle license", "firearm license", "firearm licence", "ndal", "alis", "हथियार", "बंदूक लाइसेंस"],
    "trade_license": ["trade license", "trade licence", "shop license", "shop licence", "shop and establishment", "commercial licence", "commercial license", "municipal trade", "ट्रेड लाइसेंस", "दुकान लाइसेंस"],
    "fire_noc": ["fire noc", "fire safety", "fire license", "fire licence", "fire safety certificate", "फायर एनओसी"],
    "senior_citizen": ["senior citizen", "senior citizen card", "elderly card", "old age card", "senior citizen certificate", "वरिष्ठ नागरिक", "बुजुर्ग कार्ड"],
    "udid": ["udid", "disability card", "disability certificate", "handicapped certificate", "divyang card", "swavlamban", "दिव्यांग", "विकलांग कार्ड"],
    "ews": ["ews", "ews certificate", "economically weaker section", "income and asset certificate", "ईडब्ल्यूएस"],
    "legal_heir": ["legal heir", "varisu", "surviving member", "succession certificate", "heirship", "वारिस"],
    "voter": ["voter", "voter id", "epic", "election card", "nvsp", "eci", "वोटर आईडी", "ਵੋਟਰ ਕਾਰਡ", "मतदाता पहचान पत्र"],
    "ration": ["ration card", "rationcard", "ration", "pds", "nfsa", "राशन कार्ड", "ਰਾਸ਼ਨ ਕਾਰਡ"],
    "vehicle": ["vehicle", "rc", "registration certificate", "car registration", "bike registration", "vahan", "वाहन पंजीकरण", "आरसी"],
    "pf": ["pf", "epf", "epfo", "provident fund", "uan", "पीएफ"],
    "esic": ["esic", "esi", "employee state insurance", "ईएसआईसी"],
    "ayushman": ["ayushman", "pmjay", "pm-jay", "golden card", "health card", "आयुष्मान", "गोल्डन कार्ड"],
    "pmkisan": ["pm kisan", "pm-kisan", "pmkisan", "kisan samman", "पीएम किसान"],
    "eshram": ["e-shram", "eshram", "shram card", "ई-श्रम"],
    "birth_cert": ["birth certificate", "birth cert", "जन्म प्रमाण पत्र", "ਜਨਮ ਸਰਟੀਫਿਕੇਟ"],
    "death_cert": ["death certificate", "death cert", "मृत्यु प्रमाण पत्र", "ਮੌਤ ਸਰਟੀਫਿਕੇਟ"],
    "marriage_cert": ["marriage certificate", "marriage registration", "विवाह प्रमाण पत्र", "ਵਿਆਹ ਸਰਟੀਫਿਕੇਟ"],
    "caste_cert": ["caste certificate", "community certificate", "obc certificate", "sc certificate", "st certificate", "जाति प्रमाण पत्र", "ਜਾਤੀ ਸਰਟੀਫਿਕੇਟ"],
    "income_cert": ["income certificate", "आय प्रमाण पत्र", "ਆਮਦਨ ਸਰਟੀਫਿਕੇਟ"],
    "domicile_cert": ["domicile certificate", "residence certificate", "prc", "निवास प्रमाण पत्र", "मूल निवास", "ਰਿਹਾਇਸ਼ੀ ਸਰਟੀਫਿਕੇਟ"],
    "scholarship": ["scholarship", "nsp", "national scholarship"],
    "udyam": ["udyam", "msme registration"],
    "gst": ["gst", "gstin", "goods and services tax registration"],
    "fssai": ["fssai", "food licence", "food license", "foscos"],
    "fir": ["fir", "police complaint", "lost report"],
    "property_reg": ["property registration", "registry", "sale deed"],
    "land_record": ["land record", "jamabandi", "khata", "khasra", "ror", "land mutation"],
    "cpgrams": ["cpgrams", "pgportal", "public grievance", "grievance portal", "central grievance", "शिकायत"],
    "consumer_forum": ["consumer court", "consumer forum", "consumer complaint", "edaakhil", "e-daakhil", "consumer helpline"],
    "cyber_crime": ["cyber crime", "cyber fraud", "online fraud", "1930", "cyber helpline", "cyber police", "cyber complaint"],
    "legal_aid": ["legal aid", "nalsa", "free legal services"],
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


CATEGORY_SCHEME_GROUPS = {
    "farmer": {
        "keywords": ["farmer", "farmers", "kisan", "agriculture", "krishi", "fasal", "crop", "cultivator", "farming", "crop insurance"],
        "title": "Government Schemes & Services for Farmers",
        "description": "Key government services and welfare schemes available for farmers:"
    },
    "employment": {
        "keywords": ["employment", "job", "jobs", "career", "recruitment", "apprentice", "apprenticeship", "shram", "labour", "worker", "epfo", "esic", "unemployed", "rojgar", "workforce"],
        "title": "Employment, Skill & Labour Welfare Schemes",
        "description": "Key employment, career, training, and social security services:"
    },
    "health": {
        "keywords": ["health", "medical", "hospital", "treatment", "ayushman", "pmjay", "cghs", "jan aushadhi", "abha", "health insurance", "medicine"],
        "title": "Healthcare & Medical Welfare Schemes",
        "description": "Key public healthcare, insurance, and medical assistance schemes:"
    },
    "pension": {
        "keywords": ["pension", "pensioners", "senior citizen", "old age", "social security", "atal pension", "jeevan pramaan", "shram yogi", "widow pension"],
        "title": "Pension & Social Security Schemes",
        "description": "Key national and state pension and social welfare schemes:"
    },
    "education": {
        "keywords": ["education", "scholarship", "scholarships", "student", "students", "fellowship", "cbse", "ugc", "academic bank", "higher education", "study"],
        "title": "Education & Scholarship Schemes",
        "description": "Key education, examination, and scholarship portals:"
    },
    "business": {
        "keywords": ["business", "startup", "startups", "msme", "udyam", "company", "trade licence", "gst registration", "fssai", "gem", "patent", "trademark"],
        "title": "Business, MSME & Startup Services",
        "description": "Key portals for business registration, licensing, and trade:"
    },
    "housing": {
        "keywords": ["housing", "pmay", "awas", "house", "rooftop solar", "solar rooftop", "shelter"],
        "title": "Housing & Energy Welfare Schemes",
        "description": "Key public housing and clean energy schemes:"
    },
    "women": {
        "keywords": ["women", "woman", "child welfare", "maternity", "anganwadi", "women commission", "girl child", "female"],
        "title": "Women & Child Welfare Schemes",
        "description": "Key welfare schemes and grievance redressal for women and children:"
    }
}


def is_category_scheme_discovery(text: str) -> Optional[str]:
    """
    Detect if the user is asking broadly for schemes/services available in a domain
    (e.g., 'which schemes for farmers', 'employment related services', 'schemes for students',
    'what health schemes are available').
    """
    q = text.lower().strip()
    discovery_triggers = [
        "which scheme", "which schemes", "what scheme", "what schemes",
        "schemes for", "scheme for", "services for", "service for",
        "schemes available", "services available", "list of schemes",
        "list of services", "related services", "related schemes",
        "welfare scheme", "welfare schemes", "government schemes",
        "government services", "available for", "options for"
    ]

    has_discovery_trigger = any(dt in q for dt in discovery_triggers)
    # Also support very short queries like "for farmers", "farmer schemes", "employment schemes"
    words = q.split()
    is_short = len(words) <= 5

    for cat_id, cfg in CATEGORY_SCHEME_GROUPS.items():
        has_cat_kw = any(re.search(rf"\b{re.escape(k)}\b", q) for k in cfg["keywords"])
        if has_cat_kw:
            if has_discovery_trigger:
                return cat_id
            if is_short and any(w in q for w in ["scheme", "schemes", "service", "services", "for", "related"]):
                return cat_id

    return None


def get_category_services(cat_id: str, services_catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Retrieve services in catalog matching the category definition."""
    cfg = CATEGORY_SCHEME_GROUPS.get(cat_id)
    if not cfg:
        return []
    keywords = cfg["keywords"]
    matched = []
    seen = set()
    for s in services_catalog:
        name = s.get("name", "")
        if name in seen:
            continue
        # Filter false positives
        if cat_id == "health" and name == "Soil Health Card":
            continue
        name_cat = (name + " " + (s.get("category") or "")).lower()
        if any(k in name_cat for k in keywords):
            seen.add(name)
            matched.append(s)
    return matched


def is_aspect_only_follow_up(query: str, detected_info: List[str]) -> Optional[str]:
    """
    Check if query is asking strictly for a specific aspect (e.g. 'fees', 'cost',
    'how much', 'documents', 'what are the documents') without asking for full steps.
    """
    q_clean = re.sub(r"[^\w\s]", "", query.lower()).strip()
    words = q_clean.split()
    if len(words) > 6:
        return None

    if "fees" in detected_info:
        fee_words = ["fee", "fees", "cost", "charges", "charge", "price", "how much", "rate", "payment"]
        if any(w in q_clean for w in fee_words) and not any(w in q_clean for w in ["apply", "procedure", "steps", "how to"]):
            return "fees"

    if "documents" in detected_info:
        doc_words = ["document", "documents", "docs", "proof", "proofs", "paperwork"]
        if any(w in q_clean for w in doc_words) and not any(w in q_clean for w in ["apply", "procedure", "steps", "how to"]):
            return "documents"

    if "processing_time" in detected_info:
        time_words = ["time", "processing time", "duration", "how long", "timeline", "days"]
        if any(w in q_clean for w in time_words) and not any(w in q_clean for w in ["apply", "procedure", "steps"]):
            return "processing_time"

    if "eligibility" in detected_info:
        elig_words = ["eligible", "eligibility", "who can apply", "age limit", "criteria"]
        if any(w in q_clean for w in elig_words) and not any(w in q_clean for w in ["apply", "procedure", "steps"]):
            return "eligibility"

    return None


INDIAN_STATES = {
    "delhi": "Delhi",
    "punjab": "Punjab",
    "haryana": "Haryana",
    "maharashtra": "Maharashtra",
    "karnataka": "Karnataka",
    "tamil nadu": "Tamil Nadu",
    "uttar pradesh": "Uttar Pradesh",
    "up": "Uttar Pradesh",
    "bihar": "Bihar",
    "rajasthan": "Rajasthan",
    "gujarat": "Gujarat",
    "west bengal": "West Bengal",
    "kerala": "Kerala",
    "telangana": "Telangana",
    "andhra pradesh": "Andhra Pradesh",
    "madhya pradesh": "Madhya Pradesh",
    "mp": "Madhya Pradesh",
    "odisha": "Odisha",
    "assam": "Assam",
    "jharkhand": "Jharkhand",
    "chhattisgarh": "Chhattisgarh",
    "uttarakhand": "Uttarakhand",
    "himachal pradesh": "Himachal Pradesh",
    "goa": "Goa",
    "chandigarh": "Chandigarh",
    "jammu and kashmir": "Jammu & Kashmir",
    "j&k": "Jammu & Kashmir"
}


def detect_state(text: str) -> Optional[str]:
    """Detect if the citizen explicitly mentioned an Indian state or Union Territory."""
    q = text.lower()
    for state_kw, state_formal in INDIAN_STATES.items():
        if re.search(rf"\b{re.escape(state_kw)}\b", q):
            return state_formal
    return None


def detect_aspect_only(query: str, detected_info: List[str]) -> Optional[str]:
    """
    Check if query is asking specifically for one aspect (fees, documents, eligibility, processing_time)
    without asking for procedural steps or whole application instructions.
    """
    if not detected_info:
        return None
    q_clean = re.sub(r"[^\w\s]", "", query.lower()).strip()
    # If user explicitly asks for steps / procedure, give full structured guide
    procedural = ["how to apply", "how do i apply", "steps to apply", "steps", "procedure", "process", "where to apply", "how can i get", "apply karna", "banwana", "karna hai"]
    if any(p in q_clean for p in procedural):
        return None

    if "fees" in detected_info:
        if any(w in q_clean for w in ["fee", "fees", "cost", "charge", "charges", "price", "how much", "rate", "रुपये", "फीस"]):
            return "fees"

    if "documents" in detected_info:
        if any(w in q_clean for w in ["document", "documents", "docs", "proof", "proofs", "paperwork", "दस्तावेज़", "कागजात"]):
            return "documents"

    if "processing_time" in detected_info:
        if any(w in q_clean for w in ["processing time", "how long", "timeline", "duration", "kitne din", "how many days"]):
            return "processing_time"

    if "eligibility" in detected_info:
        if any(w in q_clean for w in ["eligible", "eligibility", "who can apply", "age limit", "criteria", "who is eligible", "पात्रता"]):
            return "eligibility"

    return None


def check_action_documented(service: Dict[str, Any], requested_actions: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Check if the matched service record actually documents the action the user wants to perform
    (e.g., renewal, correction, duplicate/replacement), or if it is merely a first-time application record.
    Returns: (is_documented, undocumented_action_name_or_None)
    """
    if not requested_actions or not service:
        return True, None

    name_lower = (service.get("name") or "").lower()
    steps_lower = str(service.get("steps") or "").lower()
    docs_lower = str(service.get("documents_required") or "").lower()
    record_text = f"{name_lower} {steps_lower} {docs_lower}"

    for action in requested_actions:
        if action == "renewal":
            if not any(w in record_text for w in ["renew", "renewal", "re-issue", "reissue", "extend validity"]):
                return False, "renewal"

        elif action == "correction":
            if not any(w in record_text for w in ["correction", "correct", "update", "modify", "modification", "rectify", "name change", "dob", "reprint"]):
                return False, "correction"

        elif action == "replacement":
            if not any(w in record_text for w in ["duplicate", "replacement", "reprint", "lost", "damaged", "pvc"]):
                return False, "duplicate/replacement"

        elif action == "update":
            if not any(w in record_text for w in ["update", "change", "modify", "modification", "shift", "re-issue", "reissue"]):
                return False, "update"

        elif action == "status_check":
            if not any(w in record_text for w in ["status", "track", "tracking", "appointment"]):
                return False, "status tracking"

    return True, None


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
    5. Category & Scheme Discovery: handles domain-wide inquiries (farmers, employment,
       health, pension, education, women, business) by aggregating matching services.
    6. Guardrails for unsupported / out-of-scope / ambiguous queries.
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

    # 3. Check domain scheme discovery (e.g., 'which schemes for farmers', 'employment related services')
    cat_discovery = is_category_scheme_discovery(q_lower)
    if cat_discovery:
        cat_services = get_category_services(cat_discovery, services_catalog)
        if cat_services:
            cfg = CATEGORY_SCHEME_GROUPS[cat_discovery]
            return {
                "status": "category_discovery",
                "category_id": cat_discovery,
                "category_title": cfg["title"],
                "category_description": cfg["description"],
                "services": cat_services,
                "score": 5.0,
                "vector_score": 0.5
            }

    # 4. Extract entities present in query
    user_entities = set()
    for canonical, syn_list in CORE_SERVICE_SYNONYMS.items():
        if any(re.search(rf"\b{re.escape(syn)}\b", q_lower) for syn in syn_list):
            user_entities.add(canonical)

    # 5. Conversational follow-up resolution:
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
        is_follow_up_aspect = bool(detected_info)
        is_implicit_ref = any(re.search(rf"\b{w}\b", q_lower) for w in ["it", "this", "that", "same", "charges", "cost", "fee", "fees", "paper", "papers", "detail", "details", "documents", "procedure", "steps"])
        
        # Follow up only if user is actively inquiring about aspects/details of the current active service
        if is_follow_up_aspect or is_implicit_ref:
            aspect_only = is_aspect_only_follow_up(q_lower, detected_info)
            return {
                "status": "matched",
                "service": active_ctx_svc,
                "score": 5.0,
                "vector_score": 0.5,
                "detected_actions": detected_actions,
                "detected_info": detected_info,
                "aspect_only": aspect_only
            }

    # 6. Map query to service candidate scores
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
            significant_overlap = [w for w in overlap if w not in {"and", "or", "for", "the", "in", "of", "to", "card", "service", "registration", "application", "license", "licence", "permit", "certificate", "portal", "online"}]
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

        # Action congruence bonus & penalty
        if "new_application" in detected_actions:
            if any(w in name_lower for w in ["enrolment", "fresh", "application", "new", "learner"]):
                if not any(w in name_lower for w in ["update", "correction", "change", "duplicate", "renewal", "link", "transfer"]):
                    score += 5.0
            if any(w in name_lower for w in ["update", "correction", "change", "duplicate", "renewal", "link", "transfer"]):
                score -= 4.0

        if any(act in ["update", "correction"] for act in detected_actions):
            if any(w in name_lower for w in ["update", "correction", "change", "reprint", "modification", "re-issue", "reissue", "shift"]):
                score += 5.0
            if any(w in name_lower for w in ["application (fresh)", "fresh application"]):
                score -= 4.0
            elif name_lower.endswith("application") and not any(w in name_lower for w in ["update", "correction", "change", "reprint"]):
                score -= 3.0

        if "renewal" in detected_actions:
            if any(w in name_lower for w in ["renewal", "re-issue", "reissue"]):
                score += 5.0
            if any(w in name_lower for w in ["application (fresh)", "fresh application", "learner"]):
                score -= 4.0
            elif name_lower.endswith("application") and not any(w in name_lower for w in ["renewal", "re-issue"]):
                score -= 3.0

        if "replacement" in detected_actions:
            if any(w in name_lower for w in ["duplicate", "replacement", "reprint", "pvc"]):
                score += 5.0
            if any(w in name_lower for w in ["application (fresh)", "fresh application"]):
                score -= 4.0
            elif name_lower.endswith("application") and not any(w in name_lower for w in ["duplicate", "reprint", "replacement", "pvc"]):
                score -= 3.0

        if "status_check" in detected_actions:
            if any(w in name_lower for w in ["status", "track", "tracking", "appointment"]):
                score += 4.0

        if "grievance" in detected_actions:
            if any(w in name_lower for w in ["grievance", "complaint", "cyber crime", "consumer", "police", "fir", "redressal", "cpgrams"]):
                score += 5.0

        if "transfer" in detected_actions:
            if "transfer" in name_lower:
                score += 4.0

        if "linking" in detected_actions:
            if "link" in name_lower:
                score += 4.0

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

    detected_state = detect_state(clean_query)
    aspect_only = detect_aspect_only(clean_query, detected_info)

    if not scored_candidates:
        return {
            "status": "not_found",
            "decision": "unsupported",
            "service": None,
            "action": detected_actions,
            "detected_state": detected_state,
            "aspect_only": aspect_only,
            "message": "I am sorry, this information is not available in our database. May I help you with anything else?"
        }

    # Sort descending by score
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    best = scored_candidates[0]

    # Verify minimum confidence and prevent weak single-word match
    if best["score"] < 2.5 and best["vector_score"] < 0.35:
        return {
            "status": "not_found",
            "decision": "weak_match",
            "service": None,
            "action": detected_actions,
            "detected_state": detected_state,
            "aspect_only": aspect_only,
            "message": "I am sorry, this information is not available in our database. May I help you with anything else?"
        }

    # Check for potential ambiguity between top 2 candidates
    if len(scored_candidates) > 1:
        second = scored_candidates[1]
        if abs(best["score"] - second["score"]) < 0.5 and best["score"] < 4.0 and not user_entities:
            return {
                "status": "clarification_needed",
                "decision": "potential_match_requiring_clarification",
                "service": None,
                "action": detected_actions,
                "detected_state": detected_state,
                "aspect_only": aspect_only,
                "message": f"Did you mean **{best['name']}** or **{second['name']}**? Please specify so I can assist you with the exact procedure."
            }

    matched_svc = best["service"]
    is_doc, undoc_action = check_action_documented(matched_svc, detected_actions)

    return {
        "status": "matched",
        "decision": "strongly_supported",
        "service": matched_svc,
        "score": best["score"],
        "vector_score": best["vector_score"],
        "detected_actions": detected_actions,
        "detected_info": detected_info,
        "aspect_only": aspect_only,
        "detected_state": detected_state,
        "undocumented_action": undoc_action,
        "action_documented": is_doc
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

