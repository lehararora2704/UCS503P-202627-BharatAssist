"""
LLM wrapper for BharatAssist.
Uses Google Gemini API (google-genai SDK) for:
- Grounded civic assistant query answering
- Audio transcription (speech-to-text / listening path)
- Text-to-speech audio synthesis
- Document simplification
- General out-of-scope fallback answering
"""

import io
import os
import re
import wave
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env
_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(_ENV_PATH):
    load_dotenv(_ENV_PATH, override=True)
else:
    load_dotenv()

_client = None
_cached_key = None

# Primary and fallback model choices for high throughput & quota safety
PRIMARY_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash"
]
TTS_MODEL = "gemini-2.5-flash-preview-tts"


def _get_client():
    global _client, _cached_key
    if os.path.exists(_ENV_PATH):
        load_dotenv(_ENV_PATH, override=True)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    if _client is None or _cached_key != api_key:
        _client = genai.Client(api_key=api_key)
        _cached_key = api_key
    return _client


def is_llm_available() -> bool:
    """Check if Gemini API key is configured."""
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


def _call_gemini(contents, model_candidates=None, config=None):
    """
    Helper to call Gemini API with model fallback.
    Tries primary candidates in order in case of quota or model deprecations.
    """
    client = _get_client()
    if not client:
        raise RuntimeError("GEMINI_API_KEY is not configured in .env")

    models = model_candidates or PRIMARY_MODELS
    last_err = None
    for model_name in models:
        try:
            if config:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
            else:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents
                )
            return response
        except Exception as e:
            last_err = e
            # If 429 quota or 404 model not found, try next candidate
            err_str = str(e)
            if "429" in err_str or "404" in err_str:
                continue
            raise e

    raise last_err or RuntimeError("No Gemini models available")


# ============================================================
# PROMPT TEMPLATES & MULTILINGUAL SPECIFICATIONS
# ============================================================

SIMPLIFY_PROMPT = """You are simplifying an Indian government or legal document for a common citizen \
with no legal background. Rewrite the following text in plain, simple {language} (use English if unspecified). \
Preserve all factual details, eligibility rules, required documents, dates, fees, and steps exactly. \
Do not omit any requirement. Use short sentences and structured bullet points. \
Do not add a legal disclaimer yourself; the app displays one separately.

Document text:
---
{text}
---

Plain-language explanation:"""

LANGUAGE_SPECS = {
    "English": {
        "directive": (
            "The citizen has selected English.\n"
            "CRITICAL LANGUAGE RULE: You MUST write your entire response 100% in English.\n"
            "All headings, bullet points, numbers, steps, and tips MUST be in English only."
        ),
        "greeting_sample": "Namaste",
        "greeting_msg": "Hello! I am BharatAssist, your civic assistant for Indian government services. How may I assist you with government services or welfare schemes today?",
        "greeting_with_name": "Hello {name}! I am BharatAssist, your civic assistant. How may I assist you with Indian government services or welfare schemes today?",
        "not_found_msg": "I am sorry, this information is not available in our database. May I help you with anything else?",
        "sections": [
            "**Quick Summary**: 1-2 clear sentences in English explaining the service.",
            "**Step-by-Step Procedure**: Clear, numbered steps (1., 2., 3.) in English.",
            "**Required Documents**: Bulleted list of needed identity/address proofs in English.",
            "**Fees & Processing Time**: Exact cost and estimated days in English.",
            "**Helpful Tip**: Official portal link or practical advice in English."
        ]
    },
    "Hindi": {
        "directive": (
            "नागरिक ने हिन्दी भाषा का चयन किया है।\n"
            "मुख्य नियम: आपको अपना पूरा उत्तर 100% देवनागरी हिन्दी में ही देना होगा।\n"
            "कोष्ठक में बार-बार अंग्रेज़ी अनुवाद न लिखें।"
        ),
        "greeting_sample": "नमस्ते",
        "greeting_msg": "नमस्ते! मैं भारतअसिस्ट हूँ, आपका नागरिक सहायक। आज मैं सरकारी सेवाओं या कल्याणकारी योजनाओं में आपकी क्या सहायता कर सकता हूँ?",
        "greeting_with_name": "नमस्ते {name} जी! मैं भारतअसिस्ट हूँ, आपका नागरिक सहायक। आज मैं सरकारी सेवाओं या योजनाओं में आपकी क्या सहायता कर सकता हूँ?",
        "not_found_msg": "क्षमा करें, यह जानकारी हमारे डेटाबेस में उपलब्ध नहीं है। क्या मैं आपकी किसी अन्य सरकारी सेवा में सहायता कर सकता हूँ?",
        "sections": [
            "**संक्षिप्त विवरण**: सेवा के बारे में 1-2 स्पष्ट वाक्य।",
            "**आवेदन प्रक्रिया**: स्पष्ट क्रमांकित चरण (1., 2., 3.)।",
            "**आवश्यक दस्तावेज़**: आवश्यक पहचान और पते के प्रमाण की सूची।",
            "**शुल्क एवं समय सीमा**: अनुमानित लागत और कार्य दिवस।",
            "**महत्वपूर्ण सुझाव**: आधिकारिक पोर्टल लिंक या व्यावहारिक सलाह।"
        ]
    },
    "Punjabi": {
        "directive": (
            "ਨਾਗਰਿਕ ਨੇ ਪੰਜਾਬੀ ਭਾਸ਼ਾ ਦੀ ਚੋਣ ਕੀਤੀ ਹੈ।\n"
            "ਮਹੱਤਵਪੂਰਨ ਨਿਰਦੇਸ਼: ਤੁਹਾਨੂੰ ਆਪਣਾ ਪੂਰਾ ਜਵਾਬ 100% ਪੰਜਾਬੀ (ਗੁਰਮੁਖੀ ਲਿਪੀ) ਵਿੱਚ ਹੀ ਦੇਣਾ ਪਵੇਗਾ।\n"
            "ਬਰੈਕਟਾਂ ਵਿੱਚ ਵਾਰ-ਵਾਰ ਅੰਗਰੇਜ਼ੀ ਅਨੁਵਾਦ ਨਾ ਲਿਖੋ।"
        ),
        "greeting_sample": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਜੀ",
        "greeting_msg": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਭਾਰਤਅਸਿਸਟ ਹਾਂ, ਤੁਹਾਡਾ ਨਾਗਰਿਕ ਸਹਾਇਕ। ਅੱਜ ਮੈਂ ਸਰਕਾਰੀ ਸੇਵਾਵਾਂ ਜਾਂ ਭਲਾਈ ਸਕੀਮਾਂ ਵਿੱਚ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?",
        "greeting_with_name": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ {name} ਜੀ! ਮੈਂ ਭਾਰਤਅਸਿਸਟ ਹਾਂ। ਅੱਜ ਮੈਂ ਸਰਕਾਰੀ ਸੇਵਾਵਾਂ ਜਾਂ ਸਕੀਮਾਂ ਵਿੱਚ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?",
        "not_found_msg": "ਮਾਫ਼ ਕਰਨਾ, ਇਹ ਜਾਣਕਾਰੀ ਸਾਡੇ ਡੇਟਾਬੇਸ ਵਿੱਚ ਉਪਲਬਧ ਨਹੀਂ ਹੈ। ਕੀ ਮੈਂ ਕਿਸੇ ਹੋਰ ਸਰਕਾਰੀ ਸੇਵਾ ਵਿੱਚ ਤੁਹਾਡੀ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?",
        "sections": [
            "**ਸੰਖੇਪ ਜਾਣਕਾਰੀ**: ਸੇਵਾ ਬਾਰੇ 1-2 ਸਪਸ਼ਟ ਵਾਕ।",
            "**ਅਪਲਾਈ ਕਰਨ ਦਾ ਤਰੀਕਾ**: ਸਪਸ਼ਟ ਨੰਬਰਵਾਰ ਕਦਮ (1., 2., 3.)।",
            "**ਜ਼ਰੂਰੀ ਦਸਤਾਵੇਜ਼**: ਲੋੜੀਂਦੇ ਪਛਾਣ ਅਤੇ ਪਤੇ ਦੇ ਸਬੂਤਾਂ ਦੀ ਸੂਚੀ।",
            "**ਫੀਸ ਅਤੇ ਸਮਾਂ**: ਸਰਕਾਰੀ ਖਰਚਾ ਅਤੇ ਅੰਦਾਜ਼ਨ ਦਿਨ।",
            "**ਮਹੱਤਵਪੂਰਨ ਸੁਝਾਅ**: ਅਧਿਕਾਰਤ ਪੋਰਟਲ ਲਿੰਕ ਜਾਂ ਵਿਵਹਾਰਕ ਸਲਾਹ।"
        ]
    },
    "Bengali": {
        "directive": (
            "নাগরিক বাংলা ভাষা নির্বাচন করেছেন।\n"
            "প্রধান নিয়ম: আপনার সম্পূর্ণ উত্তর ১০০% বাংলা লিপিতে হওয়া আবশ্যক।\n"
            "বন্ধনীতে বারবার ইংরেজি অনুবাদ লিখবেন না।"
        ),
        "greeting_sample": "নমস্কার",
        "greeting_msg": "নমস্কার! আমি ভারতঅ্যাসিস্ট, আপনার নাগরিক সহায়ক। আজ আমি সরকারি পরিষেবা বা কল্যাণমূলক প্রকল্পে আপনাকে কীভাবে সাহায্য করতে পারি?",
        "greeting_with_name": "নমস্কার {name}! আমি ভারতঅ্যাসিস্ট। আজ আমি সরকারি পরিষেবা বা প্রকল্পে আপনাকে কীভাবে সাহায্য করতে পারি?",
        "not_found_msg": "দুঃখিত, এই তথ্যটি আমাদের ডেটাবেসে উপলব্ধ নেই। আমি কি আপনাকে অন্য কোনো সরকারি সেবায় সাহায্য করতে পারি?",
        "sections": [
            "**সংক্ষিপ্ত বিবরণ**: পরিষেবা সম্পর্কে ১-২টি স্পষ্ট বাক্য।",
            "**আবেদন প্রক্রিয়া**: স্পষ্ট ক্রমিক ধাপ (১., ২., ৩.)।",
            "**প্রয়োজনীয় কাগজপত্র**: প্রয়োজনীয় পরিচয় এবং ঠিকানার প্রমাণ।",
            "**ফি এবং সময়সীমা**: আনুমানিক খরচ এবং কর্মদিবস।",
            "**গুরুত্বপূর্ণ পরামর্শ**: অফিসিয়াল পোর্টাল লিঙ্ক বা ব্যবহারিক পরামর্শ।"
        ]
    },
    "Marathi": {
        "directive": (
            "नागरिकाने मराठी भाषा निवडली आहे।\n"
            "मुख्य नियम: तुमचे संपूर्ण उत्तर १००% मराठीमध्येच असणे आवश्यक आहे।"
        ),
        "greeting_sample": "नमस्कार",
        "greeting_msg": "नमस्कार! मी भारतअसिस्ट आहे, आपला नागरिक सहाय्यक. आज मी आपल्याला सरकारी सेवा किंवा योजनांमध्ये कशी मदत करू शकतो?",
        "greeting_with_name": "नमस्कार {name} जी! मी भारतअसिस्ट आहे. आज मी आपल्याला सरकारी सेवा किंवा योजनांमध्ये कशी मदत करू शकतो?",
        "not_found_msg": "क्षमस्व, ही माहिती आमच्या डेटाबेसमध्ये उपलब्ध नाही. मी तुम्हाला इतर कोणत्याही सरकारी सेवेसाठी मदत करू शकतो का?",
        "sections": [
            "**थोडक्यात माहिती**: सेवेबद्दल 1-2 स्पष्ट वाक्ये.",
            "**अर्ज प्रक्रिया**: स्पष्ट टप्पे (1., 2., 3.).",
            "**आवश्यक कागदपत्रे**: आवश्यक ओळख आणि पत्त्याचा पुरावा.",
            "**शुल्क आणि कालावधी**: अंदाजे खर्च आणि दिवस.",
            "**महत्त्वाची टीप**: अधिकृत पोर्टल लिंक किंवा सल्ला."
        ]
    },
    "Gujarati": {
        "directive": (
            "નાગરિકે ગુજરાતી ભાષા પસંદ કરી છે.\n"
            "મુખ્ય નિયમ: તમારો સંપૂર્ણ ઉત્તર ૧૦૦% ગુજરાતી લિપિમાં હોવો જોઈએ."
        ),
        "greeting_sample": "નમસ્તે",
        "greeting_msg": "નમસ્તે! હું ભારતઅસિસ્ટ છું, તમારો નાગરિક સહાયક. આજે હું તમને સરકારી સેવાઓ અથવા યોજનાઓમાં કેવી રીતે મદદ કરી શકું?",
        "greeting_with_name": "નમસ્તે {name}! હું ભારતઅસિસ્ટ છું. આજે હું તમને સરકારી સેવાઓ અથવા યોજનાઓમાં કેવી રીતે મદદ કરી શકું?",
        "not_found_msg": "માફ કરશો, આ માહિતી અમારા ડેટાબેઝમાં ઉપલબ્ધ નથી. શું હું તમને અન્ય કોઈ સરકારી સેવામાં મદદ કરી શકું?",
        "sections": [
            "**સંક્ષિપ્ત માહિતી**: સેવા વિશે 1-2 સ્પષ્ટ વાક્યો.",
            "**અરજી પ્રક્રિયા**: સ્પષ્ટ ક્રમબદ્ધ પગલાં (1., 2., 3.).",
            "**જરૂરી દસ્તાવેજો**: જરૂરી પુરાવાઓની યાદી.",
            "**ફી અને સમય**: અંદાજિત ખર્ચ અને દિવસો.",
            "**મહત્વપૂર્ણ સલાહ**: સત્તાવાર પોર્ટલ લિંક અથવા સલાહ."
        ]
    },
    "Tamil": {
        "directive": (
            "குடிமகன் தமிழ் மொழியைத் தேர்ந்தெடுத்துள்ளார்.\n"
            "முக்கிய விதி: உங்கள் முழு பதிலும் 100% தமிழ் எழுத்துக்களில் மட்டுமே இருக்க வேண்டும்."
        ),
        "greeting_sample": "வணக்கம்",
        "greeting_msg": "வணக்கம்! நான் பாரத்அசிஸ்ட், உங்கள் குடிமக்கள் உதவியாளர். அரசு சேவைகள் அல்லது நலத்திட்டங்களில் இன்று நான் உங்களுக்கு எவ்வாறு உதவ முடியும்?",
        "greeting_with_name": "வணக்கம் {name}! நான் பாரத்அசிஸ்ட். அரசு சேவைகள் அல்லது திட்டங்களில் இன்று நான் உங்களுக்கு எவ்வாறு உதவ முடியும்?",
        "not_found_msg": "மன்னிக்கவும், இந்த தகவல் எங்கள் தரவுத்தளத்தில் கிடைக்கவில்லை. வேறு ஏதேனும் அரசு சேவையில் நான் உங்களுக்கு உதவலாமா?",
        "sections": [
            "**சுருக்கமான தகவல்**: சேவை பற்றிய 1-2 தெளிவான வாக்கியங்கள்.",
            "**விண்ணப்பிக்கும் முறை**: தெளிவான படிநிலைகள் (1., 2., 3.).",
            "**தேவையான ஆவணங்கள்**: தேவையான ஆவணங்களின் பட்டியல்.",
            "**கட்டணம் மற்றும் நேரம்**: அரசு கட்டணம் மற்றும் ஆகும் காலம்.",
            "**முக்கிய குறிப்பு**: அதிகாரப்பூர்வ இணையதள விவரம்."
        ]
    },
    "Telugu": {
        "directive": (
            "పౌరుడు తెలుగు భాషను ఎంచుకున్నారు.\n"
            "ముఖ్య నియమం: మీ పూర్తి సమాధానం 100% తెలుగు లిపిలోనే ఉండాలి."
        ),
        "greeting_sample": "నమస్కారం",
        "greeting_msg": "నమస్కారం! నేను భారత్‌అసిస్ట్, మీ పౌర సహాయకుడిని. ఈరోజు ప్రభుత్వ సేవలు లేదా సంక్షేమ పథకాలలో నేను మీకు ఎలా సహాయపడగలను?",
        "greeting_with_name": "నమస్కారం {name} గారు! నేను భారత్‌అసిస్ట్. ఈరోజు ప్రభుత్వ సేవలు లేదా పథకాలలో నేను మీకు ఎలా సహాయపడగలను?",
        "not_found_msg": "క్షమించండి, ఈ సమాచారం మా డేటాబేస్‌లో అందుబాటులో లేదు. నేను మీకు మరేదైనా ప్రభుత్వ సేవలో సహాయం చేయగలనా?",
        "sections": [
            "**సంక్షిప్త సమాచారం**: సేవ గురించి 1-2 స్పష్టమైన వాక్యాలు.",
            "**దరఖాస్తు విధానం**: స్పష్టమైన దశలు (1., 2., 3.).",
            "**కావలసిన పత్రాలు**: అవసరమైన పత్రాల జాబితా.",
            "**ఫీజు మరియు సమయం**: అంచనా వ్యయం మరియు సమయం.",
            "**ముఖ్యమైన సలహా**: అధికారిక పోర్టల్ సమాచారం."
        ]
    },
    "Kannada": {
        "directive": (
            "ನಾಗರಿಕರು ಕನ್ನಡ ಭಾಷೆಯನ್ನು ಆಯ್ಕೆ ಮಾಡಿದ್ದಾರೆ.\n"
            "ಮುಖ್ಯ ನಿಯಮ: ನಿಮ್ಮ ಸಂಪೂರ್ಣ ಉತ್ತರವು 100% ಕನ್ನಡ ಲಿಪಿಯಲ್ಲೇ ಇರಬೇಕು."
        ),
        "greeting_sample": "ನಮಸ್ಕಾರ",
        "greeting_msg": "ನಮಸ್ಕಾರ! ನಾನು ಭಾರತ್‌ಅಸಿಸ್ಟ್, ನಿಮ್ಮ ನಾಗರಿಕ ಸಹಾಯಕ. ಇಂದು ಸರ್ಕಾರಿ ಸೇವೆಗಳು ಅಥವಾ ಕಲ್ಯಾಣ ಯೋಜನೆಗಳಲ್ಲಿ ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
        "greeting_with_name": "ನಮಸ್ಕಾರ {name} ಅವರೇ! ನಾನು ಭಾರತ್‌ಅಸಿಸ್ಟ್. ಇಂದು ಸರ್ಕಾರಿ ಸೇವೆಗಳು ಅಥವಾ ಯೋಜನೆಗಳಲ್ಲಿ ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
        "not_found_msg": "ಕ್ಷಮಿಸಿ, ಈ ಮಾಹಿತಿಯು ನಮ್ಮ ಡೇಟಾಬೇಸ್‌ನಲ್ಲಿ ಲಭ್ಯವಿಲ್ಲ. ನಾನು ನಿಮಗೆ ಬೇರೆ ಯಾವುದೇ ಸರ್ಕಾರಿ ಸೇವೆಯಲ್ಲಿ ಸಹಾಯ ಮಾಡಬಹುದೇ?",
        "sections": [
            "**ಸಂಕ್ಷಿಪ್ತ ಮಾಹಿತಿ**: ಸೇವೆಯ ಬಗ್ಗೆ 1-2 ಸ್ಪಷ್ಟ ವಾಕ್ಯಗಳು.",
            "**ಅರ್ಜಿ ಸಲ್ಲಿಸುವ ವಿಧಾನ**: ಹಂತ ಹಂತವಾದ ವಿವರಗಳು (1., 2., 3.).",
            "**ಅಗತ್ಯ ದಾಖಲೆಗಳು**: ಬೇಕಾಗುವ ದಾಖಲೆಗಳ ಪಟ್ಟಿ.",
            "**ಶುಲ್ಕ ಮತ್ತು ಕಾಲಾವಧಿ**: ಅಂದಾಜು ವೆಚ್ಚ ಮತ್ತು ದಿನಗಳು.",
            "**ಪ್ರಮುಖ ಸಲಹೆ**: ಅಧಿಕೃತ ಪೋರ್ಟಲ್ ಅಥವಾ ಸಲಹೆ."
        ]
    },
    "Malayalam": {
        "directive": (
            "പൗരൻ മലയാളം ഭാഷ തിരഞ്ഞെടുത്തു.\n"
            "പ്രധാന നിർദ്ദേശം: നിങ്ങളുടെ പൂർണ്ണമായ മറുപടി 100% മലയാളത്തിൽ മാത്രമായിരിക്കണം."
        ),
        "greeting_sample": "നമസ്കാരം",
        "greeting_msg": "നമസ്കാരം! ഞാൻ ഭാരത് അസിസ്റ്റ്, നിങ്ങളുടെ പൗര സഹായി. ഇന്ന് സർക്കാർ സേവനങ്ങളിലോ ക്ഷേമ പദ്ധതികളിലോ ഞാൻ നിങ്ങളെ എങ്ങനെ സഹായിക്കണം?",
        "greeting_with_name": "നമസ്കാരം {name}! ഞാൻ ഭാരത് അസിസ്റ്റ്. ഇന്ന് സർക്കാർ സേവനങ്ങളിലോ പദ്ധതികളിലോ ഞാൻ നിങ്ങളെ എങ്ങനെ സഹായിക്കണം?",
        "not_found_msg": "ക്ഷമിക്കണം, ഈ വിവരം ഞങ്ങളുടെ ഡാറ്റാബേസിൽ ലഭ്യമല്ല. മറ്റ് ഏതെങ്കിലും സർക്കാർ സേവനത്തിൽ ഞാൻ നിങ്ങളെ സഹായിക്കട്ടെയോ?",
        "sections": [
            "**സംക്ഷിപ്ത വിവരണം**: സേവനത്തെക്കുറിച്ചുള്ള 1-2 വ്യക്തമായ വാക്യങ്ങൾ.",
            "**അപേക്ഷാ നടപടിക്രമം**: ഘട്ടം ഘട്ടമായുള്ള വിവരങ്ങൾ (1., 2., 3.).",
            "**ആവശ്യമായ രേഖകൾ**: ആവശ്യമായ രേഖകളുടെ പട്ടിക.",
            "**ഫീസും സമയവും**: സർക്കാർ ഫീസും എടുക്കുന്ന സമയവും.",
            "**പ്രധാന നിർദ്ദേശം**: ഔദ്യോഗിക പോർട്ടൽ അല്ലെങ്കിൽ പ്രായോഗിക നിർദ്ദേശം."
        ]
    },
    "Urdu": {
        "directive": (
            "شہری نے اردو زبان منتخب کی ہے۔\n"
            "اہم اصول: آپ کو اپنا پورا جواب 100% اردو میں ہی دینا ہوگا۔"
        ),
        "greeting_sample": "السلام علیکم",
        "greeting_msg": "السلام علیکم! میں بھارت اسسٹ ہوں، آپ کا شہری مددگار۔ آج میں سرکاری خدمات یا فلاحی اسکیموں میں آپ کی کیا مدد کر سکتا ہوں؟",
        "greeting_with_name": "السلام علیکم {name}! میں بھارت اسسٹ ہوں۔ آج میں سرکاری خدمات یا اسکیموں میں آپ کی کیا مدد کر سکتا ہوں؟",
        "not_found_msg": "معذرت، یہ معلومات ہمارے ڈیٹا بیس میں دستیاب نہیں ہے۔ کیا میں کسی اور سرکاری سروس میں آپ کی مدد کر سکتا ہوں؟",
        "sections": [
            "**مختصر معلومات**: سروس کے بارے میں 1-2 واضح جملے۔",
            "**درخواست کا طریقہ کار**: واضح مراحل (1.، 2.، 3.)۔",
            "**مطلوبہ دستاویزات**: شناختی اور پتہ کے ضروری ثبوت۔",
            "**فیس اور درکار وقت**: سرکاری فیس اور متوقع دن۔",
            "**اہم مشورہ**: سرکاری پورٹل یا عملی مشورہ۔"
        ]
    },
    "Odia": {
        "directive": (
            "ନାଗରିକ ଓଡ଼ିଆ ଭାଷା ଚୟନ କରିଛନ୍ତି।\n"
            "ମୁଖ୍ୟ ନିୟମ: ଆପଣଙ୍କର ସମ୍ପୂର୍ଣ୍ଣ ଉତ୍ତର ୧୦୦% ଓଡ଼ିଆ ଲିପିରେ ହେବା ଆବଶ୍ୟକ।"
        ),
        "greeting_sample": "ନମସ୍କାର",
        "greeting_msg": "ନମସ୍କାର! ମୁଁ ଭାରତଆସିଷ୍ଟ, ଆପଣଙ୍କର ନାଗରିକ ସହାୟକ। ଆଜି ମୁଁ ସରକାରୀ ସେବା ବା କଲ୍ୟାଣକାରୀ ଯୋଜନାରେ ଆପଣଙ୍କୁ କିପରି ସାହାଯ୍ୟ କରିପାରିବି?",
        "greeting_with_name": "ନମସ୍କାର {name}! ମୁଁ ଭାରତଆସିଷ୍ଟ। ଆଜି ମୁଁ ସରକାରୀ ସେବା ବା ଯୋଜନାରେ ଆପଣଙ୍କୁ କିପରି ସାହାଯ୍ୟ କରିପାରିବି?",
        "not_found_msg": "କ୍ଷମା କରିବେ, ଏହି ସୂଚନା ଆମ ଡାଟାବେସରେ ଉପଲବ୍ଧ ନାହିଁ। ମୁଁ ଆପଣଙ୍କୁ ଅନ୍ୟ କୌଣସି ସରକାରୀ ସେବାରେ ସାହାଯ୍ୟ କରିପାରିବି କି?",
        "sections": [
            "**ସଂକ୍ଷିପ୍ତ ସୂଚନା**: ସେବା ବିଷୟରେ 1-2ଟି ସ୍ପଷ୍ਟ ବାକ୍ୟ।",
            "**ଆବେଦନ ପ୍ରକ୍ରିୟା**: ସ୍ପଷ୍ਟ ପଦକ୍ଷେପ (1., 2., 3.)।",
            "**ଆବଶ୍ୟକୀୟ କାଗଜପତ୍ର**: ଆବଶ୍ୟକ ପରିଚୟ ଏବଂ ଠିକଣା ପ୍ରମାଣ।",
            "**ଫି ଏବଂ ସମୟ**: ଆନୁମାନିକ ଖର୍ଚ୍ଚ ଏବଂ ସମୟ।",
            "**ମହତ୍ତ୍ୱପୂର୍ଣ୍ଣ ପରାମର୍ଶ**: ଅଫିସିଆଲ୍ ପୋର୍ଟାଲ୍ ବା ପରାମର୍ଶ।"
        ]
    }
}


# ============================================================
# GREETINGS & INTENT HELPERS
# ============================================================

GREETING_PATTERNS = [
    r"^(?:hi|hello|hey|hola|heya)(?:\s+(?:there|all|bot|assistant|bharatassist))?[!?. ]*$",
    r"^(?:namaste|namaskar|pranam|namaskaram|namaskara|vanakkam|sat\s*sri\s*akal|sasriakal|satshriakal|kiddan|kida|adaab|salam|assalam\s*o?\s*alaikum|as-salamu\s*alaykum)(?:\s+ji)?[!?. ]*$",
    r"^(?:good\s+(?:morning|afternoon|evening|day)|have\s+a\s+good\s+day)[!?. ]*$",
    r"^(?:who\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do|how\s+are\s+you|how\s+do\s+you\s+work|introduce\s+yourself|tell\s+me\s+about\s+yourself)[!?. ]*$",
    r"^(?:help|help\s+me|can\s+you\s+help\s+me|madad|sahayata)[!?. ]*$",
    r"^[\u0928\u092e\u0938\u094d\u0924\u0947\u092a\u094d\u0930\u0923\u093e\u092e\u0938\u0924\u093f\u0938\u094d\u0930\u0940\u0905\u0915\u093e\u0932!?. ]+$"
]


def is_conversational_greeting(text: str) -> bool:
    """
    Detect whether user text is purely a greeting, introduction, or pleasantry.
    """
    if not text:
        return False
    clean = text.strip().lower()
    # Normalize punctuation and spaces
    clean_normalized = re.sub(r"\s+", " ", clean)
    for pat in GREETING_PATTERNS:
        if re.search(pat, clean_normalized, re.IGNORECASE):
            return True
    return False


def get_not_found_response(language: str = "English") -> str:
    """
    Return standard polite decline when service/information is not available in database.
    Format: 'I am sorry, this information is not available in our database. May I help you with anything else?'
    """
    lang_key = language if language in LANGUAGE_SPECS else "English"
    return LANGUAGE_SPECS[lang_key]["not_found_msg"]


def get_greeting_response(language: str = "English", citizen_name: str = None) -> str:
    """
    Return warm localized greeting.
    """
    lang_key = language if language in LANGUAGE_SPECS else "English"
    spec = LANGUAGE_SPECS[lang_key]
    if citizen_name and citizen_name.strip():
        name = citizen_name.strip()
        template = spec.get("greeting_with_name", spec["greeting_msg"])
        return template.format(name=name)
    return spec["greeting_msg"]


# ============================================================
# DOCUMENT SIMPLIFICATION
# ============================================================

def simplify_document(text: str, language: str = "English") -> str:
    """Simplify government/legal text using Gemini."""
    if not is_llm_available():
        return "[LLM not configured] Set GEMINI_API_KEY in your .env file to enable AI simplification."

    prompt = SIMPLIFY_PROMPT.format(text=text[:12000], language=language or "English")
    response = _call_gemini(contents=prompt)
    return response.text


# ============================================================
# UNIFIED CONVERSATIONAL AI ASSISTANT
# ============================================================

def generate_assistant_response(
    question: str,
    context_chunks: list = None,
    language: str = "English",
    citizen_name: str = None
) -> str:
    """
    Unified, intelligent conversational assistant engine.
    - If context_chunks is empty:
      * If user sends a greeting/pleasantry: responds warmly without decline.
      * If user asks a question not in database: politely responds that the information
        is not available in our database and asks if it may help with anything else.
        NEVER answers out-of-database queries from Gemini general knowledge.
    - If context_chunks is provided:
      * Answers strictly using verified database context.
      * Strictly forbids hallucinating out-of-database services or general knowledge.
    """
    lang_key = language if language in LANGUAGE_SPECS else "English"
    spec = LANGUAGE_SPECS[lang_key]

    has_context = bool(context_chunks and len(context_chunks) > 0)

    # 1. No verified database context found
    if not has_context:
        if is_conversational_greeting(question):
            return get_greeting_response(language=lang_key, citizen_name=citizen_name)
        # Not a greeting, and no context in database: Polite decline
        return get_not_found_response(language=lang_key)

    # 2. Verified database context is available
    if not is_llm_available():
        return "<strong>Verified Information:</strong><br><br>" + "<br><br>".join(context_chunks)

    context_str = (
        "--- VERIFIED GOVERNMENT SERVICE CONTEXT ---\n"
        + "\n\n".join(context_chunks)
        + "\n-----------------------------------------"
    )

    sections_formatted = "\n".join([f"     * {s}" for s in spec["sections"]])

    prompt_instructions = f"""You are BharatAssist, an intelligent, warm, polite, and practical AI Civic Assistant for Indian citizens.

Core Guidelines:
1. STRICT DATABASE GROUNDING POLICY (CRITICAL):
   - You are strictly an assistant for the government services present in our database.
   - You must answer ONLY using the verified context from our database provided below.
   - If the citizen's question is NOT answered by the provided verified context, OR if they ask about something not present in the context:
     YOU MUST NOT invent, hallucinate, or answer from your general knowledge or Gemini training data!
     DO NOT bypass the database under any circumstances.
     Instead, reply ONLY with this exact sentence:
     "{spec['not_found_msg']}"

2. Structured Answer Pattern (When Context Answers the Question):
   - Organize your response clearly in structured sections:
{sections_formatted}
   - Use bold labels for key details so citizens can scan quickly.

3. Language Directives:
   - Selected Language: {lang_key}
   - {spec['directive']}
"""

    if citizen_name and citizen_name.strip():
        clean_name = citizen_name.strip()
        greeting_word = spec.get("greeting_sample", "Namaste")
        identity_instructions = f"""
4. Citizen Identity & Personal Greeting:
   - The citizen's verified name is "{clean_name}".
   - When greeting or addressing the citizen, address them warmly and politely by their name (e.g., "{greeting_word}, {clean_name}!").
"""
    else:
        identity_instructions = """
4. Citizen Identity:
   - The user is currently browsing as an unauthenticated guest.
   - Do NOT assume, invent, or use any personal name. Keep your tone polite and neutral.
"""

    prompt = f"""{prompt_instructions}{identity_instructions}

Context:
{context_str}

Citizen Message: {question}

BharatAssist:"""

    try:
        response = _call_gemini(contents=prompt)
        text_resp = (response.text or "").strip()
        if not text_resp:
            return get_not_found_response(language=lang_key)
        return text_resp
    except Exception as e:
        print("Gemini assistant generation error:", repr(e))
        if context_chunks:
            return "<strong>Verified Information:</strong><br><br>" + "<br><br>".join(context_chunks)
        return get_not_found_response(language=lang_key)


def answer_query(question: str, context_chunks: list, language: str = "English", citizen_name: str = None) -> str:
    return generate_assistant_response(question=question, context_chunks=context_chunks, language=language, citizen_name=citizen_name)


def answer_greeting(question: str, language: str = "English", citizen_name: str = None) -> str:
    return generate_assistant_response(question=question, context_chunks=[], language=language, citizen_name=citizen_name)


def answer_general_fallback(question: str, language: str = "English", citizen_name: str = None) -> str:
    return generate_assistant_response(question=question, context_chunks=[], language=language, citizen_name=citizen_name)


# ============================================================
# AUDIO: SPEECH-TO-TEXT (LISTENING PATH)
# ============================================================

def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/webm", language: str = "English") -> str:
    """
    Transcribe audio recorded from the user's microphone.
    Sends raw audio bytes directly to Gemini multimodal models.
    """
    if not is_llm_available():
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    if not audio_bytes or len(audio_bytes) == 0:
        raise ValueError("Audio recording is empty.")

    clean_mime = mime_type.split(";")[0].strip().lower() if mime_type else "audio/webm"
    if clean_mime not in {"audio/webm", "audio/wav", "audio/mp3", "audio/ogg", "audio/m4a"}:
        clean_mime = "audio/webm"

    part = types.Part.from_bytes(data=audio_bytes, mime_type=clean_mime)
    prompt = (
        f"Transcribe the spoken words in this audio exactly as spoken in {language or 'English'} or the original spoken language. "
        "Return ONLY the transcribed text. Do not add explanations, introductory phrases, or formatting."
    )

    response = _call_gemini(contents=[part, prompt])
    transcript = (response.text or "").strip()

    if transcript.lower() in {"[no speech detected]", "no speech detected", "silence"}:
        return ""
    return transcript


# ============================================================
# AUDIO: TEXT-TO-SPEECH (TTS PATH)
# ============================================================

LANGUAGE_TTS_CODES = {
    "Hindi": "hi",
    "English": "en",
    "Punjabi": "pa",
    "Bengali": "bn",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Urdu": "ur",
    "Odia": "hi",
}


def detect_script_language(text: str, fallback_language: str = "English") -> str:
    """Detect Indian language from unicode script."""
    if re.search(r"[\u0A00-\u0A7F]", text):
        return "Punjabi"
    if re.search(r"[\u0980-\u09FF]", text):
        return "Bengali"
    if re.search(r"[\u0A80-\u0AFF]", text):
        return "Gujarati"
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "Tamil"
    if re.search(r"[\u0C00-\u0C7F]", text):
        return "Telugu"
    if re.search(r"[\u0C80-\u0CFF]", text):
        return "Kannada"
    if re.search(r"[\u0D00-\u0D7F]", text):
        return "Malayalam"
    if re.search(r"[\u0600-\u06FF]", text):
        return "Urdu"
    if re.search(r"[\u0900-\u097F]", text):
        return "Marathi" if fallback_language == "Marathi" else "Hindi"
    return fallback_language or "English"


def _backup_neural_tts(text: str, language: str = "English") -> bytes:
    """High-reliability backup neural voice supporting all 12 Indian languages."""
    lang_code = LANGUAGE_TTS_CODES.get(language, "hi" if language != "English" else "en")
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()[:600]

    parts = re.split(r"([।?!.\n]+)", clean)
    chunks = []
    curr = ""
    for part in parts:
        if len(curr) + len(part) < 120:
            curr += part
        else:
            if curr.strip():
                chunks.append(curr.strip())
            curr = part
    if curr.strip():
        chunks.append(curr.strip())

    if not chunks:
        chunks = [clean[:120]]

    combined_audio = b""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for chunk in chunks:
        try:
            res = requests.get(
                "https://translate.google.com/translate_tts",
                params={"ie": "UTF-8", "q": chunk, "tl": lang_code, "client": "tw-ob"},
                headers=headers,
                timeout=5
            )
            if res.status_code == 200 and len(res.content) > 100:
                combined_audio += res.content
        except Exception as e:
            print("Backup TTS chunk error:", repr(e))

    if not combined_audio:
        raise RuntimeError("Speech synthesis failed across all neural providers.")
    return combined_audio


def text_to_speech(text: str, language: str = "English", return_mimetype: bool = False):
    """
    Convert text to speech audio bytes using Gemini TTS as primary neural voice,
    with an automatic backup neural voice for 100% reliability across all Indian languages.
    Returns (audio_bytes, mimetype) if return_mimetype is True, else audio_bytes.
    """
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Text for speech synthesis cannot be empty.")

    clean_text = clean_text[:800]
    effective_language = detect_script_language(clean_text, fallback_language=language)

    # 1. Primary: Gemini Multimodal TTS
    client = _get_client()
    if client:
        try:
            tts_prompt = f"Read the following text aloud clearly, naturally, and warmly in {effective_language}: {clean_text}"
            response = client.models.generate_content(
                model=TTS_MODEL,
                contents=tts_prompt,
                config={"response_modalities": ["AUDIO"]}
            )

            pcm_bytes = None
            if response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                                pcm_bytes = part.inline_data.data
                                break
                    if pcm_bytes:
                        break

            if pcm_bytes:
                wav_buf = io.BytesIO()
                with wave.open(wav_buf, "wb") as wav_file:
                    wav_file.setnchannels(1)       # Mono
                    wav_file.setsampwidth(2)       # 16-bit
                    wav_file.setframerate(24000)   # 24kHz
                    wav_file.writeframes(pcm_bytes)
                wav_bytes = wav_buf.getvalue()
                if return_mimetype:
                    return wav_bytes, "audio/wav"
                return wav_bytes
        except Exception as e:
            print("Gemini TTS warning, switching to backup neural voice:", repr(e))

    # 2. Backup: High-availability Indian Neural TTS
    mp3_bytes = _backup_neural_tts(clean_text, language=effective_language)
    if return_mimetype:
        return mp3_bytes, "audio/mpeg"
    return mp3_bytes
