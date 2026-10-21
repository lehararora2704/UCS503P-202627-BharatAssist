# 🇮🇳 BharatAssist — Intelligent Civic Guidance & Citizen Application Tracker

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-8E75B2?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FC6D26?style=for-the-badge)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-Academic%20Evaluation-orange?style=for-the-badge)]()

**BharatAssist** is an AI-powered, privacy-first civic guidance and application tracking platform designed for Indian citizens. It bridges the accessibility gap in government services by providing:
- Source-grounded AI guidance in **12 Indian languages** with Speech-to-Text and Text-to-Speech.
- Step-by-step procedure and document checklists for **200+ government services**.
- An **authenticated citizen portal** with localized profile preferences, personal application tracking, persistent interactive checklist progress, conversation thread history, and strict **Zero-LLM privacy isolation**.

---

## 📸 Key Features & Architecture

```
                          ┌─────────────────────────────┐
                          │   Citizen Browser / Mobile  │
                          └──────────────┬──────────────┘
                                         │ HTTPS / Session
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │          BharatAssist Flask Server           │
                 └──────┬───────────────┬───────────────┬───────┘
                        │               │               │
            Zero-LLM    │               │ RAG           │ Neural
         Privacy Shield │               │ Semantic      │ Audio
                        ▼               ▼               ▼
                 ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
                 │ Local SQLite│ │  ChromaDB   │ │Google Gemini│
                 │bharatassist │ │Vector Store │ │  2.5 Flash  │
                 │    .db      │ │(Embeddings) │ │ & Audio TTS │
                 └─────────────┘ └─────────────┘ └─────────────┘
```

### 1. 👤 Authenticated Citizen Portal & Profile
- **Personalized Details**: Full Name, Email, Mobile number, Preferred Language, State/UT, and District.
- **Cascading Location Selectors**: Static dataset of all Indian States and Districts (`static/india_states_districts.json`). Changing State instantly updates the District dropdown.
- **View & Edit Modes**: Clean card view with an inline edit form featuring validation (required name, email format, 10-digit Indian mobile number).
- **Security & Password Management**: In-modal password updating and strict session management.
- **Permanent Account Deletion (Danger Zone)**: Requires typing `"DELETE"` or entering the account password to permanently wipe all personal records, applications, checklist ticks, and chat history.

### 2. 📋 Saved Checklists ("My Applications")
- **One-Click Saving**: Save any government service directly from the Services Directory or assistant guidance.
- **Applications Dashboard**: Card grid displaying saved services, coverage, processing time, status badges (`Not started`, `In progress`, `Submitted`), and real-time completion progress (e.g., *"3 of 6 steps done (50%)"*).
- **Status Lifecycle**: Update application status directly from the card dropdown or delete applications.

### 3. ✅ Step-Level Interactive Checklist Progress
- **Stable Checkpoint IDs**: Every document requirement (`doc_0`, `doc_1`, ...) and procedure step (`step_0`, `step_1`, ...) has a deterministic ID.
- **Database Persistence**: Ticking a checkbox immediately syncs to SQLite (`user_step_progress` table) — progress survives browser reloads, logouts, and device switches.
- **Auto-Status Progression**: Ticking the first checkbox automatically advances the application from `Not started` to `In progress`.
- **Auto-Completion & "Submitted" Offer**: When all documents and steps are ticked (100%), a celebratory banner automatically appears allowing the user to mark the application as `"Submitted"` in a single click.

### 4. 💬 Persistent Chat History & Conversation Threads
- **Threaded Conversations**: Saves citizen interactions grouped by conversation threads in SQLite (`chat_conversations` and `chat_history`).
- **Thread Title Generation**: Automatically titled using the citizen's initial question.
- **Thread Review & Continuation**:
  - View full past dialogues in an interactive modal.
  - One-click **"Continue"** button loads past messages into the AI Assistant (`/assistant?c=<thread_id>`) for follow-up questions.
  - Delete unwanted threads permanently.
- **Zero Impact on AI Decision Core**: Conversation state is stored locally without modifying the underlying RAG pipeline.

### 5. 🤖 Multilingual Civic AI Assistant (`/assistant`)
- **12 Indian Languages**: UI localization and prompt engineering for English, Hindi (हिंदी), Punjabi (ਪੰਜਾਬੀ), Bengali (বাংলা), Marathi (मराठी), Gujarati (ગુજરાતી), Tamil (தமிழ்), Telugu (తెలుగు), Kannada (ಕನ್ನಡ), Malayalam (മലയാളം), Odia (ଓଡ଼ିଆ), and Urdu (اردو).
- **Voice-to-Text (STT)**: Dual-layer audio input using the browser Web Speech API with fallback to Gemini multimodal audio transcription (`/api/speech-to-text`).
- **Auto-Read (TTS)**: High-quality spoken responses powered by neural speech synthesis (`/api/text-to-speech`).
- **Strictly Grounded RAG**: Top-K chunk retrieval from ChromaDB over verified civic procedures, preventing hallucinations.

### 6. 🛡️ Zero-LLM Privacy Shield
- **PII Scrubbing**: Regex masking of Aadhaar numbers (12-digit UIDAI), PAN cards, phone numbers, and emails before sending queries to LLMs.
- **Credential Isolation**: Passwords, OTPs, mobile numbers, and personal application checklists are stored strictly in local SQLite and encrypted session cookies — **never** transmitted to public LLM endpoints.

---

## 🗄️ Database Architecture (`bharatassist.db`)

All tables are managed locally in SQLite with automatic schema migration on startup:

| Table Name | Description | Key Attributes |
|---|---|---|
| **`users`** | Citizen profiles and credentials | `id`, `phone`, `name`, `email`, `preferred_language`, `state`, `district`, `password_hash`, `otp`, `is_verified` |
| **`user_applications`**| Saved service checklists | `id`, `user_id`, `service_id`, `status` (`Not started`, `In progress`, `Submitted`), timestamps |
| **`user_step_progress`**| Individual document & step checkbox states | `id`, `application_id`, `item_id` (`doc_0`, `step_1`), `item_type`, `completed` (1/0), `updated_at` |
| **`chat_conversations`**| Conversation threads | `id` (token), `user_id`, `user_phone`, `title`, `created_at`, `updated_at` |
| **`chat_history`** | Message turns within threads | `id`, `conversation_id`, `user_id`, `user_phone`, `question`, `answer`, `language`, `sources_json`, `created_at` |
| **`services`** | Catalog of 200+ government services | `id`, `name`, `category`, `state`, `eligibility`, `documents_required`, `steps`, `fees`, `processing_time`, `source_url` |

---

## 📡 REST API Reference

### Profile & Account Management
| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/profile` | Render Citizen Profile page | Yes (Redirects) |
| `POST` | `/api/profile/update` | Update name, email, phone, language, state, district | Yes |
| `POST` | `/api/profile/change-password` | Update account password hash | Yes |
| `POST` | `/api/profile/delete-account` | Permanently delete account, applications, and chats | Yes |

### Applications & Checklist Tracking
| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/applications` | Render "My Applications" dashboard | Yes (Redirects) |
| `GET` | `/applications/<id>` | Render interactive step-by-step checklist | Yes (Redirects) |
| `POST` | `/api/applications/save` | Save a service checklist to citizen account | Yes |
| `POST` | `/api/applications/<id>/status` | Update status (`Not started`/`In progress`/`Submitted`) | Yes |
| `DELETE`| `/api/applications/<id>` | Remove saved application and associated step progress | Yes |
| `POST` | `/api/applications/<id>/toggle-step` | Toggle checkbox for document/step & recalculate progress | Yes |

### Chat History & Assistant
| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/chat-history` | Render Chat History page with past threads | Yes (Redirects) |
| `GET` | `/api/conversations` | List conversation threads for logged-in citizen | Yes |
| `GET` | `/api/conversations/<id>` | Get all message turns for a conversation thread | Yes |
| `DELETE`| `/api/conversations/<id>` | Delete conversation thread and its turns | Yes |
| `POST` | `/api/assistant` | Multilingual civic RAG chat (links to thread) | Optional |
| `POST` | `/api/speech-to-text` | Multimodal audio voice input transcription | No |
| `POST` | `/api/text-to-speech` | Speech synthesis for assistant responses | No |

### Authentication & Core
| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/login` | Dual-tab citizen login/registration page | No |
| `POST` | `/api/auth/login-password` | Mobile + Password login | No |
| `POST` | `/api/auth/send-otp` | Dispatch login OTP | No |
| `POST` | `/api/auth/verify-otp` | Verify login OTP | No |
| `POST` | `/api/auth/register-send-otp` | Initiate registration with mandatory name & mobile | No |
| `POST` | `/api/auth/register-verify` | Verify registration OTP & set password hash | No |
| `GET` | `/logout` | Flush private session and redirect | Optional |
| `GET` | `/api/health` | System and SQLite health check | No |

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.11+**
- A **Google Gemini API Key** (from [Google AI Studio](https://aistudio.google.com/))

### 2. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/your-username/bharatassist.git
cd bharatassist

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
FLASK_SECRET_KEY=bharatassist-citizen-privacy-shield-secret-2026
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 4. Database Setup & Seeding
```bash
# Populate database with 200+ civic services and initial schema
python seed_data.py
```

### 5. Run the Application
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

## 🔑 Demo Account Credentials (For Immediate Testing)

A pre-configured citizen account is available with sample applications, checklist ticks, and chat history:
- **URL**: [http://127.0.0.1:5000/login](http://127.0.0.1:5000/login)
- **Mobile Number**: `9876543210`
- **Password**: `Citizen@123`

*(You can also register a new account or log in via OTP — the 6-digit verification code is displayed on-screen for testing).*

---

## 🧪 Testing

Execute the automated integration test suite covering profile updates, application tracking, checklist step toggling, chat history threading, and permanent account deletion:

```bash
# Run User Profile & Applications integration suite
python -m unittest tests.test_user_profile_and_applications -v

# Run full project endpoint tests
python -m unittest tests.test_all_endpoints -v
```

---

## 📂 Project Directory Structure

```
BHARATASSIST/
│
├── app.py                     # Main Flask Application & API Routes
├── bharatassist.db            # SQLite Database (users, applications, chat, services)
├── seed_data.py               # Database initialization and seeding script
├── requirements.txt           # Project dependencies
├── .env                       # Environment configuration
│
├── static/
│   ├── style.css              # Custom styles & design tokens
│   └── india_states_districts.json # Indian states & districts mapping
│
├── templates/
│   ├── base.html              # Base layout with navbar & user dropdown
│   ├── index.html             # Homepage / Hero portal
│   ├── login.html             # Citizen sign-in & registration
│   ├── profile.html           # Citizen Profile (view, edit, delete account)
│   ├── applications.html      # "My Applications" dashboard
│   ├── application_checklist.html # Step-by-step checklist tracker
│   ├── chat_history.html      # Past AI conversation threads
│   ├── assistant.html         # Multilingual civic AI assistant
│   ├── services.html          # Government services directory
│   ├── service_details.html   # Service details with "Save to My Applications"
│   └── simplify.html          # Plain-language document simplifier
│
├── utils/
│   ├── llm.py                 # Gemini model calls, multilingual prompts & TTS
│   ├── rag.py                 # ChromaDB vector retrieval pipeline
│   ├── redact.py              # Zero-LLM PII regex redactor
│   ├── matcher.py             # Rule-based service matching & grounding
│   └── metrics.py             # Civic grounding and evaluation metrics
│
└── tests/
    ├── test_user_profile_and_applications.py # Profile, checklist, & chat tests
    └── test_all_endpoints.py                 # Full suite endpoint tests
```

---

## 📄 Academic Integrity & Project Credits
Developed as part of the **UCS503: Software Engineering** curriculum. All government service guidelines and procedures are grounded in official state and central portal documentation.
