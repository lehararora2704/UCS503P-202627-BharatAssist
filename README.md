# BharatAssist

### AI-Powered Government Services Assistant

BharatAssist is a citizen-focused AI assistant designed to make Indian government services, schemes, procedures, and official information easier to discover and understand.

The project combines a modern web interface with AI-powered responses, Retrieval-Augmented Generation (RAG), government service information, and document simplification to provide users with clear and accessible guidance.

---

##  Project Objective

Government websites and official documents often contain complex procedures, technical language, and information spread across multiple sources.

BharatAssist aims to simplify this experience by providing:

- Easy access to government services
- Clear explanations of procedures and requirements
- AI-assisted question answering
- Retrieval of relevant government information
- Simplification of government documents
- A clean and accessible user interface

---

## Key Features

### Government Services

BharatAssist provides information about a growing collection of Indian government services, including:

- Service descriptions
- Eligibility information
- Required documents
- Application procedures
- Fees and related information
- Official sources

The current prototype includes approximately **20 government services**.

### AI-Powered Assistant

The application uses Google's Generative AI capabilities to provide natural-language assistance.

Users can ask questions in a conversational way and receive responses based on the available government service information.

### Retrieval-Augmented Generation (RAG)

BharatAssist uses a RAG-based approach to improve the relevance of generated answers.

The system:

1. Receives the user's query
2. Retrieves relevant information from the service knowledge base
3. Provides the retrieved context to the AI model
4. Generates a response grounded in the available information

### 📄 Document Guide

Users can provide government documents through:

- Pasted text
- PDF files
- DOCX files
- TXT files

The Document Guide processes the content and produces a simpler, easier-to-understand summary.

### Modern User Interface

The frontend has been redesigned to provide:

- Modern card-based layouts
- Improved navigation
- Responsive design
- Clear visual hierarchy
- Better forms and buttons
- Improved document upload experience
- Mobile-friendly layouts

---

## Project Structure

```text
BharatAssist/
│
├── assets/
│   └── Static assets and project resources
│
├── code/
│   └── Application source code
│
├── docs/
│   ├── ROADMAP.md
│   ├── architecture.md
│   ├── evaluation.md
│   ├── requirements.md
│   ├── testing.md
│   └── BharatAssist_Week1_Progress.pptx
│
├── journals/
│   └── Individual team member progress journals
│
├── project-proposal/
│   └── Project proposal documents
│
├── project-report-final/
│   └── Final project documentation
│
├── project-report-prototype-stage/
│   └── Prototype-stage documentation
│
├── .gitignore
├── Makefile
├── README.md
└── pyproject.toml
