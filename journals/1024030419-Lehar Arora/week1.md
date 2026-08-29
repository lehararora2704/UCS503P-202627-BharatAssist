# Week 01 — Lehar Arora

## Objective

Improve and extend the BharatAssist application by strengthening the frontend,
expanding the available government services, improving the RAG-based information
retrieval workflow, and preparing project documentation and presentation material.

## Work Completed

### 1. Frontend Modernization

- Redesigned the BharatAssist frontend with a cleaner, modern and professional UI.
- Improved the overall visual consistency across the application.
- Added a more structured layout for service pages and user interactions.
- Improved buttons, cards, forms, navigation elements and content sections.
- Added responsive styling for better usability across different screen sizes.
- Improved the Document Guide interface with clearer input, upload and result sections.
- Added improved visual feedback for document processing and file uploads.
- Applied a consistent BharatAssist visual language using a professional tricolour-inspired theme.

### 2. Government Service Expansion

- Expanded the service dataset significantly.
- Added approximately 20 government services to the application.
- Added service information, eligibility details, required documents and application
  guidance for the supported services.
- Improved the overall usefulness of the service discovery and information workflow.

### 3. RAG Improvements

- Worked on improving the Retrieval-Augmented Generation (RAG) workflow.
- Expanded the information available to the retrieval system through additional
  service data.
- Improved the retrieval context used for answering user queries.
- Worked on making generated answers more relevant to the selected government service.
- Improved source/context grounding so that responses can be connected to the
  underlying service information.

### 4. Document Guide

- Implemented and refined the document simplification interface.
- Added support for pasting government document text.
- Added document upload functionality for PDF, DOCX and TXT files.
- Added file validation and a 10 MB file-size restriction on the frontend.
- Added drag-and-drop file upload interaction.
- Added loading and error states for document processing.
- Improved the presentation of simplified document results.

### 5. Backend and Integration Work

- Integrated the frontend functionality with the existing Flask backend APIs.
- Connected the document simplification interface with the `/api/simplify` endpoint.
- Configured and tested the project virtual environment.
- Installed and configured required Python dependencies.
- Resolved environment and dependency issues encountered during development.

### 6. Testing and Debugging

- Tested the application locally using the Python virtual environment.
- Debugged missing Python package dependencies.
- Resolved Google Generative AI SDK dependency issues.
- Worked through ChromaDB/RAG dependency configuration.
- Tested frontend interactions and API communication.

### 7. Project Presentation

- Prepared a Week 1 progress presentation for demonstrating the development work.
- Structured the presentation around the project objective, technical work,
  frontend improvements, service expansion, RAG improvements and current progress.
- Included dedicated spaces for application UI screenshots to demonstrate the
  implemented interface.

## Problems / Challenges Identified

- Python virtual-environment configuration differed between development folders.
- Required packages were not initially available in the active virtual environment.
- Google Generative AI and ChromaDB dependencies required additional configuration.
- RAG functionality required additional data and retrieval improvements.
- The original frontend required significant visual and usability improvements.

## Current Status

The BharatAssist prototype has progressed from the initial project structure toward
a more complete and usable application. The frontend has been substantially
modernized, the number of supported government services has been expanded, and
the RAG/document-processing workflow has been improved.

## Next Steps

- Continue expanding and validating government service information.
- Further improve RAG retrieval quality and source grounding.
- Perform broader end-to-end testing.
- Improve error handling and edge-case handling.
- Evaluate response quality across different government-service queries.
- Continue refining the UI based on testing and feedback.
- Complete project documentation and final presentation material.
