# Resume Tailoring App

This project provides a Streamlit-based web interface that takes a user’s resume (PDF or DOCX) and a job description (plain text), then uses Ollama (Llama 3.2 3b) to rewrite the bullet points under “Work Experience” and “Projects” so they align more closely with the job requirements—preserving all original formatting except for the adjusted bullets.

## Project Structure
.
├── app.py # Streamlit frontend

├── tailor.py # Main logic to tie everything together

├── llm_utils.py # Ollama (Llama 3.2 3b) setup and bullet-alignment functions

├── bullet_utils.py # Helpers for extracting/replacing sections & bullets

├── resume_loader.py # Helpers for loading PDF/DOCX text

├── requirements.txt # Python dependencies

└── README.md # This file


## Prerequisites

1. **Python 3.8+**  
2. **Ollama** installed and running locally, with the `llama3.2:3b` model downloaded.  
   - Follow the instructions at https://ollama.com/docs/installation to install Ollama.
   - After installing, run `ollama pull llama3.2:3b` (or equivalent) so that the model is available.

3. **pip** (Python package installer).

## Installation

1. Clone or download this repository to your local machine:

   ```bash
   https://github.com/Devarshi07/Resume_Tailor.git
   cd resume-tailor

python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

pip install --upgrade pip
pip install -r requirements.txt


## Ensure Ollama is running locally
ollama serve
streamlit run app.py
