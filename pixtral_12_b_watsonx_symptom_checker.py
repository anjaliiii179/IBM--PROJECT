# Pixtral 12B on IBM watsonx.ai — Agentic Health Symptom Checker
# Author: Generated notebook by ChatGPT
# Purpose: Jupyter-style Python notebook (linear .py) to run Pixtral 12B multimodal calls
# Note: Save as a .ipynb if you prefer a notebook interface. Replace placeholders for API_KEY and PROJECT_ID.

# -----------------------------
# 1) Install required packages (run in a notebook cell or terminal)
# -----------------------------
# %pip install -U ibm_watsonx_ai Pillow requests python-dotenv
# After installing packages restart the kernel if needed.

# -----------------------------
# 2) Imports & environment setup
# -----------------------------
import base64
import requests
import textwrap
import os
from PIL import Image
from io import BytesIO
from getpass import getpass

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# Optional: load from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -----------------------------
# 3) Configuration (replace with your values or set environment variables)
# -----------------------------
# NOTE: For security, do NOT commit API keys to source control. Use environment variables or IBM Secrets.
WATSONX_EU_APIKEY = os.getenv('WATSONX_EU_APIKEY') or getpass('Enter your WATSONX_EU_APIKEY: ')
WATSONX_EU_PROJECT_ID = os.getenv('WATSONX_EU_PROJECT_ID') or input('Enter your WATSONX_EU_PROJECT_ID: ')
WATSONX_URL = os.getenv('WATSONX_URL') or 'https://eu-gb.ml.cloud.ibm.com'  # adjust region if needed

credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_EU_APIKEY)

# -----------------------------
# 4) Helper: encode image (from URL or PIL image)
# -----------------------------
def encode_image_from_url(url: str) -> str:
    """Download image from URL and return base64-encoded JPEG bytes string."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    img_bytes = resp.content
    b64 = base64.b64encode(img_bytes).decode('utf-8')
    return b64


def encode_image_from_pil(img: Image.Image, fmt='JPEG') -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return b64

# -----------------------------
# 5) Build the multimodal prompt body for watsonx ModelInference.chat
# -----------------------------

def augment_api_request_body(user_query: str, image_b64: str):
    """Return messages list compatible with ModelInference.chat multimodal usage.
    This example asks the model to provide probable conditions and next steps in short form.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a careful, safety-conscious medical assistant. "
                        "Analyze the attached clinical image and answer the user's question in 2-4 sentences. "
                        "Provide: (1) most likely differential diagnosis, (2) immediate next-step advice, "
                        "and (3) a clear medical disclaimer telling the user to consult a professional. "
                        "Do NOT provide prescriptions or dose-specific instructions."
                    ) + "\nUser query: " + user_query
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                }
            ]
        }
    ]
    return messages

# -----------------------------
# 6) Instantiate Pixtral 12B model (and optionally a second model for cross-check)
# -----------------------------

model = ModelInference(
    model_id="mistralai/pixtral-12b",
    credentials=credentials,
    project_id=WATSONX_EU_PROJECT_ID,
    params={
        "max_tokens": 256,
        "temperature": 0.0
    }
)

# Optional verifier model (smaller/alternate) -- uncomment if you have quota
# verifier = ModelInference(
#     model_id="meta-llama/llama-3-2-11b-vision-instruct",
#     credentials=credentials,
#     project_id=WATSONX_EU_PROJECT_ID,
#     params={"max_tokens":256, "temperature":0.0}
# )

# -----------------------------
# 7) Simple RAG retrieval helper (fetch trusted text snippets)
# -----------------------------

def fetch_rag_snippets(query: str, sources: list = None, max_chars=2000):
    """A basic retriever that fetches content from a small set of trusted URLs.
    In production replace with an embedding-based vector DB (e.g., IBM Cloudant + OpenSearch, Pinecone, or Weaviate).
    """
    snippets = []
    # Example sources list (WHO, NHS, CDC pages) -- replace with actual endpoints you plan to use
    default_sources = [
        'https://www.who.int/news-room/fact-sheets',
        'https://www.nhs.uk/conditions',
        'https://www.cdc.gov/conditions'
    ]
    for url in (sources or default_sources):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                text = r.text.replace('\n', ' ')[:max_chars]
                snippets.append({'source': url, 'text': text})
        except Exception:
            continue
    # Filter snippets that mention the query (simple heuristic)
    related = [s for s in snippets if query.lower().split()[0] in s['text'].lower()] or snippets
    return related

# -----------------------------
# 8) Run multimodal inference (example function)
# -----------------------------

def analyze_image_with_pixtral(image_b64: str, user_query: str):
    messages = augment_api_request_body(user_query, image_b64)
    response = model.chat(messages=messages)
    raw = response.get('choices', [])[0]['message']['content']
    # Optionally call verifier and include cross-check
    # verifier_resp = verifier.chat(messages=messages)
    # verifier_raw = verifier_resp.get('choices', [])[0]['message']['content']
    return raw

# -----------------------------
# 9) Example: run on sample images (uncomment to run)
# -----------------------------
if __name__ == '__main__':
    print('Example run: Provide a public image URL of a visible skin condition.')
    sample_url = input('Enter image URL (or press Enter to skip example): ').strip()
    if sample_url:
        try:
            b64 = encode_image_from_url(sample_url)
            user_q = "What condition might this be? What are immediate next steps?"
            print('\nRunning Pixtral 12B inference...')
            out = analyze_image_with_pixtral(b64, user_q)
            print('\n<== Pixtral 12B Output ==>')
            print(textwrap.fill(out, width=100))

            # Fetch RAG snippets for user's query
            print('\nFetching RAG snippets (basic retriever)...')
            rag = fetch_rag_snippets(user_q)
            for s in rag[:3]:
                print(f"\n-- Source: {s['source']} --\n{s['text'][:500]}\n")

            print('\n*** IMPORTANT MEDICAL DISCLAIMER: This tool provides informational suggestions only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for diagnosis and treatment. ***')

        except Exception as e:
            print('Error during example run:', str(e))
    else:
        print('No example image provided. Notebook setup complete.')

# -----------------------------
# 10) Next steps & production recommendations (do not run directly)
# -----------------------------
# - Replace the simple fetch_rag_snippets with a vector DB + embedding pipeline for reliable RAG.
# - Add access control, logging, and rate-limiting for API usage.
# - Add unit tests and adversarial-safety filters to sanitize user text and images.
# - For medical apps: register appropriate disclaimers, implement human-in-loop review, and consult legal/regulatory guidance.
# - Consider model lifecycle warnings (check model availability and deprecation states in watsonx).

# End of file
