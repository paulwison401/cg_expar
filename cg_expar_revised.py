
import os
from flask import Flask, request, jsonify
import pdfplumber
import spacy
import requests
from spacy.util import is_package
from io import BytesIO
import logging
from urllib.parse import urlparse

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Check if the language model is downloaded
if not is_package('en_core_web_sm'):
    try:
        spacy.cli.download('en_core_web_sm')
    except Exception as e:
        logging.error(f"Failed to download language model: {e}")
        raise

nlp = spacy.load("en_core_web_sm")

def extract_information(text):
    # Process the text with spaCy
    doc = nlp(text)

    # Initialize variables to store extracted information
    name = ""
    credit_score = ""
    open_accounts = ""
    accounts_ever_late = ""

    # Find the entity with the label "PERSON"
    for entity in doc.ents:
        if entity.label_ == "PERSON":
            name = entity.text
            break

    # Extract Credit Score
    for sent in doc.sents:
        if "FICO" in sent.text and "Score" in sent.text:
            credit_score = sent.text.split("Score")[-1].strip()
            break

    # Extract Open Accounts
    for sent in doc.sents:
        if "Open accounts" in sent.text:
            open_accounts = sent.text.split(":")[-1].strip()
            break

    # Extract Accounts Ever Late
    for sent in doc.sents:
        if "Accounts ever late" in sent.text:
            accounts_ever_late = sent.text.split(":")[-1].strip()
            break

    # Return the extracted information as a dictionary
    return {"name": name, "credit_score": credit_score, "open_accounts": open_accounts, "accounts_ever_late": accounts_ever_late}

@app.route("/extract_and_summarize", methods=["POST"])
def extract_and_summarize():
    try:
        json_data = request.get_json(force=True)
        pdf_url = json_data.get("pdf_url")
    except Exception as e:
        logging.error(f"Failed to get JSON data: {e}")
        return jsonify({"error": "Invalid JSON data"}), 400

    if not pdf_url:
        return jsonify({"error": "No url provided"}), 400

    # Validate URL
    try:
        parsed_url = urlparse(pdf_url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            return jsonify({"error": "Invalid URL provided"}), 400
    except Exception as e:
        logging.error(f"Invalid URL: {e}")
        return jsonify({"error": "Invalid URL provided"}), 400

    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        logging.error(f"HTTP Error: {errh}")
        return jsonify({"error": f"HTTP Error: {errh}"}), 400
    except requests.exceptions.ConnectionError as errc:
        logging.error(f"Error Connecting: {errc}")
        return jsonify({"error": f"Error Connecting: {errc}"}), 400
    except requests.exceptions.Timeout as errt:
        logging.error(f"Timeout Error: {errt}")
        return jsonify({"error": f"Timeout Error: {errt}"}), 400
    except requests.exceptions.RequestException as err:
        logging.error(f"Something went wrong: {err}")
        return jsonify({"error": f"Something went wrong: {err}"}), 400

    text = ""
    try:
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {e}")
        return jsonify({"error": f"Failed to extract text from PDF: {e}"}), 400

    info = extract_information(text)

    return jsonify(info)

if __name__ == "__main__":
    app.run(debug=False, host=os.getenv('HOST', '0.0.0.0'), port=int(os.getenv('PORT', 5000)))
