import streamlit as st
import requests
import os
import numpy as np
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from PIL import Image
import io
import spacy
import validators
from dotenv import load_dotenv

# ------------------------------ Scraper --------------------------------- #
def scrape_website(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None, None
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text_content = ' '.join([p.get_text() for p in paragraphs])
    images = [img['src'] for img in soup.find_all('img') if 'src' in img.attrs and not img['src'].startswith('data:')]
    return text_content, images

# ------------------------------ Fact Check ------------------------------ #
def extract_key_sentence(text):
    try:
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        claims = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'EVENT']]
        return claims[0] if claims else ' '.join(text.split('.')[:3])
    except:
        return ' '.join(text.split('.')[:3])

def check_text_fact(text, api_key):
    endpoint = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": ' '.join(text.split()[:20]),
        "key": api_key
    }
    response = requests.get(endpoint, params=params)

    if response.status_code != 200:
        return "Error accessing Fact Check API", None

    data = response.json()
    if 'claims' in data and len(data['claims']) > 0:
        claim_review = data['claims'][0].get('claimReview', [{}])[0]
        return claim_review.get('textualRating', 'Unknown'), claim_review.get('title', 'No additional details available')
    
    return "No fact-check information found", None

# --------------------------- Deepfake Checker --------------------------- #
def check_image_deepfake(image_url, model):
    response = requests.get(image_url, stream=True)
    if response.status_code != 200:
        return "Error fetching image"
    try:
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        return "Deepfake" if prediction[0][0] > 0.5 else "Real"
    except Exception:
        return "Invalid Image"

# ----------------------------- Streamlit UI ----------------------------- #
st.title("🧠 Fake News Detector")
st.write("Enter a news article URL to check for misinformation.")

url = st.text_input("Enter News URL:")

load_dotenv()
apikey = os.getenv("FACT_CHECK_API_KEY")
if not apikey:
    st.error("❌ API Key not found. Please ensure FACT_CHECK_API_KEY is set in environment.")

if st.button("Check News"):
    if url and apikey:
        if not validators.url(url) or not (url.startswith("http://") or url.startswith("https://")):
            st.error("⚠️ Invalid URL. Please enter a valid HTTP/HTTPS URL.")
        else:
            with st.spinner("🔍 Scraping the website..."):
                text, images = scrape_website(url)

            text_flag = False

            if text:
                st.subheader("📝 Extracted Text")
                st.write(text[:500] + "...")

                key_sentence = extract_key_sentence(text)

                with st.spinner("🔍 Verifying facts from Google Fact Check..."):
                    text_result, review_details = check_text_fact(key_sentence, apikey)

                if text_result == "No fact-check information found":
                    st.warning("ℹ️ No fact-check results found.")
                    text_flag = None
                elif text_result.startswith("Error"):
                    st.error("❌ Fact Check API access failed.")
                    text_flag = None
                else:
                    st.write("🧐 Fact Check Result: ", text_result)
                    if review_details:
                        st.write("📄 Supporting Evidence: ", review_details)
                    text_flag = text_result.lower() in {
                        "half true", "false", "mostly", "misrepresentation", "pants", 
                        "fake", "incorrect", "misleading", "no", "out", "unfounded", 
                        "exaggerated", "debunked"
                    }
            else:
                st.warning("No readable text found.")
                text_flag = False

            if images:
                st.subheader("🖼️ Extracted Images")
                model = load_model("deepfake_model.h5")
                deepfake_results = {}

                with st.spinner("🧠 Running deepfake analysis..."):
                    for img_url in images[:3]:
                        result = check_image_deepfake(img_url, model)
                        deepfake_results[img_url] = result
                        st.image(img_url, caption=result, use_container_width=True)

                fake_score = sum(1 for v in deepfake_results.values() if v == "Deepfake") / max(len(deepfake_results), 1)
            else:
                st.info("No images found.")
                fake_score = 0

            # --------------------- Final Verdict --------------------- #
            st.subheader("✅ Final Verdict")
            st.write("🤖 Combining image and text analysis...")

            if text_flag is True:
                combined_confidence = max(fake_score, 0.7)
            elif text_flag is None:
                combined_confidence = fake_score
            else:
                combined_confidence = fake_score * 0.5

            if text_flag is True and combined_confidence > 0.5:
                st.error(f"🚨 This news might be FAKE! Confidence: {combined_confidence * 100:.2f}%")
            elif text_flag is None and combined_confidence > 0.5:
                st.warning(f"⚠️ This news might be PARTIALLY FAKE. Confidence: {combined_confidence * 100:.2f}%")
            else:
                st.success(f"✅ This news appears REAL. Confidence: {(1 - combined_confidence) * 100:.2f}%")
    else:
        st.warning("Please enter a valid URL and ensure API Key is available.")
