import streamlit as st
import requests
import os
import numpy as np
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64
import spacy
import validators  
from dotenv import load_dotenv

def scrape_website(url):
    """Scrape the given news website for text and images."""
    response = requests.get(url)
    if response.status_code != 200:
        return None, None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract text content
    paragraphs = soup.find_all('p')
    text_content = ' '.join([p.get_text() for p in paragraphs])
    
    # Extract image URLs, excluding data URLs
    images = [img['src'] for img in soup.find_all('img') if 'src' in img.attrs and not img['src'].startswith('data:')]
    return text_content, images

def check_text_fact(text, api_key):
    """Use Google Fact Check API to verify the text."""
    endpoint = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": text,ry by truncating it to a reasonable length
        "key": api_key ' '.join(text.split()[:20])  # Use the first 20 words of the text
    }arams = {
    response = requests.get(endpoint, params=params)
        "key": api_key
    # Debugging: Log the full request and response
    print("API Request URL:", response.url)s=params)
    print("API Response Status Code:", response.status_code)
    print("API Response Content:", response.text)e
    print("API Request URL:", response.url)
    if response.status_code != 200::", response.status_code)
        return "Error accessing Fact Check API", None
    
    data = response.json()e != 200:
    if 'claims' in data and len(data['claims']) > 0:e
        claim = data['claims'][0]
        claim_review = claim.get('claimReview', [{}])[0]
        textual_rating = claim_review.get('textualRating', 'Unknown')
        review_text = claim_review.get('title', 'No additional details available')
        return textual_rating, review_textiew', [{}])[0]
    return "No fact-check available", None'textualRating', 'Unknown')
        review_text = claim_review.get('title', 'No additional details available')
def check_image_deepfake(image_url, model):
    """Predict if the given image is a deepfake using a pre-trained model."""
    response = requests.get(image_url, stream=True)
    if response.status_code != 200:on found", None
        return "Error fetching image"
    check_image_deepfake(image_url, model):
    try:redict if the given image is a deepfake using a pre-trained model.
        # Convert image to RGB format, stream=True)
        img = Image.open(io.BytesIO(response.content))
        img = img.convert('RGB')mage"
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)content))
        return "Deepfake" if prediction[0][0] > 0.5 else "Real"
    except Exception:ize((128, 128))
        return Invalid Image"mg) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
# Streamlit UItion = model.predict(img_array)
st.title("Fake News Detector")rediction[0][0] > 0.5 else "Real"
st.write("Enter a news article URL to check its authenticity.")
        return "Invalid Image"
url = st.text_input("Enter News URL:")
# Streamlit UI
load_dotenv()  # Load environment variables from .env file
st.write("Enter a news article URL to check its authenticity.")
# Debugging: Check if the API key is loaded
if not os.getenv("FACT_CHECK_API_KEY"):
    print("Error: FACT_CHECK_API_KEY is not set or loaded.")
else:dotenv()  # Load environment variables from .env file
    print("FACT_CHECK_API_KEY loaded successfully.")
# Debugging: Check if the API key is loaded
apikey = os.getenv("FACT_CHECK_API_KEY")  # Fetch API key from environment variable
    print("Error: FACT_CHECK_API_KEY is not set or loaded.")
if st.button("Check News"):
    if url and apikey:API_KEY loaded successfully.")
        # Validate the URL
        if not validators.url(url) or not (url.startswith("http://") or url.startswith("https://")):
            st.error("Invalid URL. Please enter a valid HTTP or HTTPS URL.")
        else:"Check News"):
            st.write("Scraping the website...")
            text, images = scrape_website(url)
            ot validators.url(url) or not (url.startswith("http://") or url.startswith("https://")):
            text_flag = False  # Initialize text_flag with a default value")
            :
            if text:("Scraping the website...")
                st.subheader("Extracted Text")
                st.write(text[:500] + "...")
                _flag = False  # Initialize text_flag with a default value
                # Extract key sentences for fact-checking
                try:
                    nlp = spacy.load('en_core_web_sm')
                    doc = nlp(text) + "...")
                    key_claims = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'EVENT']]
                    key_sentences = key_claims[0] if key_claims else ' '.join(text.split('.')[:3])
                except:
                    key_sentences = ' '.join(text.split('.')[:3])  # Extract first 3 sentences
                    doc = nlp(text)
                st.write("Checking text authenticity...").ents if ent.label_ in ['ORG', 'PERSON', 'EVENT']]
                text_result, review_details = check_text_fact(key_sentences, apikey)plit('.')[:3])
                
                if text_result == "No fact-check information found":
                    st.warning("No fact-check information found for the given text.")
                    text_flag = None
                elif text_result.startswith("Error"):act(key_sentences, apikey)
                    st.error("Error accessing the Fact Check API.")ling period
                    text_flag = Nonedetails.split()[2].rstrip('.') if review_details and len(review_details.split()) > 2 else ""  # Get third word
                else:    
                    st.write("Fact Check Result: ", text_result)
                    if review_details:
                        st.write("Supporting Evidence: ", review_details)eviewed by Google Claim Review yet.")
                    text_flag = text_result in {"Half true", "False", "Mostly", "Misrepresentation", "Pants", "Fake", "Incorrect", "Misleading", "No", "Out", "Unfounded", "Exaggerated", "Debunked"}
            else:
                st.write("No text found on the page.")e, "False", "Mostly", "Misrepresentation", "Pants", "Fake", "Incorrect", "Misleading", "No", "Out", "Unfounded", "Exaggerated", "Debunked"} or third_word in {"Half true", "False", "Mostly", "Misrepresentation", "Pants", "Fake", "Incorrect", "Misleading", "No", "Out", "Unfounded", "Exaggerated", "Debunked"}:
                text_result = "Unknown"
                review_details = None, text_result)
                text_flag = False  # Ensure text_flag is set even if no text is found_details)
            ext_flag = True
            if images:
                st.subheader("Extracted Images")ck Result: Independent assessment provided")
                model = load_model("deepfake_model.h5")
                deepfake_results = {}    
                lse:
                for img_url in images[:3]:  # Limit to 3 images for performancext_result)
                    result = check_image_deepfake(img_url, model)
                    deepfake_results[img_url] = resultporting Evidence: ", review_details)
                    st.image(img_url, caption=result, use_column_width=True)
                else:
                # Calculate fake scorete("No text found on the page.")
                fake_score = sum(1 for v in deepfake_results.values() if v == "Deepfake") / max(len(deepfake_results), 1)
            else:
                st.write("No images found.")Ensure text_flag is set even if no text is found
                fake_score = 0
            
            # Final Verdict Logic
            st.subheader("Final Verdict"))
            st.write("Combining text and image analysis...")
            # Adjust confidence calculation to prioritize text_flag
            if text_flag is True::3]:  # Limit to 3 images for performance
                combined_confidence = max(fake_score, 0.7)  # At least 70% if text is flagged as fake
            elif text_flag is None:   deepfake_results[img_url] = result
                combined_confidence = fake_score  # Use only fake_score if no fact-check is availablen=result, use_column_width=True)
            else:
                combined_confidence = fake_score * 0.5  # Reduce weight of fake_score if text is real    # Calculate fake score
            1 for v in deepfake_results.values() if v == "Deepfake") / max(len(deepfake_results), 1)
            # Display final verdict
            if text_flag is True and combined_confidence > 0.5:
                st.error(f"🚨 This news might be FAKE! Confidence: {combined_confidence * 100:.2f}%")
            elif text_flag is None and combined_confidence > 0.5:
                st.warning(f"⚠️ This news might be PARTIALLY FAKE. Confidence: {combined_confidence * 100:.2f}%")
            elif text_flag is False or combined_confidence <= 0.5:dict")
                st.success(f"✅ This news appears REAL. Confidence: {(1 - combined_confidence) * 100:.2f}%")
    else:ust confidence calculation to prioritize text_flag
        st.warning("Please enter a valid URL and ensure the API Key is set in the environment.")                st.warning(f"⚠️ This news might be PARTIALLY FAKE. Confidence: {combined_confidence * 100:.2f}%")
            elif text_flag is False or combined_confidence <= 0.5:
                st.success(f"✅ This news appears REAL. Confidence: {(1 - combined_confidence) * 100:.2f}%")
    else:
        st.warning("Please enter a valid URL and ensure the API Key is set in the environment.")
