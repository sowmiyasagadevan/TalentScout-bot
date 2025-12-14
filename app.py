import streamlit as st
import re
import os
from transformers import pipeline, MarianMTModel, MarianTokenizer

# Disable unnecessary backend warnings
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Streamlit page configuration
st.set_page_config(page_title="TalentScout Hiring Assistant", page_icon="🤖", layout="centered")
# Load pretrained QA model (offline)
@st.cache_resource
def load_models():
    qa = pipeline("text2text-generation", model="google/flan-t5-base")
    return qa
# Load translation models for multilingual support
@st.cache_resource
def load_translator(src, tgt):
    name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    tok = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name)
    return tok, model

qa_model = load_models()
translator_cache = {}
# Language selection
language = st.selectbox("🌐 Select Language", ["English", "Tamil", "Hindi"])
LANG = {"English": None, "Tamil": "ta", "Hindi": "hi"}
# Convert user input to English for processing
def to_english(text):
    if language == "English":
        return text
    key = (LANG[language], "en")
    if key not in translator_cache:
        translator_cache[key] = load_translator(LANG[language], "en")
    tok, model = translator_cache[key]
    out = model.generate(**tok(text, return_tensors="pt", padding=True))
    return tok.decode(out[0], skip_special_tokens=True)
# Convert bot response back to selected language
def from_english(text):
    if language == "English":
        return text
    key = ("en", LANG[language])
    if key not in translator_cache:
        translator_cache[key] = load_translator("en", LANG[language])
    tok, model = translator_cache[key]
    out = model.generate(**tok(text, return_tensors="pt", padding=True))
    return tok.decode(out[0], skip_special_tokens=True)
# Initialize session state
if "stage" not in st.session_state:
    st.session_state.stage = "greeting"
    st.session_state.chat = []
    st.session_state.data = {}
    st.session_state.questions = []
    st.session_state.index = 0
    st.session_state.score = 0
# Validation helpers
def valid_email(e): return re.match(r"[^@]+@[^@]+\.[^@]+", e)
def valid_phone(p): return re.match(r"^\+?\d{10,13}$", p)
# Mask sensitive data for privacy
def mask_email(e):
    n, d = e.split("@")
    return n[:2] + "*" * (len(n) - 2) + "@" + d

def mask_phone(p):
    return "*" * (len(p) - 4) + p[-4:]
# Generate technical questions based on tech stack
def generate_questions(stack):
    techs = [t.strip() for t in stack.split(",")]
    q = []
    for t in techs:
        q += [
            f"Explain the core concepts of {t}.",
            f"Write a simple program using {t}.",
            f"What are common mistakes in {t} and how do you avoid them?",
            f"Explain a real-world use case of {t}."
        ]
    return q
# UI header
st.markdown("<h1 style='text-align:center;'>🤖 TalentScout Hiring Assistant</h1>", unsafe_allow_html=True)
# Display chat history
for m in st.session_state.chat:
    st.chat_message(m["role"]).write(m["content"])
# Chat input
user = st.chat_input("Type your response...")

if user:
    # Store user message
    st.session_state.chat.append({"role": "user", "content": user})
    # Convert input to English for processing
    u = to_english(user)
    s = st.session_state.stage
    reply = ""
    # Exit keywords
    if u.lower() in ["exit", "bye", "quit"]:
        reply = "🙏 Thank you for your time. We will contact you soon."
        st.session_state.chat.append({"role": "assistant", "content": from_english(reply)})
        st.stop()
    # If user asks any question (contains '?'), answer using pretrained model
    if "?" in u:
        answer = qa_model(f"Answer this question clearly: {u}", max_length=200)[0]["generated_text"]
        reply = answer
    # Conversation flow stages
    elif s == "greeting":
        reply = "Hello 👋 Welcome to TalentScout!\nWhat is your full name?"
        st.session_state.stage = "name"

    elif s == "name":
        st.session_state.data["name"] = u
        reply = "Enter your email address:"
        st.session_state.stage = "email"

    elif s == "email":
        if not valid_email(u):
            reply = "Invalid email. Try again:"
        else:
            st.session_state.data["email"] = mask_email(u)
            reply = "Enter your phone number:"
            st.session_state.stage = "phone"

    elif s == "phone":
        if not valid_phone(u):
            reply = "Invalid phone number. Try again:"
        else:
            st.session_state.data["phone"] = mask_phone(u)
            reply = "Years of experience?"
            st.session_state.stage = "experience"

    elif s == "experience":
        st.session_state.data["experience"] = u
        reply = "Desired position?"
        st.session_state.stage = "position"

    elif s == "position":
        st.session_state.data["position"] = u
        reply = "Current location?"
        st.session_state.stage = "location"

    elif s == "location":
        st.session_state.data["location"] = u
        reply = (
            f"Details captured:\n"
            f"Email: {st.session_state.data['email']}\n"
            f"Phone: {st.session_state.data['phone']}\n\n"
            "Enter your tech stack (comma separated):"
        )
        st.session_state.stage = "tech"

    elif s == "tech":
        st.session_state.data["tech"] = u
        st.session_state.questions = generate_questions(u)
        st.session_state.index = 0
        reply = st.session_state.questions[0]
        st.session_state.stage = "qa"

    elif s == "qa":
        st.session_state.score += 3
        st.session_state.index += 1
        if st.session_state.index < len(st.session_state.questions):
            reply = st.session_state.questions[st.session_state.index]
        else:
            reply = (
                f"🎉 Interview completed!\n"
                f"Final Score: {st.session_state.score}/{len(st.session_state.questions)*5}\n\n"
                "Thank you for applying. You may ask any question now."
            )
            st.session_state.stage = "done"

    elif s == "done":
        reply = "Thank you 🙏 You may ask any further questions."
    # Store assistant reply
    st.session_state.chat.append({"role": "assistant", "content": from_english(reply)})
    st.rerun()
