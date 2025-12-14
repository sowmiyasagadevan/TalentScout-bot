# TalentScout-bot
1)Project Overview

This project implements an AI-powered Hiring Assistant chatbot for TalentScout. The chatbot performs initial screening of candidates by collecting basic information and generating technical interview questions based on the candidate’s declared tech stack. The system is built using free, open-source pretrained models and provides a conversational interface using Streamlit.

2)Features

Greeting and onboarding

Collection of candidate details (name, email, phone, experience, role, location, tech stack)

Tech stack–based technical question generation

Adaptive question difficulty based on experience

Sentiment analysis for user responses

Multilingual support (English, Tamil, Hindi)

Secure handling of sensitive information

Graceful conversation termination

3)Models Used

FLAN-T5 Large – for technical question and answer generation

DistilBERT SST-2 – for sentiment analysis

MarianMT (Helsinki-NLP) – for language translation

All models are open-source and free to use.

4)Technologies Used

Python

Streamlit

HuggingFace Transformers

PyTorch

5)Installation

Install the required dependencies using:

pip install streamlit transformers torch sentencepiece

Running the Application

Run the chatbot locally using:

streamlit run app.py

6)Data Privacy

Candidate data is not stored on disk

Email addresses are masked

Phone numbers are protected

All data is handled only during the active session

7)Demo

The application can be demonstrated using a screen recording recorded with any screen capture tool and shared via a public link.

8)Conclusion

This chatbot demonstrates the use of pretrained language models, prompt engineering, and conversational AI design to build a practical hiring assistant for technical candidate screening.
