﻿# Chatbot Mental Health - Tutur Laras
## Project Overview
This project develops **a mental health chatbot web-based** for university students using LLaMA 3, RoBERTa, and a rule-based MBTI test. It offers empathetic, personalized responses, detects user emotions, and generates session summaries for counseling support. Testing showed **97% emotion detection accuracy and 87% BERTScore-F1**. Feedback from students and psychologists confirms the system is user-friendly and effective as an early mental health support tool.
## Model Training
### Fine-tune Emotion Recognition Model
- Model: RoBERTa (https://huggingface.co/w11wo/indonesian-roberta-base-sentiment-classifier)
- Dataset: 11.418 [data-emosi.csv](https://github.com/alweismiau/chatbot-kesehatan-mental/blob/main/dataset/dataset-emosi/data-emosi.csv) and 1.309 [kamus_singkatan.csv](https://github.com/alweismiau/chatbot-kesehatan-mental/blob/main/dataset/dataset-emosi/kamus_singkatan.csv)
- Evaluation: accuracy, precision, recall, and F1-score
- Output model: https://huggingface.co/keioio/emotion-prediction-roberta
### Fine-tune Chatbot Model
- Model: LLaMA 3.1 Instruct (https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- Dataset: 10.353 [data-kesehatan-mental.csv](https://github.com/alweismiau/chatbot-kesehatan-mental/blob/main/dataset/dataset-kesehatan-mental/data-kesehatan-mental.csv) 
- Evaluation: ROUGE and BERTScore
- Output model: https://huggingface.co/keioio/chatbot_english_llama_3_1_instruct_v3 
## Website App 
### Project Demonstration
[![Project Demo](https://img.youtube.com/vi/Ugh2ykR7GTA/0.jpg)](https://youtu.be/Ugh2ykR7GTA)
### Key Features
- User Authentication 
- MBTI Testi
- Chatbot Mental Health
- Emotion Recognition
- History and Summary Chat
### Tools
| **Layer**      | **Tools / Frameworks**                                      |
|----------------|-------------------------------------------------------------|
| Frontend       | ReactJS                                                    |
| Backend        | Flask  + Express.js (API)                   |
| Database       | MySQL                            |
| Model API      | HuggingFace Transformers  |

### Getting Started
Before running this code, please make sure **your device has at least 20GB of free storage**, as the total model size is over 16GB and using python version 3.11. 

```bash
# Clone the repository
git clone https://github.com/alweismiau/chatbot-kesehatan-mental
cd app

# Run backend
cd backend
npm install 
pip install -r requirements.txt
npm start

# Run frontend
cd frontend
npm install
npm run dev