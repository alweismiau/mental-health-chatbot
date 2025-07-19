import tensorflow as tf
import numpy as np
import pickle
import re
import emoji
# import tf_keras as k3
from tensorflow import keras as k3
import json 
# import requests
import pandas as pd
import nltk
import time
import torch
import os
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForCausalLM, TFRobertaModel, BitsAndBytesConfig

from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
login(hf_token)

nltk.download('stopwords')

# Setup
EMOTION_MODEL_PATH = "keioio/emotion-prediction-roberta/"
PROMPT_PATH = "src/mbti/prompting-mbti-eng.json"
DICTIONARY_PATH = "src/mbti/kamus_singkatan.csv"
KAMUS_PATH = "src/mbti/kamus_indonesia.txt"
CHATBOT_MODEL_PATH = "keioio/chatbot_english_llama_3_1_instruct_v3"

emotion_model = k3.models.load_model(EMOTION_MODEL_PATH +"emotion_model", custom_objects={"TFRobertaModel": TFRobertaModel})
with open(EMOTION_MODEL_PATH + "label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

emotion_tokenizer = AutoTokenizer.from_pretrained("w11wo/indonesian-roberta-base-sentiment-classifier")

dictionary = pd.read_csv(DICTIONARY_PATH, delimiter=";", header=None)
dictionary.columns = ['singkatan', 'arti']
dictionaries = dict(zip(dictionary['singkatan'], dictionary['arti']))

def load_prompts(PROMPT_PATH):
    with open(PROMPT_PATH, 'r') as file:
        return json.load(file)

# Preprocessing text for emotion recognition
stop_words = set(stopwords.words("indonesian")) - {"tidak", "jangan", "kurang"}
def preprocess(text):
    text = text.lower()
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'(.)\1{2,}', r'\1', text)  
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r'&[a-z]+;', '', text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    words = [dictionaries.get(word, word) for word in words]
    # words = [word for word in words if word not in stop_words]
    words = [word for word in words if len(word) > 2]

    return " ".join(words)

def predict_emotion(text):
    text = preprocess(text)
    if not text:
        return "netral"
    
    encoded = emotion_tokenizer(text, return_tensors="np", truncation=True, padding="max_length", max_length=128)
    input_ids = np.array(encoded["input_ids"])
    attention_mask = np.array(encoded["attention_mask"])
    prediction = emotion_model.predict([input_ids, attention_mask], verbose=0)
    predicted_class = np.argmax(prediction)
    return label_encoder.inverse_transform([predicted_class])[0]

# Model LLaMa 3
# Kuantisasi model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    CHATBOT_MODEL_PATH,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto" 
).to(device)
tokenizer = AutoTokenizer.from_pretrained(CHATBOT_MODEL_PATH)
# print("Inference Device:", device)

prompts = load_prompts(PROMPT_PATH)
MBTI_TYPES = {"ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP", "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"}

def chatbot_response(user_input, chat_histories, mbti_result):
    start_time = time.time()
    user_tokens = tokenizer.encode(user_input, add_special_tokens=False)
    short_input = len(user_tokens) < 2
    emotion = "netral" if short_input else predict_emotion(user_input)
    if not mbti_result:
        return emotion, None, "MBTI tidak tersedia.", 0

    try:
        mbti = mbti_result.upper()
    except AttributeError:
        return emotion, None, "MBTI tidak valid.", 0

    mbti_data = next((item for item in prompts if item["mbti"] == mbti), None)

    if not mbti_data:
        return emotion, None, "MBTI tidak ditemukan.", 0

    description = mbti_data.get("description", "")
    emotion_instruction = mbti_data.get(emotion.lower(), mbti_data.get("netral", ""))

    asked_mbti = None
    for mbti_type in MBTI_TYPES:
        if mbti_type.lower() in user_input.lower():
            asked_mbti = mbti_type
            break

    if not description or not emotion_instruction:
        return emotion, None, "Kombinasi MBTI atau emosi tidak ditemukan.", 0

    instruction = f"Respond to the {emotion.lower()} {mbti} user, {description.lower().replace('the user is', 'who is')} {emotion_instruction.lower().replace('the user is feeling ' + emotion.lower() + '.', '').strip()}"

    mbti_query_patterns = [
        "mbtiku", "mbti saya", "jenis mbti saya", "tipe kepribadianku", "tipe mbtiku",
        "tipe mbti saya", "saya termasuk mbti", "mbti saya itu"
    ]
    is_self_mbti_query = any(p in user_input.lower() for p in mbti_query_patterns)

    is_mbti_question = is_self_mbti_query or asked_mbti

    target_mbti = asked_mbti if asked_mbti else mbti
    if is_mbti_question:
        instruction = (
            f"User is asking for an explanation of the {target_mbti} MBTI type. "
            f"Explain in Bahasa Indonesia with correct grammar and no spelling mistakes. "
            f"Describe the personality traits of someone with the {target_mbti} type clearly and formally. "
            f"Use the MBTI name exactly as {target_mbti} — do not change or misspell it."
        )

    else:
        instruction = (
            f"Respond to the {emotion.lower()} {mbti} user, "
            f"{description.lower().replace('the user is', 'who is')} "
            f"{emotion_instruction.lower().replace('the user is feeling ' + emotion.lower() + '.', '').strip()}."
        )

    history_text = ""
    for u, r in chat_histories[-10:]:
        history_text += f"User: {u}\nBot: {r}\n"

    prompt = f"""<|start_header_id|>system<|end_header_id|>
    You are a mental health chatbot that provides supportive and empathetic responses to users in Bahasa Indonesia. 
    Your responses should always be written in fluent, natural, and correct Bahasa Indonesia, with no spelling errors or typos.
    Respond carefully and thoughtfully, ensuring that the answer is relevant, well-structured, and consistent with the user's MBTI type and emotional state.

    ### Instruction:
    {instruction}
    Make sure your response:
    - is written in **fluent, natural, and grammatically correct** Bahasa Indonesia.
    - **contains no spelling errors or typos**.
    - is empathetic, supportive, and contextually appropriate.
    - considers any relevant conversation history.

    <|start_header_id|>user<|end_header_id|>
    ### Conversation History:
    {history_text}

    ### User Input:
    {user_input}

    <|start_header_id|>assistant<|end_header_id|>
    ### Response:
    """

    user_tokens = tokenizer.encode(user_input, add_special_tokens=False)
    short_input = len(user_tokens) < 2
    max_tokens = 8 if short_input else 256
    # min_tokens = 4 if short_input else None

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            # min_new_tokens=min_tokens,
            do_sample=False,
            temperature=0.5,
            # top_p=0.85,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=False,
            num_beams=3,
            no_repeat_ngram_size=2
        )

    end_time = time.time()
    response_time = round(end_time - start_time, 2)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.split("### Response:")[-1].strip()  
    response = re.split(r'(?<=[.!?]) +', response)
    response = " ".join(response[:10])

    print("Instruction:", instruction)  

    return emotion, instruction, response, response_time

def format_chat_history(chat_histories):
    conversation = ""
    for item in chat_histories:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        user, bot = item
        conversation += f"👤 User: {user}\n🤖 Bot: {bot}\n"
    return conversation.strip()

def build_summary_prompt(chat_text):
    return f"""<|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are a chatbot tasked with summarizing conversations between a user and chatbot.
    
    ### Instruction:
    Create a summary of the following conversation between the user and the chatbot. Focus on:
    1. The main problem or topic discussed.
    2. The emotions expressed by the user and the chatbot's responses.
    3. Any advice or support provided by the chatbot.
    Use a natural and flowing style, writing in 3-7 sentences only. Avoid adding unnecessary details. Using Indonesian language.

    <|start_header_id|>user<|end_header_id|>
    ### Chat:
    {chat_text}

    <|start_header_id|>assistant<|end_header_id|>
    ### Summary:
    """

def summarize_chat(chat_histories):
    chat_text = format_chat_history(chat_histories)
    print("Formatted chat text:", chat_text)  # Debug
    prompt = build_summary_prompt(chat_text)
    print("Summary prompt:", prompt)  # Debug

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=False,
        num_beams=3,
        temperature=0.4,
        top_p=0.85,
        early_stopping=True,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Raw summary output:", summary)  
    return summary.split("### Summary:")[-1].strip()