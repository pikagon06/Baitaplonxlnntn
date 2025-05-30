import streamlit as st
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Setup device and load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json', encoding='utf-8') as json_data:
    intents = json.load(json_data)
data = torch.load("data.pth")

# Load mô hình
model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"]).to(device)
model.load_state_dict(data["model_state"])
model.eval()

# Các biến hỗ trợ
all_words = data['all_words']
tags = data['tags']
all_sentences = data['all_sentences']  # Dùng cho TF-IDF
bot_name = "HUMG 2"

# Hàm phản hồi sử dụng TF-IDF
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words, all_sentences).reshape(1, -1)  # SỬ DỤNG TF-IDF!
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    if probs[0][predicted.item()].item() > 0.65:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return "Tôi không hiểu bạn đang nói gì"

# Giao diện Streamlit
st.title("Chat with HUMG 2")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("You:"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = get_response(prompt)
    with st.chat_message("assistant"):
        st.markdown(f"{bot_name}: {response}")
    st.session_state.messages.append({"role": "assistant", "content": f"{bot_name}: {response}"})
