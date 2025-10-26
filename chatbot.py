import streamlit as st
import joblib
from scipy.sparse import hstack, csr_matrix
import numpy as np
import re, unicodedata

# Load Model & Vectorizer
model = joblib.load("spam_classifier_model.joblib")
word_vec = joblib.load("word_vectorizer.joblib")
char_vec = joblib.load("char_vectorizer.joblib")
label_encoder = joblib.load("label_encoder.joblib")

def normalize_text(s: str) -> str:
    s = unicodedata.normalize('NFKC', str(s))
    s = s.replace("’", "'").replace("‘", "'").replace("`","'")
    s = s.replace("\u2013", "-").replace("\u2014","-")
    s = re.sub(r'[^\x00-\x7F]+', ' ', s)
    return s.lower().strip()

money_re = re.compile(r'(\$|₱|£|€)\s?\d+[\d,]*')
num_re = re.compile(r'\b\d+(\.\d+)?\b')

def force_money_tokens(s):
    s = money_re.sub(' <MONEY> ', s)
    s = num_re.sub(' <NUM> ', s)
    return s

def extract_meta(series):
    rows = []
    for t in series:
        lc = str(t).lower()
        rows.append([
            int('<money>' in lc),
            int('http' in lc or 'www.' in lc),
            lc.count('!'),
            len(lc.split()),
            int('congrat' in lc),
            int('free' in lc),
            int('win' in lc or 'winner' in lc),
        ])
    return np.array(rows)

def predict_text(text):
    # Preprocess like in training
    t = normalize_text(text)
    t = force_money_tokens(t)
    
    # Extract features for ML model
    Xw = word_vec.transform([t])
    Xc = char_vec.transform([t])
    Xm = csr_matrix(extract_meta([t]))
    X_comb = hstack([Xw, Xc, Xm])
    
    # ML prediction
    ml_pred_num = model.predict(X_comb)[0]
    ml_pred = label_encoder.inverse_transform([ml_pred_num])[0].lower()
    
    # Heuristic rules
    spam_keywords = ['win', 'winner', 'prize', 'reward', 'claim', 'offer', 'free', 'urgent', 
                     'selected', 'brand new', 'click', 'limited', 'gift', 'survey', 
                     'lottery', 'credit card', 'pre-approved', 'discount', 'deal',
                     'money', 'earn', 'opportunity', 'verify', 'account', 'update']
    positive_keywords = ['accepted', 'hired', 'enrolled', 'passed', 'approved']
    
    has_money = '<money>' in t
    has_link = 'http' in t or 'www.' in t
    num_exclam = t.count('!')
    spam_hits = sum(1 for k in spam_keywords if k in t)
    
    if any(p in t for p in positive_keywords):
        return 'ham'
    if spam_hits >= 2 or has_link or has_money or num_exclam >= 2:
        return 'spam'
    if spam_hits >= 1 and (has_link or has_money or num_exclam > 0):
        return 'spam'
    
    return ml_pred

# Session State Initialization
if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0
if "last_input" not in st.session_state:
    st.session_state.last_input = ""

# Streamlit Page Config
st.set_page_config(
    page_title="Spam Detector Chatbot",
    page_icon="📧",
    layout="centered"
)

# CSS
st.markdown("""
<style>
header {background: transparent !important;}
footer {visibility: hidden;}
.stApp { 
    background-color: #FAF9EE !important;  /* main solid pastel yellow */
}
.block-container { padding-top: 0 !important; padding-bottom: 1rem !important; background-color: rgba(0,0,0,0) !important; }
div[data-testid="stVerticalBlock"] div:empty {display: none !important;}
div[data-testid="stVerticalBlock"] > div:first-child {margin-top:0 !important; padding-top:0 !important;}

/* Buttons */
div[data-testid="stButton"] button {
    background-color: #99A799;
    color: #ffffff;
    border: none;
    border-radius: 25px;
    font-family: 'Poppins', sans-serif;
    font-weight: bold;
    font-size: 1em;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    height: 38px;
    padding: 10px 22px;
    display: flex;
    align-items: center;
    justify-content: center;
}
div[data-testid="stButton"] button:hover {
    background-color: white;
    color: #99A799 !important;
    border: 2px solid #99A799 !important;
    box-shadow: 0 3px 8px rgba(0,0,0,0.1) !important;
}

/* Chat */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    height: 85vh;
    width: 100%;
    max-width: 700px;
    margin: auto;
    background: rgba(255,255,255,0.0);
    border-radius: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    padding: 20px;
}
.chat-history { flex-grow: 1; overflow-y: auto; max-height: 70vh; margin-bottom: 15px; padding-right: 5px; }
.user-msg {
    background-color: #F2DDC1;
    padding: 14px 20px;
    border-radius: 25px 25px 8px 25px;
    margin-bottom: 12px;
    text-align: right;
    width: fit-content;
    margin-left: auto;
    color: #4a4a4a;
    font-family: 'Poppins', sans-serif;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    transition: transform 0.2s ease, box-shadow 0.3s ease;
}
.user-msg:hover { transform: translateY(-2px); box-shadow: 0 3px 8px rgba(0,0,0,0.08); }
.bot-msg {
    background-color: #D3E4CD;
    padding: 14px 20px;
    border-radius: 25px 25px 25px 8px;
    margin-bottom: 12px;
    text-align: left;
    width: fit-content;
    margin-right: auto;
    color: #4a4a4a;
    font-family: 'Poppins', sans-serif;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    transition: transform 0.2s ease, box-shadow 0.3s ease;
}
.bot-msg:hover { transform: translateY(-2px); box-shadow: 0 3px 8px rgba(0,0,0,0.08); }

/* Input */
.input-area { display: flex; align-items: center; justify-content: center; gap: 10px; width: 100%; position: sticky; bottom: 0; background-color: #FAF9EE; padding: 10px 0; border-top: 1px solid #DCCFC0; margin-top: 10px; }
input[type="text"] {
    flex: 1;
    background-color: rgba(255,255,255,0.0) !important;
    border: 2px solid #DCCFC0 !important;
    border-radius: 25px;
    padding: 12px 16px !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 1em;
    color: #4a4a4a;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
}
input[type="text"]:focus {
    outline: none !important;
    border-color: #99A799 !important; 
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important; 
    background-color: rgba(255,255,255,0.0) !important;
}

div[data-baseweb="input"],
div[data-baseweb="input"]:focus-within,
div[data-baseweb="input"] > div {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}
.sentiment-gray { color: #7a7a7a; font-size: 0.85em; margin-top: 5px; display: block; }

/* Header */
.chat-header {
    display: flex; 
    justify-content: center; 
    align-items: center; 
    flex-direction: column;
    text-align: center; 
    padding: 50px 35px; 
    margin: 60px auto 25px auto; 
    width: 100%; 
    max-width: 750px;
    border-radius: 80px 80px 10px 10px; 
    background-color: transparent; 
    border: 5px solid #99A799; 
    position: relative; 
    overflow: hidden;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08), 0 4px 15px rgba(0,0,0,0.04);
    transition: transform 0.2s ease;
}

/* Header text */
.header-inner h1 {
    font-family: 'Poppins', sans-serif; 
    color: #99A799; 
    font-size: 2.7em; 
    font-weight: 700; 
    margin-bottom: 15px; 
}

.header-inner p {
    font-family: 'Poppins', sans-serif; 
    color: #99A799;
    font-size: 1.1em; 
    margin: 0; 
    line-height: 1.5em;
}

/* Home Button */           
.home-btn {
    position: fixed;
    top: 25px;
    left: 25px;
    background-color: #99A799;
    color: white;
    border: none;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    cursor: pointer;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    z-index: 999;
    display: flex;
    align-items: center;
    justify-content: center;
}
.home-btn:hover {
    background-color: white;
    color: #99A799;
    border: 2px solid #99A799;
    box-shadow: 0 3px 8px rgba(0,0,0,0.1);
}
.home-icon {
    width: 24px;
    height: 24px;
    fill: white;
    transition: fill 0.3s ease;
}
.home-btn:hover .home-icon {
    fill: #99A799;
}

.chat-header:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,0.1); }
.header-inner { position: relative; z-index: 1; }
.header-inner h1 { font-family: 'Poppins', sans-serif; color: #5C5C5C; font-size: 2.7em; font-weight: 700; margin-bottom: 25px; display: inline-block; position: relative; }
.header-text { color: #99A799; }
.header-inner h1::after { content: ""; position: absolute; bottom: -5px; left: 50%; transform: translateX(-50%); width: 50%; height: 3px; background-color: #99A799; border-radius: 2px; }
.header-inner p { color: #4a4a4a; font-size: 1.1em; font-family: 'Poppins', sans-serif; margin: 0; line-height: 1.5em; }
.header-inner span { color: #5c5c5c; font-size: 0.95em; }
</style>
""", unsafe_allow_html=True)

# Welcome Page
if st.session_state.page == "welcome":
    st.markdown("""
    <div style="
        text-align:center;
        padding:60px 40px;
        font-family:'Poppins', sans-serif;
        max-width: 750px;
        margin: 80px auto;
        background-color: rgba(153, 167, 153, 0.6);
        border-radius: 50% 40% 60% 60% / 60% 30% 50% 40%;  /* abstract shape */
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);  /* soft shadow */
        color: #4a4a4a;
        transform: rotate(-1deg);
    ">
        <h1 style="
            font-size:3em;
            font-weight: 700;
            margin-bottom: 25px;
            color: #4a4a4a;
        ">Welcome to Spam Detector ChatBot</h1>
        <p style="
            font-size:1.2em;
            max-width:600px;
            margin:auto;
            line-height:1.6em;
            color:#4a4a4a;
        ">
            Type any message, and I’ll tell you if it’s spam or ham. 
            Click below to begin!
        </p>
    </div>
    """, unsafe_allow_html=True)

    # "Get Started" button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Get Started", key="start_btn"):
            st.session_state.start_clicked = True
            st.rerun()

    if st.session_state.get("start_clicked", False):
        st.session_state.start_clicked = False
        st.session_state.page = "chat"
        st.rerun()

# Chat Page
if st.session_state.page == "chat":
    # Home Button
    home_button_html = """
    <form action="#" method="get">
        <button class="home-btn" name="home_clicked">
            <svg xmlns="http://www.w3.org/2000/svg" class="home-icon" viewBox="0 0 24 24">
                <path d="M3 9.75L12 3l9 6.75V21a.75.75 0 0 1-.75.75H3.75A.75.75 0 0 1 3 21V9.75z"/>
                <path d="M9 21V12h6v9" stroke="none"/>
            </svg>
        </button>
    </form>
    """
    st.markdown(home_button_html, unsafe_allow_html=True)

    # Click Home Button
    home_clicked = st.query_params.get("home_clicked")
    if home_clicked is not None:
        st.session_state.page = "welcome"
        st.session_state.messages = []  
        st.query_params.clear()
        st.rerun()

    # Header
    st.markdown("""
    <div class="chat-header">
        <div class="header-inner">
            <h1><span class="header-text">Spam Detector ChatBot</span> 📧</h1>
            <p>Type any message, and I’ll tell you if it’s spam or ham.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Chat Wrapper
    st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)
    st.markdown("<div class='chat-history' id='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Input Area
    st.markdown("<div class='input-area'>", unsafe_allow_html=True)
    col1, col2 = st.columns([6, 1])
    input_key = f"input_text_{st.session_state.input_counter}"

    with col1:
        user_input = st.text_input(
            "Your message:",
            placeholder="Type your message here...",
            label_visibility="collapsed",
            key=input_key
        )
    with col2:
        send = st.button("Send", use_container_width=True)
    
    # Message Handling
    if send or (user_input and st.session_state.get("last_input", "") != user_input):
        user_text = (user_input or "").strip()
        if user_text:
            st.session_state.last_input = user_text
            st.session_state.messages.append({"role": "user", "content": user_text})
            
            processed_text = normalize_text(user_text)
            processed_text = force_money_tokens(processed_text)

            prediction = predict_text(user_text)

            # Respond based on prediction
            if prediction == "ham":
                bot_response_text = "This message looks safe. ✅"
            else:  # spam
                bot_response_text = "Warning! This message might be spam. ⚠️"
            
            bot_message = f"{bot_response_text}<span class='sentiment-gray'>Prediction: {prediction.upper()}</span>"
            st.session_state.messages.append({"role": "bot", "content": bot_message})
            st.session_state.input_counter += 1
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)