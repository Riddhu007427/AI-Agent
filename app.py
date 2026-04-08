import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

# Load environment variables
load_dotenv()

# --- INITIAL SETUP ---
st.set_page_config(page_title="Token-Managed AI", page_icon="🤖")
st.title("🤖 Managed Token Agent")

# API Key Check
api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("API key not found.")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

MODEL = "openrouter/free"
MAX_TOKENS_LIMIT = 500  # Total limit for conversation history
SYSTEM_PROMPT = "You are a helpful assistant that provides concise answers."

# --- TOKEN MANAGEMENT FUNCTIONS ---
def get_tokenizer(model):
    try:
        return tiktoken.encoding_for_model(model)
    except (KeyError, ValueError):
        return tiktoken.get_encoding("cl100k_base")

ENCODING = get_tokenizer(MODEL)

def count_tokens(text):
    return len(ENCODING.encode(text))

def ensure_token_limit(messages, max_limit):
    # Total tokens = sum of all message contents
    while sum(count_tokens(m["content"]) for m in messages) > max_limit and len(messages) > 1:
        # Keep index 0 (System Prompt), remove index 1 (Oldest chat)
        messages.pop(1)
    return messages

# --- SESSION STATE (Memory) ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# --- UI: DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# --- UI: CHAT INPUT & LOGIC ---
if user_input := st.chat_input("Ask me anything..."):
    # 1. Add and display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Trim history BEFORE sending to API
    st.session_state.messages = ensure_token_limit(st.session_state.messages, MAX_TOKENS_LIMIT)

    # 3. Generate response
    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=st.session_state.messages,
                temperature=0.7,
                max_tokens=200 # Limit for the single response
            )
            reply = response.choices[0].message.content
            st.markdown(reply)
            
            # 4. Save assistant response to state
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"Error: {e}")

# Display current token usage in sidebar
current_usage = sum(count_tokens(m["content"]) for m in st.session_state.messages)
st.sidebar.write(f"📊 Current Session Tokens: {current_usage} / {MAX_TOKENS_LIMIT}")