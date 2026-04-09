import os
import tiktoken
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load local .env (only used for local testing)
load_dotenv()

# --- 1. CONFIGURATION & CLIENT ---
st.set_page_config(page_title="Sassy Bot", page_icon="🙄")

# Safe API Key retrieval
api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("Missing API Key! Add OPENROUTER_API_KEY to secrets.")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# Constants
MODEL = "openrouter/free"
TOKEN_BUDGET = 1000

# --- 2. TOKEN MANAGEMENT ---
def get_encoding(model):
    try:
        return tiktoken.encoding_for_model(model)
    except (KeyError, ValueError):
        return tiktoken.get_encoding("cl100k_base")

ENCODING = get_encoding(MODEL)

def count_tokens(text):
    return len(ENCODING.encode(text))

def enforce_token_budget():
    """Trims st.session_state.messages to stay within budget."""
    messages = st.session_state.messages
    while sum(count_tokens(m["content"]) for m in messages) > TOKEN_BUDGET and len(messages) > 1:
        # Keep index 0 (System), remove index 1 (Oldest interaction)
        messages.pop(1)

# --- 3. SIDEBAR UI ---
st.sidebar.header("Agent Settings")
max_tokens_val = st.sidebar.slider("Max Response Tokens", 1, 500, 150)
temp_val = st.sidebar.slider("Temperature (Chaos)", 0.0, 1.0, 0.7)

system_type = st.sidebar.selectbox("Personality", ("Sassy", "Angry", "Custom"))
if system_type == "Sassy":
    sys_msg = "You are a fed up and sassy assistant who hates answering questions."
elif system_type == "Angry":
    sys_msg = "YOU ARE AN ANGRY ASSISTANT WHO YELLS IN ALL CAPS AND IS VERY ANNOYED."
else:
    sys_msg = st.sidebar.text_area("Custom Persona", "You are a helpful robot.")

# --- 4. SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": sys_msg}]

# Apply Personality Changes
if st.sidebar.button("Update Personality"):
    st.session_state.messages[0] = {"role": "system", "content": sys_msg}
    st.sidebar.success("Personality Updated!")

if st.sidebar.button("Reset Chat"):
    st.session_state.messages = [{"role": "system", "content": sys_msg}]
    st.rerun()

# --- 5. CHAT INTERFACE ---
st.title("🙄 Sassy Bot")
st.caption("Ask me something... if you absolutely must.")

# Display existing messages from history
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Input logic
# if prompt := st.chat_input("What do you want?"):
#     # Check for exit
#     if prompt.lower() in ["exit", "quit", "goodbye"]:
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         st.session_state.messages.append({"role": "assistant", "content": "FINALLY. Goodbye! 🙄"})
#         st.session_state.terminated = True
#         st.rerun()

# Chat Input Logic
if prompt := st.chat_input("What do you want now?"):
    # 1. Add User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Enforce budget before calling API
    enforce_token_budget()

    # 3. Call API
    with st.chat_message("assistant"):
        try:
            with st.spinner("Ugh, fine..."):
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=st.session_state.messages,
                    temperature=temp_val,
                    max_tokens=max_tokens_val
                )
                response_text = completion.choices[0].message.content
                st.markdown(response_text)
                
                # 4. Save to history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
        except Exception as e:
            st.error(f"The API is as broken as my spirit: {e}")

# Footer Info
with st.sidebar:
    st.divider()
    current_tokens = sum(count_tokens(m["content"]) for m in st.session_state.messages)
    st.info(f"Memory Usage: {current_tokens}/{TOKEN_BUDGET} tokens")