import os
import tiktoken
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load local .env (only used for local testing)
load_dotenv()

# --- 1. CONFIGURATION & CLIENT ---
st.set_page_config(page_title="Sassy Bot", page_icon="🙄")

# Safe API Key retrieval (Checks Streamlit Secrets, then .env)
api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("Missing API Key! Please add OPENROUTER_API_KEY to your secrets or .env file.")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# Constants
MODEL = "openrouter/free"
TOKEN_BUDGET = 1000

# --- 2. TOKEN MANAGEMENT (Fixed for Null Safety) ---
def get_encoding(model):
    try:
        return tiktoken.encoding_for_model(model)
    except (KeyError, ValueError):
        return tiktoken.get_encoding("cl100k_base")

ENCODING = get_encoding(MODEL)

def count_tokens(text):
    """Safely count tokens in a string."""
    if not isinstance(text, str):
        return 0
    return len(ENCODING.encode(text))

def enforce_token_budget():
    """Trims st.session_state.messages to stay within budget."""
    messages = st.session_state.messages
    # Safely sum tokens using .get() to avoid crashes
    while sum(count_tokens(m.get("content", "")) for m in messages) > TOKEN_BUDGET and len(messages) > 1:
        # Keep index 0 (System Prompt), remove index 1 (Oldest interaction)
        messages.pop(1)

# --- 3. SESSION STATE INITIALIZATION ---
if "terminated" not in st.session_state:
    st.session_state.terminated = False

# Initialize messages if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. SIDEBAR UI ---
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

# Ensure System Prompt is always at index 0
if len(st.session_state.messages) == 0:
    st.session_state.messages.append({"role": "system", "content": sys_msg})

# Sidebar Buttons
if st.sidebar.button("Update Personality"):
    st.session_state.messages[0] = {"role": "system", "content": sys_msg}
    st.sidebar.success("Personality Updated!")

if st.sidebar.button("Reset Chat"):
    st.session_state.messages = [{"role": "system", "content": sys_msg}]
    st.session_state.terminated = False
    st.rerun()

# --- 5. CHAT INTERFACE ---
st.title("🙄 Sassy Bot")

# Termination Screen
if st.session_state.terminated:
    st.warning("Chat Session Ended. Reset the chat in the sidebar to talk again.")
    # Show history but disable input
    for message in st.session_state.messages[1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    st.stop()

# Display existing messages from history
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Logic
if prompt := st.chat_input("What do you want now?"):
    
    # Check for exit keywords
    if prompt.lower() in ["exit", "quit", "goodbye"]:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": "FINALLY. Goodbye! 🙄"})
        st.session_state.terminated = True
        st.rerun()

    # 1. Add and display User message
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
                
                # Safely get response content
                response_text = completion.choices[0].message.content
                
                if response_text:
                    st.markdown(response_text)
                    # 4. Save to history
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                else:
                    st.error("The API returned an empty response.")
                    
        except Exception as e:
            st.error(f"The API is as broken as my spirit: {e}")

# Footer Info (Sidebar)
with st.sidebar:
    st.divider()
    # Calculate current usage safely
    current_tokens = sum(count_tokens(m.get("content", "")) for m in st.session_state.messages)
    st.info(f"Memory Usage: {current_tokens}/{TOKEN_BUDGET} tokens")