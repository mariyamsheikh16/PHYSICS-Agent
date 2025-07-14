import streamlit as st
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY is not set in the .env file")
    st.stop()

# Gemini-compatible client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Gemini model setup
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Run config
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Define Physics Agent
physics_agent = Agent(
    name='Physics Expert Agent',
    instructions="""
You are a helpful physics tutor. Only answer if the question is about physics concepts like motion, forces, energy, electromagnetism, quantum physics, thermodynamics, optics, etc.
If a question is unrelated to physics, respond: 'Sorry, I can only help with physics-related questions.'
"""
)

# Run the agent
async def run_agent(prompt: str):
    result = await Runner.run(physics_agent, input=prompt, run_config=config)
    return result.final_output

# Simple keyword check
def is_physics_related(text):
    physics_keywords = [
        "velocity", "motion", "acceleration", "quantum", "thermodynamics", "gravity",
        "relativity", "electric", "magnetism", "wave", "optics", "force", "energy",
        "kinematics", "newton", "mass", "friction", "resistance", "voltage", "current",
        "circuit", "momentum", "projectile", "laws of motion", "mechanics", "physics"
    ]
    return any(re.search(rf"\b{word}\b", text.lower()) for word in physics_keywords)

# Streamlit UI
st.set_page_config(page_title="Physics Assistant", page_icon="📘", layout="centered")

# Basic styling
st.markdown("""
    <style>
    body {
        background-color: #f4f8fb;
    }
    .stTextArea textarea {
        border: 1px solid #cfd8dc;
        border-radius: 10px;
        padding: 12px;
        font-size: 1rem;
    }
    .stButton>button {
        background-color: #1976d2;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #115293;
    }
    .footer {
        text-align: center;
        font-size: 0.85rem;
        color: #888;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center;'>🔬 AI Physics Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Ask me anything about physics — I'm here to help!</p>", unsafe_allow_html=True)

# Prompt input
with st.form("physics_form", clear_on_submit=False):
    user_input = st.text_area("Enter your physics question below:", height=180, placeholder="e.g., What is Newton's second law?")
    submitted = st.form_submit_button("Get Answer")

# Handle response
if submitted:
    if not user_input.strip():
        st.warning("⚠️ Please enter a question first.")
    elif not is_physics_related(user_input):
        st.error("❌ Sorry, I can only help with physics-related questions.")
    else:
        with st.spinner("🔍 Thinking like a physicist..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(run_agent(user_input))
        st.success("✅ Here's your answer:")
        st.markdown("### 📄 Response:")
        st.markdown(response)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='footer'>Made with ❤️ by Mariyam Sheikh | Powered by Gemini + Agentic SDK</div>", unsafe_allow_html=True)
