import streamlit as st
import os

from api import chat_query
from components import render_source_card, render_telemetry_badges, render_disclaimer

# --- Page Config ---
st.set_page_config(
    page_title="Bharat Law AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load CSS ---
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            
load_css()

# --- Initialize State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Left Sidebar (Settings) ---
with st.sidebar:
    st.markdown("<h1>Bharat Law AI</h1>", unsafe_allow_html=True)
    st.markdown("An expert RAG knowledge system for Indian Penal, Civil, and Constitutional Law.")
    
    st.divider()
    
    domain_override = st.selectbox(
        "Forced Override Domain",
        ["Auto", "Criminal", "Civil", "Economic", "Family", "Constitutional", "Misc"],
        help="Forces the domain router to search only specific directories. Auto uses LLM selection."
    )
    
    retrieval_k = st.slider("Retrieval Depth (Chunks)", 1, 10, 5, help="Number of chunks fed to Gemini context.")
    
    st.divider()
    
    if st.button("Clear Deposition", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()

# --- Helper to render briefcase ---
def render_briefcase(ctx):
    st.markdown("<div style='margin-bottom: 10px;'><strong style='color: #D4AF37; font-family: Merriweather, serif; font-size: 1.2rem;'>The Briefcase</strong></div>", unsafe_allow_html=True)
    
    # Badges
    render_telemetry_badges(
        tier=ctx.get("confidence_tier", "LOW"),
        latency=ctx.get("latency_ms", 0),
        chunks=ctx.get("context_chunks_used", 0),
        domain=ctx.get("domain_detected", "Unknown")
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Legal Explanation Accordion
    if ctx.get("legal_explanation"):
        with st.expander("🏛️ Legal Analysis", expanded=False):
            st.markdown(f"""
            <div style="font-size: 1.05rem; color: #CBD5E1; line-height: 1.7;">
                {ctx["legal_explanation"]}
            </div>
            """, unsafe_allow_html=True)
            
    # Disclaimer
    if ctx.get("disclaimer"):
        render_disclaimer(ctx["disclaimer"])


# --- Main Custom Layout ---
st.title("The Deposition")

# Render Message History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        # Render standard user bubble
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        # Render novel assistant block: Chat on left, Briefcase on right
        with st.chat_message("assistant"):
            col_chat, col_context = st.columns([4.5, 5.5], gap="large")
            with col_chat:
                st.markdown(msg["content"])
            with col_context:
                ctx = msg.get("context")
                if ctx:
                    render_briefcase(ctx)

# UI Divider for clean separation of current input
st.markdown("<br><br>", unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Ask a legal question (e.g., Explain Section 302 of the IPC)..."):
    
    # Store user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Local UI echo for immediate feedback
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Spinner block while fetching API
    with st.chat_message("assistant"):
        with st.spinner("Consulting Indian Law Precedents..."):
            response_data = chat_query(prompt, domain=domain_override, top_k=retrieval_k)
            
        if response_data:
            answer = response_data.get("answer", "System experienced a failure.")
            # We don't local-echo the full side-by-side array here because 
            # we need to trigger st.rerun() to lock it efficiently into state history.
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "context": response_data
            })
            st.rerun()
