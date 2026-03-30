import streamlit as st

def render_telemetry_badges(tier: str, latency: int, chunks: int, domain: str):
    """
    Renders the small status badges (Confidence, Latency, Domain) 
    using HTML/CSS for a sleek look.
    """
    tier_upper = str(tier).upper()
    
    if tier_upper == "HIGH":
        color = "#10B981" # Emerald green
    elif tier_upper == "MEDIUM":
        color = "#F59E0B" # Amber
    else:
        color = "#EF4444" # Red
        
    html = f"""
    <div class="telemetry-row">
        <span class="tel-badge" style="border-color: {color}; color: {color}">
            ● {tier_upper} CONFIDENCE
        </span>
        <span class="tel-badge">⏱️ {latency/1000:.1f}s</span>
        <span class="tel-badge">📚 {chunks} Docs</span>
        <span class="tel-badge">🏷️ {domain.title()}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_source_card(source: dict):
    """
    Renders a single source citation as a styled HTML card.
    """
    # Extract
    act_name = source.get("act_name") or "Case Law / Judgment"
    section = source.get("section") or ""
    score = source.get("score", 0.0)
    
    # Format Title
    title = f"{act_name}"
    if section:
        title = f"{section}, {act_name}"
        
    # Detail string
    doc_type = str(source.get("doc_type", "")).title()
    citation = source.get("citation") or source.get("case_name") or "No official citation"
    
    # Score Coloring (using the updated sigmoid boundaries > 0.72 = high)
    score_pct = int(score * 100)
    if score >= 0.72:
        score_bg = "rgba(16, 185, 129, 0.15)"
        score_col = "#34D399"
    elif score >= 0.50:
        score_bg = "rgba(245, 158, 11, 0.15)"
        score_col = "#FBBF24"
    else:
        score_bg = "rgba(239, 68, 68, 0.15)"
        score_col = "#F87171"
        
    html = f"""
    <div class="source-card">
        <div class="source-title">
            <span class="source-score-badge" style="background-color:{score_bg}; color:{score_col};">{score_pct}% Match</span>
            {title}
        </div>
        <div class="source-meta">
            <strong>{doc_type}</strong> | {citation}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_disclaimer(text: str):
    st.markdown(
        f"""
        <div style="font-size: 0.8rem; color: #64748B; margin-top: 2rem; border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 1rem; text-align: center;">
            ⚖️ <em>{text}</em>
        </div>
        """,
        unsafe_allow_html=True
    )
