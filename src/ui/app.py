import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Streamlit Page Configuration
st.set_page_config(
    page_title="Drug Classification & Insights",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FastAPI Endpoint URL
API_URL = "http://127.0.0.1:8000"


def fetch_api_data(endpoint: str, default=None):
    """Helper to fetch data from the FastAPI backend with fallback."""
    try:
        response = requests.get(f"{API_URL}{endpoint}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return default


# Custom header banner
st.title("💊 Drug Analytics & Assistant Dashboard")
st.caption("Interactive data exploration, clinical insights, and RAG assistant chatbot.")

# Navigation Tabs
tab_insights, tab_chatbot = st.tabs(["📊 Dataset Insights & Visualizations", "💬 Drug Assistant (Chatbot)"])

# ==========================================
# TAB 1: DATASET INSIGHTS & CHARTS
# ==========================================
with tab_insights:
    # Summary Metrics Row
    stats = fetch_api_data("/api/stats", {"total_records": 248232, "therapeutic_classes_count": 52})
    habit_data = fetch_api_data("/api/charts/habit-forming", {"labels": ["No", "Yes"], "counts": [242206, 6003]})

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Records", f"{stats.get('total_records', 0):,}")
    with col_m2:
        st.metric("Therapeutic Classes", stats.get("therapeutic_classes_count", 0))
    with col_m3:
        habit_yes = habit_data["counts"][habit_data["labels"].index("Yes")] if "Yes" in habit_data.get("labels", []) else 6003
        st.metric("Habit-Forming Drugs", f"{habit_yes:,}", delta="Requires STRIDE Security", delta_color="inverse")
    with col_m4:
        st.metric("API Status", "Connected" if stats.get("total_records") else "Offline (Using cached data)")

    st.divider()

    # ROW 1: Therapeutic Class (Top 15) & Habit-Forming Split
    col_row1_left, col_row1_right = st.columns([1.6, 1])

    with col_row1_left:
        st.subheader("1. Therapeutic Class Distribution (Top 15)")
        st.caption("Distribution of drugs across major clinical categories — highlighting extreme skew towards Anti-Infectives.")
        
        tc_data = fetch_api_data("/api/charts/therapeutic-class", {
            "classes": ["ANTI INFECTIVES", "GASTRO INTESTINAL", "PAIN ANALGESICS", "RESPIRATORY", "CARDIO VASCULAR", "DERMATOLOGY", "CNS", "ENDOCRINE", "OPHTHALMOLOGICAL", "GYNAECOLOGICAL", "VITAMINS", "UROLOGY"],
            "counts": [57500, 33600, 32300, 24100, 22000, 18500, 15000, 12000, 8500, 7200, 4100, 1800]
        })

        if tc_data.get("classes"):
            fig_tc = px.bar(
                x=tc_data["classes"],
                y=tc_data["counts"],
                labels={"x": "Therapeutic Class", "y": "Number of Drugs"},
                color=tc_data["counts"],
                color_continuous_scale="Viridis",
            )
            fig_tc.update_layout(xaxis_tickangle=-45, showlegend=False, height=450)
            st.plotly_chart(fig_tc, use_container_width=True)

    with col_row1_right:
        st.subheader("2. Habit-Forming Split")
        st.caption("Crucial for security and STRIDE modeling (controlled substance data is sensitive).")

        if habit_data.get("labels"):
            fig_hf = px.pie(
                names=habit_data["labels"],
                values=habit_data["counts"],
                hole=0.45,
                color=habit_data["labels"],
                color_discrete_map={"No": "#2ca02c", "Yes": "#d62728"}
            )
            fig_hf.update_traces(textposition='inside', textinfo='percent+label')
            fig_hf.update_layout(height=450)
            st.plotly_chart(fig_hf, use_container_width=True)

    st.divider()

    # ROW 2: Missing Data by Column & Side Effects Distribution
    col_row2_left, col_row2_right = st.columns(2)

    with col_row2_left:
        st.subheader("3. Missing Data by Column (%)")
        st.caption("Visually justifies keeping 'NA' as a valid category instead of dropping ~44% of data rows.")

        missing_data = fetch_api_data("/api/charts/missing-data", {
            "columns": ["chemical_class", "action_class", "substitutes", "therapeutic_class", "uses", "side_effects", "habit_forming", "drug_name", "id"],
            "percentages": [44.2, 44.1, 4.3, 1.2, 0.5, 0.2, 0.0, 0.0, 0.0]
        })

        if missing_data.get("columns"):
            fig_miss = px.bar(
                x=missing_data["percentages"],
                y=missing_data["columns"],
                orientation="h",
                labels={"x": "Missing Percentage (%)", "y": "Column"},
                color=missing_data["percentages"],
                color_continuous_scale="Reds"
            )
            fig_miss.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_miss, use_container_width=True)

    with col_row2_right:
        st.subheader("4. Side Effects Count Distribution")
        st.caption("Distribution of side-effect list lengths per drug (Mean: ~6.7, Median: 6, Max: 43).")

        se_dist = fetch_api_data("/api/charts/side-effects-distribution", {
            "binned_labels": ["1-2", "3-5", "6-10", "11-15", "16-20", "21-30", "31+"],
            "binned_counts": [18000, 72000, 110000, 32000, 12000, 3800, 400],
            "stats": {"mean": 6.7, "median": 6, "max": 43, "min": 1}
        })

        if se_dist.get("binned_labels"):
            fig_dist = px.bar(
                x=se_dist["binned_labels"],
                y=se_dist["binned_counts"],
                labels={"x": "Number of Side Effects", "y": "Frequency (Number of Drugs)"},
                color_discrete_sequence=["#1f77b4"]
            )
            fig_dist.update_layout(height=400)
            st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # ROW 3: Top 20 Common Side Effects & Word Cloud of Uses
    col_row3_left, col_row3_right = st.columns(2)

    with col_row3_left:
        st.subheader("5. Top 20 Most Common Side Effects")
        st.caption("Unpacked values across all medicine profiles — shows standard adverse reactions.")

        top_se = fetch_api_data("/api/charts/top-side-effects", {
            "side_effects": [
                "Nausea", "Vomiting", "Diarrhea", "Headache", "Dizziness",
                "Abdominal pain", "Rash", "Sleepiness", "Upset stomach", "Allergic reaction",
                "Weakness", "Fatigue", "Dryness in mouth", "Tremors", "Loss of appetite",
                "Constipation", "Itching", "Fever", "Sweating", "Joint pain"
            ],
            "counts": [
                78500, 72300, 64200, 52100, 48900,
                38700, 31200, 28400, 26100, 23500,
                20100, 18900, 17200, 15400, 14200,
                13800, 12500, 11200, 10100, 9300
            ]
        })

        if top_se.get("side_effects"):
            fig_top_se = px.bar(
                x=top_se["counts"],
                y=top_se["side_effects"],
                orientation="h",
                labels={"x": "Count", "y": "Side Effect"},
                color=top_se["counts"],
                color_continuous_scale="Teal"
            )
            fig_top_se.update_layout(height=480, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_top_se, use_container_width=True)

    with col_row3_right:
        st.subheader("6. Word Cloud of Clinical Uses")
        st.caption("Visual proof of why clinical text benefits from RAG semantics instead of exact keyword match.")

        uses_data = fetch_api_data("/api/charts/uses-text", {
            "text": "Treatment Bacterial infections Allergic conditions Cough Pain Relief Hypertension Diabetes Fungal infection Fever Inflammation Acid Reflux Anxiety Depression Asthma Skin conditions Gastrointestinal Ulcers"
        })

        text = uses_data.get("text", "")
        if text.strip():
            wc = WordCloud(
                width=800,
                height=480,
                background_color="white",
                colormap="magma",
                max_words=80,
                stopwords={"Treatment", "of", "and", "due", "to", "with", "in", "for"}
            ).generate(text)

            fig_wc, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)


# ==========================================
# TAB 2: DRUG ASSISTANT CHATBOT UI
# ==========================================
with tab_chatbot:
    st.subheader("🤖 Clinical Drug Assistant")
    st.caption("Ask questions regarding drug indications, therapeutic classifications, and potential side effects.")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your Drug Classification & Clinical Assistant powered by RAG. Ask me anything about medicine uses, side effects, or drug classes.",
                "sources": []
            }
        ]

    # Render previous conversation messages (with sources if present)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption("📚 Sources: " + ", ".join(msg["sources"]))

    # Chat Input Box
    user_prompt = st.chat_input("Type your question here (e.g. 'What are common side effects of Augmentin?')...")

    if user_prompt:
        # Display user query immediately
        st.session_state.messages.append({"role": "user", "content": user_prompt, "sources": []})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Send request to FastAPI backend (longer timeout — RAG + LLM takes time)
        bot_response = None
        bot_sources = []
        try:
            res = requests.post(
                f"{API_URL}/api/chat",
                json={"message": user_prompt},
                timeout=60,   # RAG + LLM can take up to ~30s on first call
            )
            if res.status_code == 200:
                data = res.json()
                bot_response = data.get("reply")
                bot_sources = data.get("sources", [])
        except Exception:
            pass

        # Fallback if FastAPI is not currently running
        if not bot_response:
            bot_response = (
                f"*(FastAPI offline)*: You asked: '{user_prompt}'. "
                "Start the FastAPI server with `python -m uvicorn src.api.app:app --port 8000` "
                "to enable the RAG-powered assistant."
            )

        # Persist and display bot response + sources
        st.session_state.messages.append({
            "role": "assistant",
            "content": bot_response,
            "sources": bot_sources,
        })
        with st.chat_message("assistant"):
            st.markdown(bot_response)
            if bot_sources:
                st.caption("📚 Sources: " + ", ".join(bot_sources))

