import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(PROJECT_ROOT / ".env")

from ask_forgeiq import (  # noqa: E402
    API_KEY,
    MODEL_NAME,
    generate_answer,
)
from retriever import (  # noqa: E402
    load_vector_store,
    search_knowledge_base,
)


# --------------------------------------------------
# 2. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ForgeIQ | Industrial Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# 3. CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263244;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 1rem;
        margin-top: 0.4rem;
    }

    .status-card {
        background-color: #111827;
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
    }

    .status-label {
        color: #9ca3af;
        font-size: 0.85rem;
    }

    .status-value {
        color: #f8fafc;
        font-size: 1.15rem;
        font-weight: 700;
    }

    .citation-card {
        background-color: #111827;
        border-left: 3px solid #38bdf8;
        border-radius: 6px;
        padding: 12px;
        margin: 8px 0;
    }

    .citation-source {
        color: #7dd3fc;
        font-weight: 700;
    }

    .muted-text {
        color: #9ca3af;
        font-size: 0.85rem;
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 12px;
    }

    .stButton button {
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 4. LOAD THE EXISTING KNOWLEDGE BASE
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def initialize_forgeiq():
    """
    Load the existing embedding model and ChromaDB collection.

    Streamlit caches these resources so they are not reloaded
    for every interaction.
    """

    model, collection = load_vector_store()

    client = None

    if API_KEY:
        client = genai.Client(api_key=API_KEY)

    return model, collection, client


# --------------------------------------------------
# 5. SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_citations" not in st.session_state:
    st.session_state.last_citations = []

if "last_invalid_citations" not in st.session_state:
    st.session_state.last_invalid_citations = []

if "last_insufficient_information" not in st.session_state:
    st.session_state.last_insufficient_information = False


# --------------------------------------------------
# 6. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ ForgeIQ")

    st.markdown(
        """
        <p class="muted-text">
        Industrial maintenance knowledge assistant
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### System overview")

    st.markdown(
        """
        **Knowledge retrieval**  
        Semantic search using Sentence Transformers

        **Vector database**  
        ChromaDB

        **Answer generation**  
        Gemini API

        **Evidence tracking**  
        Structured source citations
        """
    )

    st.divider()

    st.markdown("### Example questions")

    example_questions = [
        "What should be monitored in the hydraulic system?",
        "Was the cause of the unusual hydraulic noise confirmed?",
        "What corrective action was completed for work order FIQ-WO-003?",
        "What is the exact hydraulic pressure limit?",
        "Can a technician loosen a pressurized hydraulic connection?",
    ]

    for example in example_questions:
        if st.button(
            example,
            key=f"example_{example}",
            use_container_width=True,
        ):
            st.session_state.pending_question = example

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.last_citations = []
        st.session_state.last_invalid_citations = []
        st.session_state.last_insufficient_information = False
        st.rerun()

    st.caption(
        "ForgeIQ is a demonstration system. "
        "Always follow approved maintenance and safety procedures."
    )


# --------------------------------------------------
# 7. MAIN HEADER
# --------------------------------------------------

st.markdown(
    '<p class="main-title">ForgeIQ</p>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="subtitle">
    Evidence-grounded industrial maintenance intelligence.
    Ask questions about maintenance documentation and work orders.
    </p>
    """,
    unsafe_allow_html=True,
)

st.write("")


# --------------------------------------------------
# 8. INITIALIZE THE APPLICATION
# --------------------------------------------------

try:
    with st.spinner("Loading the ForgeIQ knowledge base..."):
        embedding_model, collection, client = initialize_forgeiq()

except Exception as error:
    st.error(
        "ForgeIQ could not load its knowledge base. "
        "Check that the existing vector database is available."
    )

    with st.expander("Technical details"):
        st.code(str(error))

    st.stop()


# --------------------------------------------------
# 9. SYSTEM STATUS CARDS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">Indexed passages</div>
            <div class="status-value">{collection.count()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    retrieval_status = (
        "Available"
        if collection.count() > 0
        else "Empty"
    )

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">Knowledge retrieval</div>
            <div class="status-value">{retrieval_status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    generation_status = (
        "Configured"
        if client is not None
        else "API key missing"
    )

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">Answer generation</div>
            <div class="status-value">{generation_status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()


# --------------------------------------------------
# 10. DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            citations = message.get("citations", [])

            if citations:
                with st.expander("View supporting sources"):
                    for citation in citations:
                        st.markdown(
                            f"**[{citation['source_id']}] "
                            f"{citation['source']}**"
                        )

                        st.caption(
                            f"Section: {citation['section']}"
                        )

                        if citation.get("claim"):
                            st.write(
                                f"Claim: {citation['claim']}"
                            )

            invalid_citations = message.get(
                "invalid_citations",
                [],
            )

            if invalid_citations:
                st.warning(
                    "Some generated citations could not be "
                    "matched to the retrieved passages."
                )

            if message.get("insufficient_information", False):
                st.info(
                    "The available documentation does not "
                    "establish all the requested information."
                )


# --------------------------------------------------
# 11. PROCESS USER QUESTIONS
# --------------------------------------------------

pending_question = st.session_state.pop(
    "pending_question",
    None,
)

question = st.chat_input(
    "Ask ForgeIQ about maintenance documentation..."
)

if pending_question:
    question = pending_question


if question:
    question = question.strip()

    if not question:
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    if client is None:
        error_message = (
            "Gemini API is not configured. "
            "Check that GEMINI_API_KEY is available in your .env file."
        )

        with st.chat_message("assistant"):
            st.error(error_message)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
                "citations": [],
                "invalid_citations": [],
                "insufficient_information": False,
            }
        )

        st.stop()

    if collection.count() == 0:
        error_message = (
            "The knowledge base is empty. "
            "Please build the index before asking questions."
        )

        with st.chat_message("assistant"):
            st.error(error_message)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
                "citations": [],
                "invalid_citations": [],
                "insufficient_information": False,
            }
        )

        st.stop()

    with st.chat_message("assistant"):
        with st.spinner(
            "Searching documentation and generating an answer..."
        ):
            try:
                # Step 1: Retrieve relevant passages.
                retrieved_chunks = search_knowledge_base(
                    question,
                    embedding_model,
                    collection,
                    top_k=3,
                )

                if not retrieved_chunks:
                    result = None
                    answer = (
                        "I couldn't retrieve relevant passages "
                        "from the available documentation."
                    )

                else:
                    # Step 2: Generate the structured answer.
                    result = generate_answer(
                        question,
                        retrieved_chunks,
                        client,
                    )

                    answer = result["answer"]

                # Step 3: Display the answer.
                st.markdown(answer)

                # Step 4: Display validated source citations.
                citations = (
                    result["citations"]
                    if result
                    else []
                )

                invalid_citations = (
                    result["invalid_citations"]
                    if result
                    else []
                )

                insufficient_information = (
                    result["insufficient_information"]
                    if result
                    else False
                )

                if citations:
                    with st.expander(
                        "View supporting sources",
                        expanded=True,
                    ):
                        for citation in citations:
                            st.markdown(
                                f"**[{citation['source_id']}] "
                                f"{citation['source']}**"
                            )

                            st.caption(
                                f"Section: {citation['section']}"
                            )

                            if citation.get("claim"):
                                st.write(
                                    f"Claim: {citation['claim']}"
                                )

                elif result:
                    st.warning(
                        "No citations could be validated for "
                        "this answer."
                    )

                if invalid_citations:
                    st.warning(
                        "Some generated citations could not be "
                        "matched to retrieved passages."
                    )

                    with st.expander(
                        "View citation validation warnings"
                    ):
                        for invalid in invalid_citations:
                            st.write(invalid["reason"])

                if insufficient_information:
                    st.info(
                        "The available documentation does not "
                        "establish all the requested information."
                    )

                # Step 5: Save the exchange to session history.
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "invalid_citations": invalid_citations,
                        "insufficient_information": (
                            insufficient_information
                        ),
                    }
                )

            except Exception as error:
                error_text = str(error)
                lowered_error = error_text.lower()

                if (
                    "quota" in lowered_error
                    or "resource_exhausted" in lowered_error
                    or "429" in lowered_error
                ):
                    user_message = (
                        "The Gemini API quota is currently "
                        "unavailable or exhausted. Your question "
                        "was not answered. Check your API quota "
                        "before trying again."
                    )

                elif (
                    "503" in lowered_error
                    or "unavailable" in lowered_error
                ):
                    user_message = (
                        "The answer-generation service is "
                        "temporarily unavailable. Please try again "
                        "later."
                    )

                else:
                    user_message = (
                        "ForgeIQ encountered an error while "
                        "processing your question."
                    )

                st.error(user_message)

                with st.expander("Technical details"):
                    st.code(error_text)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": user_message,
                        "citations": [],
                        "invalid_citations": [],
                        "insufficient_information": False,
                    }
                )


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "ForgeIQ | Industrial Maintenance RAG Demonstration | "
    "Answers must be verified against approved documentation."
)

