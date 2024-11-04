import streamlit as st
import requests
import pandas as pd
from search_logger import SearchLogger

st.set_page_config(page_title="Smart Document Search", layout="wide", initial_sidebar_state="collapsed")


# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2E4057;
        font-size: 3rem !important;
        padding-bottom: 1rem;
    }
    .search-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
            .stDataFrame {
        max-height: none !important;  /* Remove scroll */
    }
    div[data-testid="stVerticalBlock"] > div:has(div.stDataFrame) {
        overflow: visible !important;
    }
    </style>
    """, unsafe_allow_html=True)


def truncate_text(text, max_length=50):
    return text[:max_length] + "..." if len(text) > max_length else text

def display_document(docs, selected_index, doc_type):
    st.markdown(f"### 📄 {doc_type} Content")
    st.markdown(f"```{docs[selected_index]}```")

# Initialize logger
logger = SearchLogger()

st.title("🔍 Smart Document Search")
st.markdown("### Discover relevant documents using AI-powered semantic search")

# Search section
with st.container():
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    query = st.text_input(
        "🤔 What would you like to find?",
        max_chars=200,
        placeholder="Enter your search query here..."
    )
    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        search_button = st.button("🚀 Search", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if "search_performed" not in st.session_state:
    st.session_state.search_performed = False

if search_button and query:
    with st.spinner('🤖 AI is searching through documents...'):
        try:
            response = requests.post("http://backend:8051/search", json={"query": query})

            if response.status_code == 200:
                st.session_state.search_performed = True
                data = response.json()

                # Store the response data in session state
                st.session_state.rel_docs = data["rel_docs"]
                st.session_state.rel_docs_sim = data["rel_docs_sim"]
            else:
                st.error(f"🚫Error: Unable to retrieve documents. Status code: {response.status_code}")
                st.write(f"Response content: {response.text}")
        except Exception as e:
            st.error(f"🚫 Error occurred: {str(e)}")

if st.session_state.search_performed:
    df_similar = pd.DataFrame({
        "📑 Document": [truncate_text(doc) for doc in st.session_state.rel_docs],
        "🎯 Relevance Score": st.session_state.rel_docs_sim,  # Using rel_docs_sim instead of distances
    }).reset_index(drop=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📊 Search Results")
        if len(df_similar) > 0:
            selected_similar = st.selectbox(
                "📌 Select a document to view details:",
                options=list(range(len(df_similar))),
                format_func=lambda x: df_similar.loc[x, "📑 Document"],
                key="similar_select"
            )

            # Logging logic
            if "last_selected" not in st.session_state:
                st.session_state.last_selected = None
            if st.session_state.last_selected != selected_similar:
                log_data = logger.log_selection(
                    query=query,
                    all_results=st.session_state.rel_docs,
                    selected_result=st.session_state.rel_docs[selected_similar],
                    similarities=st.session_state.rel_docs_sim
                )
                st.session_state.last_selected = selected_similar

            # Style the dataframe
            st.dataframe(
                df_similar.style
                .format({"🎯 Relevance Score": "{:.2%}"})
                .background_gradient(subset=["🎯 Relevance Score"], cmap="YlOrRd"),
                use_container_width=True,
                height=len(df_similar) * 35 + 38  # Dynamic height based on number of rows
            )
        else:
            st.info("🤷‍♂️ No matching documents found. Try a different search!")

    with col2:
        st.markdown("### 📝 Document Preview")
        if len(df_similar) > 0 and selected_similar is not None:
            display_document(st.session_state.rel_docs, selected_similar, "Selected")

else:
    st.write("👋 Welcome! Enter your search query above to get started.")

# Footer
st.markdown("---")
st.markdown("### 💡 Tips")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("• Use specific keywords")
with col2:
    st.markdown("• Try different phrasings")
with col3:
    st.markdown("• Be descriptive")