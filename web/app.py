import streamlit as st
import sys
import os

if sys.platform == 'linux':
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from courserag.core.rag_system import RAGSystem

PREPROCESSED_DATA_DIR = "data/processed"

st.set_page_config(
    page_title="CourseGPT",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

def load_css(file_name):
    css_file_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(css_file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('static/styles.css')

@st.cache_resource
def initialize_rag_system():
    if "OPENAI_API_KEY" not in os.environ:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.stop()

    try:
        rag_system = RAGSystem(PREPROCESSED_DATA_DIR)
        rag_system.initialize()
        return rag_system
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        st.stop()

st.markdown("""
<div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
    <div style="text-align: center;">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 10px;">
            <svg width="50" height="42" viewBox="0 0 408 338" style="margin-right: 5px;">
                <g transform="translate(0,338) scale(0.1,-0.1)" fill="#1e3a5f" stroke="none">
                    <path d="M500 2383 c0 -380 5 -720 10 -773 15 -142 45 -254 99 -365 163 -338 477 -534 905 -565 398 -28 712 78 931 312 153 164 247 379 262 600 l6 86 111 -5 c92 -4 127 -2 200 16 270 65 446 268 446 516 0 80 -5 87 -42 53 -120 -112 -305 -134 -583 -70 l-120 27 -3 423 -2 422 -195 0 -195 0 0 -348 c0 -192 -2 -351 -5 -354 -3 -3 -32 3 -65 13 -32 10 -105 23 -163 30 -298 35 -541 -104 -679 -389 -22 -46 -38 -85 -35 -88 3 -3 38 5 78 18 66 20 94 23 249 23 150 0 187 -4 260 -23 114 -31 189 -57 284 -100 l79 -36 -7 -116 c-22 -391 -184 -624 -476 -686 -162 -34 -356 4 -497 99 -139 93 -232 231 -279 414 l-24 96 0 723 0 724 -275 0 -275 0 0 -677z"/>
                </g>
            </svg>
            <div style="color: #1e3a5f; font-size: 24px; font-weight: semibold;">
                University of Antwerp
            </div>
        </div>
        <h1 style='text-align: center; margin: 0;'>CourseGPT</h1>
        <div style="color: #666; font-size: 14px; margin-top: 5px;">
            Current Trends in Data Science and Artificial Intelligence
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #c8102e; margin-bottom: 30px;">
    <p style="margin: 0; color: #333;">
        Welcome to your interactive study companion! Ask any question about the course material 
        (papers, slides, videos, announcements), and I'll provide answers from the course content, 
        along with a comparison from a general-purpose AI to show you how much better I am. 😉
    </p>
</div>
""", unsafe_allow_html=True)

rag_system = initialize_rag_system()

user_question = st.text_area(
    "What would you like to know about the course?",
    placeholder="e.g., What were the key findings regarding LLaMA3 quantization in the course material?",
    height=100,
    key="user_question_input"
)

if st.button("Ask CourseGPT", key="ask_button"):
    if user_question:
        st.markdown("---")

        st.subheader("Your Question:")
        st.write(user_question)

        st.subheader("CourseGPT (RAG) Answer:")
        with st.spinner("CourseGPT is thinking... (Retrieving from course material)"):
            comparison = rag_system.compare_with_normal_gpt(user_question)
            rag_answer = comparison["rag_answer"]
            sources = comparison.get("sources", [])
            st.info(rag_answer)
            
            if sources:
                st.caption(f"**Sources:** {', '.join(sources)}")

        st.subheader("General AI Answer:")
        with st.spinner("General AI is thinking..."):
            normal_gpt_answer = comparison["normal_gpt_answer"]
            st.warning(normal_gpt_answer)
    else:
        st.warning("Please enter a question to get started!")

st.markdown("---")
st.markdown("Developed by Achraf and Xhejms for CTR Assignment 3.")
st.markdown("[Link to GitHub Repository](https://github.com/AchrafYndz/CTD-RAG-System)")
