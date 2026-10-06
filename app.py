import streamlit as st
import ollama
import chromadb
import json
from datetime import date

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="AI Student Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI-Based Student Assistant")
st.caption("Your AI-powered personal study companion")


# =====================================================
# AI MODEL + CHROMADB
# =====================================================

embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="student_notes"
)


# =====================================================
# SESSION STATE
# =====================================================

if "quiz" not in st.session_state:
    st.session_state.quiz = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "study_plan" not in st.session_state:
    st.session_state.study_plan = None

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("📚 Student Assistant")

    page = st.radio(
        "Choose Feature",
        [
            "🏠 Dashboard",
            "💬 AI Doubt Solver",
            "📝 AI Quiz",
            "📚 Study Planner",
            "🕘 Chat History"
        ]
    )


# =====================================================
# PDF UPLOAD
# =====================================================

st.sidebar.divider()

pdf = st.sidebar.file_uploader(
    "📄 Upload Study PDF",
    type=["pdf"]
)


if pdf:

    reader = PdfReader(pdf)

    text = ""

    for page_data in reader.pages:
        text += page_data.extract_text() or ""

    st.session_state.pdf_text = text

    chunks = [
        text[i:i + 1000]
        for i in range(0, len(text), 1000)
    ]

    if chunks:

        embeddings = embedder.encode(
            chunks
        ).tolist()

        ids = [
            f"{pdf.name}_{i}"
            for i in range(len(chunks))
        ]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings
        )

        st.sidebar.success(
            "✅ PDF uploaded!"
        )


# =====================================================
# DASHBOARD
# =====================================================

if page == "🏠 Dashboard":

    st.header("🏠 Student Dashboard")

    st.write(
        "Welcome! Use the features below to study smarter."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📄 PDF",
            "Uploaded" if pdf else "Not Uploaded"
        )

    with col2:
        st.metric(
            "📝 Quiz",
            "Ready" if st.session_state.quiz else "Not Generated"
        )

    with col3:
        st.metric(
            "💬 Chat",
            len(st.session_state.chat_history)
        )

    st.divider()

    st.info(
        "📌 Upload your study PDF from the sidebar "
        "to use AI Doubt Solver and AI Quiz."
    )


# =====================================================
# AI DOUBT SOLVER
# =====================================================

elif page == "💬 AI Doubt Solver":

    st.header("💬 AI Doubt Solver")

    if not pdf:

        st.warning(
            "⚠️ Please upload a PDF first."
        )

    question = st.text_input(
        "Ask your question"
    )

    if st.button(
        "🤖 Ask AI",
        key="ask_ai"
    ):

        if not pdf:

            st.warning(
                "Please upload a PDF first."
            )

        elif not question:

            st.warning(
                "Please enter a question."
            )

        else:

            query_embedding = embedder.encode(
                [question]
            ).tolist()

            results = collection.query(
                query_embeddings=query_embedding,
                n_results=3
            )

            documents = results["documents"][0]

            context = "\n\n".join(documents)

            prompt = f"""
You are an AI Student Assistant.

Answer the question using the study material.

Study Material:
{context}

Question:
{question}

Give a simple and accurate answer.
"""

            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response[
                "message"
            ]["content"]

            st.subheader("🤖 AI Answer")

            st.write(answer)

            # Save chat
            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer
                }
            )


# =====================================================
# AI QUIZ GENERATOR + SCORE
# =====================================================

elif page == "📝 AI Quiz":

    st.header("📝 AI Quiz Generator")

    if not pdf:

        st.warning(
            "⚠️ Upload a PDF first."
        )

    else:

        st.write(
            "Generate 5 MCQs from your study material."
        )

        if st.button(
            "📝 Generate Quiz",
            key="generate_quiz"
        ):

            quiz_prompt = f"""
Create exactly 5 multiple-choice questions
from the study material below.

Return ONLY valid JSON.

Format:

[
  {{
    "question": "Question",
    "options": ["A", "B", "C", "D"],
    "answer": "A"
  }}
]

Study Material:

{st.session_state.pdf_text[:12000]}
"""

            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": quiz_prompt
                    }
                ]
            )

            try:

                quiz_text = response[
                    "message"
                ]["content"]

                quiz_text = quiz_text.replace(
                    "```json", ""
                ).replace(
                    "```", ""
                ).strip()

                st.session_state.quiz = json.loads(
                    quiz_text
                )

                st.success(
                    "✅ Quiz generated successfully!"
                )

            except:

                st.error(
                    "❌ Quiz generation failed. "
                    "Please click Generate Quiz again."
                )


        # -----------------------------------------
        # DISPLAY QUIZ
        # -----------------------------------------

        if st.session_state.quiz:

            st.divider()

            st.subheader("📚 Answer the Questions")

            answers = {}

            for i, q in enumerate(
                st.session_state.quiz
            ):

                st.write(
                    f"### Q{i + 1}. {q['question']}"
                )

                answers[i] = st.radio(
                    "Choose answer:",
                    q["options"],
                    key=f"quiz_question_{i}"
                )


            # -----------------------------------------
            # SCORE
            # -----------------------------------------

            if st.button(
                "🏆 Submit Quiz",
                key="submit_quiz"
            ):

                score = 0

                for i, q in enumerate(
                    st.session_state.quiz
                ):

                    if answers[i] == q["answer"]:

                        score += 1

                total = len(
                    st.session_state.quiz
                )

                percentage = (
                    score / total
                ) * 100

                st.success(
                    f"🎉 Your Score: {score}/{total}"
                )

                st.info(
                    f"📊 Percentage: {percentage:.0f}%"
                )


# =====================================================
# STUDY PLANNER
# =====================================================

elif page == "📚 Study Planner":

    st.header("📚 AI Study Planner")

    subjects = st.text_input(
        "📖 Enter subjects",
        placeholder="Python, DSA, Signals, Digital Electronics"
    )

    exam_date = st.date_input(
        "📅 Exam Date",
        min_value=date.today()
    )

    daily_hours = st.number_input(
        "⏰ Study hours per day",
        min_value=1,
        max_value=12,
        value=3
    )


    if st.button(
        "📚 Generate Study Plan",
        key="generate_plan"
    ):

        if not subjects:

            st.warning(
                "Please enter your subjects."
            )

        else:

            prompt = f"""
Create a practical study timetable for a college student.

Subjects:
{subjects}

Exam Date:
{exam_date}

Daily Study Hours:
{daily_hours}

Create a day-by-day study plan.

Include:
- Subject
- Topic
- Study time
- Revision
- Breaks

Keep it simple and realistic.
"""

            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            st.session_state.study_plan = response[
                "message"
            ]["content"]


    if st.session_state.study_plan:

        st.subheader("📅 Your Study Plan")

        st.write(
            st.session_state.study_plan
        )


# =====================================================
# CHAT HISTORY
# =====================================================

elif page == "🕘 Chat History":

    st.header("🕘 Chat History")

    if not st.session_state.chat_history:

        st.info(
            "No chat history yet."
        )

    else:

        for i, chat in enumerate(
            st.session_state.chat_history
        ):

            st.subheader(
                f"Question {i + 1}"
            )

            st.write(
                "❓ " + chat["question"]
            )

            st.write(
                "🤖 " + chat["answer"]
            )

            st.divider()


        if st.button(
            "🗑️ Clear Chat History",
            key="clear_history"
        ):

            st.session_state.chat_history = []

            st.success(
                "Chat history cleared!"
            )