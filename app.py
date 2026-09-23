"""
Streamlit demo.

Point of this UI: make the agent's *process* visible, not just its answer. The
whole selling point of a tool-using agent is that you can watch it reason, call
SymPy, check the result, and then commit — so the trace is front and centre.

Run: streamlit run app.py
"""

import uuid

import streamlit as st

from src.agent import solve
from src.config import check_config

st.set_page_config(page_title="MathSolver-Agent", page_icon="🧮")

st.title("🧮 MathSolver-Agent")
st.caption("A tool-augmented LLM that solves JEE-level maths with SymPy — and shows its work.")

# One conversation per browser session: the thread_id is what the agent's
# checkpointer keys its saved messages on, so follow-ups like "now do the same
# for x^3" still see the previous question.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []

if st.button("New conversation"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []

for past_q, past_a in st.session_state.history:
    st.markdown(f"**Q:** {past_q}  \n**A:** {past_a}")

# A couple of examples so the demo isn't a blank box.
examples = {
    "— pick an example —": "",
    "Integral": "Evaluate the definite integral of x^2 from 0 to 3.",
    "Roots": "Find the sum of the roots of x^2 - 5x + 6 = 0.",
    "Limit": "Evaluate the limit of sin(x)/x as x approaches 0.",
}
choice = st.selectbox("Try an example", list(examples.keys()))

problem = st.text_area(
    "Enter a maths problem",
    value=examples[choice],
    height=120,
    placeholder="e.g. Find the area under y = x^2 between x = 0 and x = 2.",
)

if st.button("Solve", type="primary"):
    if not problem.strip():
        st.warning("Type a problem first.")
        st.stop()

    try:
        check_config()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    with st.spinner("Thinking and computing..."):
        result = solve(problem, thread_id=st.session_state.thread_id)

    st.session_state.history.append((problem, result.answer))
    st.success(f"**Answer:** {result.answer}")
    st.caption(f"Solved in {result.steps_taken} step(s) · "
               f"{result.tokens_used} tokens · stop reason: {result.stopped_reason}")

    # Render the trace so you can see exactly which tools fired and what they returned.
    st.subheader("Reasoning trace")
    for entry in result.trace:
        if entry["type"] == "tool_call":
            with st.expander(f"🔧 {entry['name']}({entry['args']})"):
                st.code(entry["observation"])
        elif entry["type"] == "loop_stop":
            st.warning(f"🔁 Loop detected on {entry['name']} — asked the model to change approach.")
        elif entry["type"] == "thought":
            st.markdown(f"💭 {entry['content']}")
        elif entry["type"] == "final_answer":
            st.markdown(f"✅ **final_answer:** {entry['content']}")
