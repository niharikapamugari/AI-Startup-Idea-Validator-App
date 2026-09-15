"""
Conversational Advisor Agent (with real LLM conversation context)
--------------------------------------------------------------------------
Implements the context/history concept exactly as described: without
memory, the model has to be re-told everything on every question
(like meeting someone for the first time, every time). The fix is
to maintain a growing list of message dicts (history = [{"role":
"user"/"assistant", "content": "..."}]) and pass the FULL list to
the model on every call, so it has real memory of the conversation
so far.

BUGFIX: the advisor was silently failing to answer. The previous
version ran the ENTIRE function - including all st.session_state
reads/writes - inside a worker thread via run_with_timeout(). Only
the main Streamlit script thread has a valid ScriptRunContext;
touching st.session_state from a background thread is unreliable
(it can silently no-op or raise), so the history was never actually
being populated with the model's answer, which is why the advisor
looked like it "wasn't giving an answer". The fix: keep every
st.session_state read/write on the main thread, and ONLY put the
actual network call (the slow, potentially-hanging part) inside the
timeout-guarded worker thread.

PERSISTENCE + RELEVANCE FILTERING (chat history in Postgres):
Every question is still answered live in-session regardless of
topic - the advisor never refuses to answer. But only questions
that are actually RELATED to the founder's startup idea/report get
written to the `advisor_messages` table (see db/database.py). This
is a deliberate design choice: a founder asking "what's the weather"
or "tell me a joke" shouldn't pollute the permanent record of
mentorship advice tied to their idea.

SUMMARIZATION (keeping the DB history from growing unbounded):
Once a validation's saved conversation passes _SUMMARIZE_THRESHOLD
messages, the OLDER messages get compressed by the LLM into a short
running `chat_summary` on the validated_ideas row, and those raw
rows are deleted - only the most recent _KEEP_RAW_MESSAGES stay as
full text. This keeps the persisted history both complete-in-spirit
(nothing important is lost, it's condensed) and bounded in size.

The UI call signature is: ask_advisor(question, result, validation_id).
validation_id is the row id returned by db.database.save_validation()
for the CURRENT session's validation - pass None if it wasn't saved
(e.g. DB unreachable), in which case chat is still answered live via
st.session_state, it just won't be persisted.
"""

import streamlit as st
from agents.idea_extraction_agent import client
from app.config import MODEL_NAME
from tools.timeout_utils import run_with_timeout
from db import database

_HISTORY_KEY = "advisor_llm_history"

# Once a validation's saved chat reaches this many raw messages,
# compress the older ones into chat_summary.
_SUMMARIZE_THRESHOLD = 10
# How many of the most recent raw messages to keep un-compressed.
_KEEP_RAW_MESSAGES = 4


def _call_llm_only(history: list) -> str:
    """
    The ONLY part of the advisor flow allowed to run inside the
    timeout-guarded worker thread - a plain network call with no
    Streamlit session_state access at all.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=history,
        temperature=0.0,  # deterministic per reviewer instruction
    )
    return response.choices[0].message.content.strip()


def _is_relevant_to_idea(question: str, state_dict: dict) -> bool:
    """
    Deterministic-as-possible relevance gate: is this question
    actually about the founder's startup idea/report, or off-topic
    chit-chat that shouldn't be permanently saved? Runs as its own
    short, cheap LLM call (temperature 0, single-word answer) rather
    than folding it into the main answer, so a bad/ambiguous
    classification never affects the actual answer given to the
    founder - it only affects whether the turn gets persisted.
    Fails OPEN (treats as relevant) on any error, since losing a
    borderline-relevant question from history is worse than
    occasionally saving one that wasn't.
    """
    idea = state_dict.get("extracted", {})
    context = (
        f"Idea: {idea.get('idea_name', '')}\n"
        f"Problem: {idea.get('problem', '')}\n"
        f"Solution: {idea.get('solution', '')}\n"
        f"Industry: {idea.get('industry', '')}"
    )
    prompt = f"""
You are filtering chat history for a startup-idea-validation advisor.

Startup context:
{context}

Founder's question: "{question}"

Is this question genuinely about the founder's startup idea, its
market, competitors, strategy, execution, funding, or the validation
report itself - i.e. something worth permanently saving as part of
this idea's advisory history? Or is it off-topic small talk /
unrelated to this idea (e.g. general trivia, unrelated topics,
testing the chatbot)?

Answer with exactly one word: RELEVANT or UNRELATED.
"""
    try:
        answer = run_with_timeout(
            _call_llm_only, args=([{"role": "user", "content": prompt}],), timeout_seconds=8.0
        )
        return "unrelated" not in answer.strip().lower()
    except Exception:
        return True  # fail open - don't silently lose history on a timeout/error


def _build_system_context(state_dict: dict, chat_summary: str = None) -> dict:
    """The first message in the history - establishes the report
    context once, so it doesn't need to be repeated on every turn.
    If a prior compressed chat_summary exists (from an earlier
    session, loaded back from Postgres), it's included too, so the
    advisor keeps continuity across page refreshes/new sessions."""
    context = f"""
You are a startup mentor. The founder received this validation report:

Idea: {state_dict.get('extracted', {}).get('idea_name', '')}
Viability Score: {state_dict.get('viability_score', {}).get('overall_score', '')}/100
Honest Summary: {state_dict.get('honest_summary', '')}
Blind Spots: {state_dict.get('blind_spots', [])}
"""
    if chat_summary:
        context += f"\nSummary of earlier conversation with this founder:\n{chat_summary}\n"
    context += """
Answer the founder's follow-up questions directly and concisely,
using this report as context. Remember earlier questions and
answers in this conversation when answering new ones - do not ask
the founder to repeat information already given.
"""
    return {"role": "system", "content": context}


def _summarize_for_storage(existing_summary: str, older_messages: list) -> str:
    """Compresses a batch of older Q&A turns (plus any existing
    summary) into one short running summary for permanent storage.
    Deterministic in intent (temperature 0) though not byte-for-byte
    reproducible, since it's an LLM call - that's acceptable here
    since this is a storage aid, not a validated business fact."""
    transcript = "\n".join(f"{m['role'].title()}: {m['content']}" for m in older_messages)
    prompt = f"""
Summarize this founder's advisory conversation into a short list of
the key questions asked and the key advice/answers given. Keep only
substantive points - specific numbers, decisions, or advice - not
pleasantries. Under 150 words. Plain text, no markdown headers.

{existing_summary_text}

Conversation to fold in:
{transcript}
"""
    try:
        return run_with_timeout(
            _call_llm_only, args=([{"role": "user", "content": prompt}],), timeout_seconds=15.0
        )
    except Exception:
        # Fall back to keeping the existing summary unchanged rather
        # than losing it or blocking the chat turn that triggered this.
        return existing_summary or ""


def _maybe_compress_history(validation_id: int):
    """Checks whether this validation's saved chat has grown past the
    threshold, and if so, folds the older messages into chat_summary
    and prunes the raw rows. Called after every persisted turn."""
    if not validation_id:
        return
    messages = database.get_advisor_messages(validation_id)
    if len(messages) <= _SUMMARIZE_THRESHOLD:
        return

    to_summarize = messages[:-_KEEP_RAW_MESSAGES] if _KEEP_RAW_MESSAGES else messages
    to_keep = messages[-_KEEP_RAW_MESSAGES:] if _KEEP_RAW_MESSAGES else []
    if not to_summarize:
        return

    existing_summary = database.get_chat_summary(validation_id) or ""
    new_summary = _summarize_for_storage(existing_summary, to_summarize)
    database.compress_advisor_messages(
        validation_id, new_summary, keep_message_ids=[m["id"] for m in to_keep]
    )


def ask_advisor(question: str, state_dict: dict, validation_id: int = None) -> str:
    # Initialize + mutate history on the MAIN thread only.
    if _HISTORY_KEY not in st.session_state:
        chat_summary = database.get_chat_summary(validation_id) if validation_id else None
        st.session_state[_HISTORY_KEY] = [_build_system_context(state_dict, chat_summary)]

    history = st.session_state[_HISTORY_KEY]
    history.append({"role": "user", "content": question})

    try:
        # Only the network call runs in the timeout-guarded thread -
        # it receives a plain list (a snapshot), not st.session_state
        # itself, so there's no cross-thread session access at all.
        answer = run_with_timeout(
            _call_llm_only, args=(list(history),), timeout_seconds=20.0
        )
    except Exception as e:
        answer = f"Sorry, I couldn't get an answer right now (possible connection issue: {e}). Please try again."
        # Don't leave a dangling user turn with no reply in history.
        history.append({"role": "assistant", "content": answer})
        st.session_state[_HISTORY_KEY] = history
        return answer

    # Save the model's answer into history too, so the NEXT question
    # has access to it - this is what makes multi-turn memory work
    history.append({"role": "assistant", "content": answer})
    st.session_state[_HISTORY_KEY] = history

    # Persist to Postgres ONLY if this question is actually related to
    # the idea - off-topic chit-chat is answered above but never saved.
    if validation_id and _is_relevant_to_idea(question, state_dict):
        database.save_advisor_message(validation_id, "user", question)
        database.save_advisor_message(validation_id, "assistant", answer)
        _maybe_compress_history(validation_id)

    return answer


def load_advisor_history(validation_id: int, state_dict: dict):
    """
    Loads a PAST validation's saved chat (chat_summary + any
    remaining raw messages from Postgres) into the live in-session
    advisor history, so the founder can pick up a conversation from
    History and keep going with full context (fixes: "continue
    chatting with AI - the user can also see the history agent
    solutions"). Call this right before switching state["result"] to
    the past validation and rerunning.
    """
    chat_summary = database.get_chat_summary(validation_id) if validation_id else None
    history = [_build_system_context(state_dict, chat_summary)]
    if validation_id:
        for msg in database.get_advisor_messages(validation_id):
            history.append({"role": msg["role"], "content": msg["content"]})
    st.session_state[_HISTORY_KEY] = history


def reset_advisor_memory():
    """Call this when a NEW idea is validated, so the advisor doesn't
    carry over context from a previous, unrelated idea."""
    if _HISTORY_KEY in st.session_state:
        del st.session_state[_HISTORY_KEY]
