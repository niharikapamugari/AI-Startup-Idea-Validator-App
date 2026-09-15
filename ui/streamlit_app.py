"""
Streamlit UI
--------------
See prompts/orchestrator.md and the various tools/*.py docstrings for
the design notes behind each feature below. High-level map of what
lives where in THIS file:

- Professional theme: style_block.CUSTOM_CSS (single theme, no
  light/dark toggle - a deliberate choice, not an oversight).
- Ctrl+Enter submit: a small JS snippet reaching into the parent
  page from an embedded iframe (components.html).
- Cancellable validation with a live progress mascot: the pipeline
  runs in a background thread (ThreadPoolExecutor persisted in
  session_state); this script polls it once a second via
  time.sleep+st.rerun while showing tools/mascot.py's progress bar,
  and a Stop button sets a threading.Event the orchestrator checks
  between steps (see app/orchestrator.py's PipelineCancelled).
- Auth + per-user history: tools/auth.py + db/database.py's
  user_id-scoped queries. Anonymous users can validate and download
  their PDF, but History is login-only, and one user's history is
  never visible to another (enforced in the DB layer, not just the
  UI - see database.get_validation/delete_validation's user_id
  ownership checks).
- Multi-language input: tools/translator.py's translate_to_english -
  founders can type their idea in any language; it's silently
  normalized to English before the (English-tuned) pipeline runs, so
  every output is always in English. No language picker in the UI.
- Agent score charts: tools/score_charts.py.
"""

import sys
import os
import time
import threading
import concurrent.futures
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import streamlit.components.v1 as components
st.set_page_config(page_title="AI Startup Idea Validator", page_icon="\U0001F680", layout="wide")


from app.orchestrator import run_pipeline, PipelineCancelled
from tools.input_validator import validate_idea_text, check_sensitive_content, check_plausibility
from tools.location_data import ALL_COUNTRIES, COUNTRY_STATES, STATE_CITIES, get_gps_location
from tools.pdf_generator import build_report_pdf
from tools.mascot import render_progress_mascot, MASCOT_OPTIONS
from tools.translator import translate_to_english
from tools.score_charts import build_agent_score_figures
from tools import auth
from db import database
from tools import offline_cache
from style_block import CUSTOM_CSS

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

MAX_PIPELINE_SECONDS = 120  # safety net auto-cancel, on top of the manual Stop button

if "db_ready" not in st.session_state:
    st.session_state.db_ready = database.init_db()

if "_executor" not in st.session_state:
    # 4 workers, not 1-2: clicking Stop now abandons the current
    # background job immediately (see the polling block below)
    # rather than waiting for it to unwind, so a quick Stop -> edit ->
    # re-validate cycle can briefly have more than one job in flight.
    st.session_state["_executor"] = concurrent.futures.ThreadPoolExecutor(max_workers=4)

if "user" not in st.session_state:
    st.session_state["user"] = None

# ---------------------------------------------------------------------------
# Ctrl+Enter submits the idea (fixes: "if we press control+enter it
# automatically takes the input and starts validating"). Reaches into
# the PARENT page's DOM from this embedded iframe, since st.iframe
# content lives in its own iframe document.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Ctrl+Enter submits the idea.
# ---------------------------------------------------------------------------
components.html(
    """
    <script>
    (function() {
        const doc = window.parent.document;

        if (doc._ctrlEnterBound) {
            return;
        }

        doc._ctrlEnterBound = true;

        doc.addEventListener("keydown", function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                e.preventDefault();

                const buttons = doc.querySelectorAll("button");

                for (const btn of buttons) {
                    if (
                        btn.innerText &&
                        btn.innerText.trim() === "Validate Idea" &&
                        !btn.disabled
                    ) {
                        btn.click();
                        break;
                    }
                }
            }
        });
    })();
    </script>
    """,
    height=1,
)

st.markdown(
    """
    <div class="app-header">
        <h1>AI Startup Idea Validator</h1>
        <p>Multi-Agent Startup Validation Platform &mdash; from a raw idea to an investor-ready report in minutes.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: Account (register/login/logout) + per-user History + mascot pick
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Account")

    if st.session_state["user"] is None:
        auth_mode = st.radio(
            "auth_mode", ["Login", "Register", "Forgot Password"],
            horizontal=True, label_visibility="collapsed", key="auth_mode_radio",
        )
        if auth_mode == "Login":
            with st.form("login_form"):
                login_email = st.text_input("Email")
                login_password = st.text_input("Password", type="password")
                login_submitted = st.form_submit_button("Log In")
            if login_submitted:
                res = auth.login(login_email, login_password)
                if res["success"]:
                    st.session_state["user"] = res["user"]
                    st.rerun()
                else:
                    st.error(res["message"])
        elif auth_mode == "Register":
            with st.form("register_form"):
                reg_email = st.text_input("Email (this becomes your username)")
                reg_password = st.text_input("Password", type="password", help="At least 6 characters.")
                reg_security_question = st.selectbox("Security Question", auth.SECURITY_QUESTIONS)
                reg_security_answer = st.text_input(
                    "Your Answer",
                    help="Used to reset your password later if you forget it - not case sensitive.",
                )
                reg_submitted = st.form_submit_button("Create Account")
            if reg_submitted:
                res = auth.register(reg_email, reg_password, reg_security_question, reg_security_answer)
                if res["success"]:
                    st.session_state["user"] = res["user"]
                    st.rerun()
                else:
                    st.error(res["message"])
        else:  # Forgot Password - security question only, no email/SMTP involved
            if "sq_reset_stage" not in st.session_state:
                st.session_state["sq_reset_stage"] = "email"

            if st.session_state["sq_reset_stage"] == "email":
                st.caption("Enter your account email to see your security question.")
                with st.form("sq_email_form"):
                    sq_email = st.text_input("Your account email")
                    sq_email_submitted = st.form_submit_button("Continue")
                if sq_email_submitted:
                    res = auth.get_security_question(sq_email)
                    if res["success"]:
                        st.session_state["sq_reset_email"] = sq_email.strip()
                        st.session_state["sq_reset_question"] = res["question"]
                        st.session_state["sq_reset_stage"] = "answer"
                        st.rerun()
                    else:
                        st.error(res["message"])
            else:  # stage == "answer"
                st.info(st.session_state["sq_reset_question"])
                with st.form("sq_answer_form"):
                    sq_answer = st.text_input("Your answer")
                    sq_new_password = st.text_input("New password", type="password", help="At least 6 characters.")
                    sq_confirm_password = st.text_input("Confirm new password", type="password")
                    sq_answer_submitted = st.form_submit_button("Reset Password")
                if sq_answer_submitted:
                    if sq_new_password != sq_confirm_password:
                        st.error("Passwords don't match.")
                    else:
                        res = auth.reset_password_with_security_answer(
                            st.session_state["sq_reset_email"], sq_answer, sq_new_password
                        )
                        if res["success"]:
                            st.success(res["message"])
                            st.session_state.pop("sq_reset_stage", None)
                            st.session_state.pop("sq_reset_email", None)
                            st.session_state.pop("sq_reset_question", None)
                        else:
                            st.error(res["message"])
                if st.button("Start over", key="sq_start_over"):
                    st.session_state.pop("sq_reset_stage", None)
                    st.session_state.pop("sq_reset_email", None)
                    st.session_state.pop("sq_reset_question", None)
                    st.rerun()
        st.caption(
            "You can validate ideas and download your PDF (report + advisor chat) "
            "without an account. Log in to save and revisit your history."
        )
    else:
        st.success(f"Logged in as **{st.session_state['user']['email']}**")
        if st.button("Log Out"):
            st.session_state["user"] = None
            st.session_state.pop("sidebar_history_cache", None)
            st.session_state.pop("result", None)
            st.session_state.pop("current_validation_id", None)
            st.rerun()

        st.divider()
        st.subheader("Your History")
        user_id = st.session_state["user"]["id"]
        fresh = database.list_validations(user_id)
        if fresh:
            st.session_state["sidebar_history_cache"] = fresh
            offline_cache.save_history_cache(user_id, fresh)
            past = fresh
        else:
            # Postgres didn't return anything - fall back first to
            # this session's in-memory copy (fastest), then to the
            # on-disk cache (fixes: "if the user is offline, history
            # should be available anytime" - this survives even a
            # brand new session or an app restart while the DB is
            # down, not just a mid-session hiccup).
            cached = st.session_state.get("sidebar_history_cache")
            source_note = "this session"
            if not cached:
                cached = offline_cache.load_history_cache(user_id)
                source_note = "your last visit"
            if cached:
                st.caption(f"\u26A0\uFE0F Showing history saved from {source_note} (couldn't reach the database right now).")
                past = cached
            else:
                past = []

        if not past:
            st.caption("No validations saved yet.")
        else:
            for row in past[:15]:
                score = row.get("viability_score")
                label = f"{row['idea_name']} ({score if score is not None else 'N/A'}/100)"
                if st.button(label, key=f"sidebar_load_{row['id']}", width='stretch'):
                    full = database.get_validation(row["id"], user_id=user_id)
                    if full:
                        st.session_state["result"] = full
                        st.session_state["current_validation_id"] = row["id"]
                        from agents.conversational_advisor import load_advisor_history
                        load_advisor_history(row["id"], full)
                        st.rerun()

    st.divider()
    st.subheader("Progress Companion")
    st.session_state["mascot_choice"] = st.selectbox(
        "Pick your walking companion while validation runs", list(MASCOT_OPTIONS.keys()),
        index=list(MASCOT_OPTIONS.keys()).index(st.session_state.get("mascot_choice", "Walking Explorer")),
    )

BUDGET_RANGES = [
    "Bootstrap (very small budget)",
    "Seed stage (small funding raised)",
    "Series A ready (significant funding raised)",
    "Not sure yet",
]
TIMELINES = ["1 Month", "3 Months", "6 Months", "12 Months"]

pipeline_running = "pipeline_future" in st.session_state

col_a, col_b = st.columns(2)
with col_a:
    idea_text = st.text_area(
        "Describe your startup idea (2-3 lines):",
        height=100,
        help="Write a real, coherent business idea, in any language - results are always shown in English. Tip: Ctrl+Enter submits.",
        disabled=pipeline_running,
    )
    budget = st.selectbox("Expected Budget", BUDGET_RANGES, disabled=pipeline_running)

with col_b:
    st.write("**Location**")
    use_gps = st.checkbox("Use my current location (GPS)", disabled=pipeline_running)

    if use_gps:
        gps = get_gps_location()
        if gps:
            st.success(f"Detected: lat {gps['latitude']:.4f}, lon {gps['longitude']:.4f}")
            target_market = f"lat {gps['latitude']:.4f}, lon {gps['longitude']:.4f}"
            country = state_input = city_input = ""
        else:
            st.info("Waiting for browser location permission...")
            target_market = ""
            country = state_input = city_input = ""
    else:
        country = st.selectbox("Country (required)", ALL_COUNTRIES, disabled=pipeline_running)

        if country in COUNTRY_STATES:
            state_input = st.selectbox("State", COUNTRY_STATES[country], disabled=pipeline_running)
            state_input = "" if state_input in ("All States",) else state_input
        else:
            state_input = st.text_input("State (optional)", help="Full dropdown not available for this country yet - enter manually.", disabled=pipeline_running)

        if state_input and state_input in STATE_CITIES:
            city_input = st.selectbox("City / Town", STATE_CITIES[state_input], disabled=pipeline_running)
            city_input = "" if city_input in ("All Cities/Towns",) else city_input
        else:
            city_input = st.text_input("City / Town (optional)", disabled=pipeline_running)

        location_parts = [p for p in [city_input.strip() if city_input else "", state_input.strip() if state_input else "", country if country != "All Countries" else ""] if p]
        target_market = ", ".join(location_parts)

    timeline = st.selectbox("Launch Timeline", TIMELINES, disabled=pipeline_running)

validate_clicked = st.button("Validate Idea", type="primary", disabled=pipeline_running)


def _run_pipeline_job(idea_text_en, target_market, cancel_event):
    """Runs entirely inside the background thread. Every possible
    outcome (success, cooperative cancellation, or any other error)
    is captured into a plain dict here - future.result() in the main
    thread never needs to catch an exception, it just reads status."""
    try:
        result = run_pipeline(idea_text_en, target_market, cancel_event=cancel_event)
        return {"status": "done", "result": result}
    except PipelineCancelled:
        return {"status": "cancelled"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _qa_history_for_pdf(validation_id):
    """
    Gathers this idea's advisor chat as a flat list of {"role",
    "content"} for the PDF's "Q & A" section. Prefers the persisted
    (Postgres) history so a downloaded PDF matches what's saved to
    the founder's account; falls back to the live in-session chat
    for anonymous runs or ones that were never persisted. Any older,
    already-compressed turns are represented by their summary line
    first, then the remaining raw turns.
    """
    qa = []
    if validation_id:
        chat_summary = database.get_chat_summary(validation_id)
        if chat_summary:
            qa.append({"role": "assistant", "content": f"(Summary of earlier conversation) {chat_summary}"})
        qa.extend(database.get_advisor_messages(validation_id))
    if not qa:
        from agents.conversational_advisor import _HISTORY_KEY
        if _HISTORY_KEY in st.session_state:
            qa = [m for m in st.session_state[_HISTORY_KEY] if m.get("role") in ("user", "assistant")]
    return qa


if validate_clicked and not pipeline_running:
    input_check = validate_idea_text(idea_text)
    sensitive_check = check_sensitive_content(idea_text) if input_check["is_valid"] else {"is_sensitive": False}
    plausibility_check = check_plausibility(idea_text) if input_check["is_valid"] else {"is_plausible": True}

    if not input_check["is_valid"]:
        st.session_state["result"] = {"invalid": True, "reason": input_check["reason"]}
    elif sensitive_check.get("is_sensitive"):
        st.session_state["result"] = {"invalid": True, "reason": sensitive_check["reason"]}
    elif not plausibility_check.get("is_plausible", True):
        st.session_state["result"] = {"invalid": True, "reason": plausibility_check["reason"]}
    else:
        # Always normalize to English before running the pipeline, no
        # matter what language the founder wrote in - there's no
        # language picker anymore, this just silently guarantees an
        # English pipeline and English output every time. Skipped for
        # plain-ASCII input (already almost certainly English) to
        # avoid an unnecessary LLM call/latency on the common case.
        idea_text_en = idea_text
        if any(ord(ch) > 127 for ch in idea_text):
            with st.spinner("Preparing your idea..."):
                idea_text_en = translate_to_english(idea_text)

        cancel_event = threading.Event()
        future = st.session_state["_executor"].submit(_run_pipeline_job, idea_text_en, target_market, cancel_event)
        st.session_state["pipeline_future"] = future
        st.session_state["pipeline_cancel_event"] = cancel_event
        st.session_state["pipeline_start_time"] = time.time()
        st.session_state["pipeline_idea_text_en"] = idea_text_en
        st.session_state.pop("result", None)
        st.rerun()

# ---------------------------------------------------------------------------
# Live polling: shows the mascot + elapsed time, and a Stop button that
# sets the cancel_event (fixes: "if the user wants to stop the
# validating process to modify, it must work").
# ---------------------------------------------------------------------------
if "pipeline_future" in st.session_state:
    future = st.session_state["pipeline_future"]
    elapsed = time.time() - st.session_state["pipeline_start_time"]

    if not future.done():
        st.divider()
        st.info("Running the multi-agent validation pipeline...")
        st.markdown(
            render_progress_mascot(elapsed, st.session_state.get("mascot_choice", "Walking Explorer")),
            unsafe_allow_html=True,
        )
        if st.button("\u23F9 Stop & Modify Input"):
            # Set the flag so the background thread stops at its next
            # checkpoint - but don't wait for it. Fixes: "stopping
            # should let me modify the input right away," not after
            # the background job finishes unwinding. We simply stop
            # tracking/waiting on this future; it keeps running
            # harmlessly in its own thread and hits the cancellation
            # checkpoint within a few seconds on its own, but nothing
            # in the UI waits on it anymore, and its eventual result
            # (if any) is never read or saved.
            st.session_state["pipeline_cancel_event"].set()
            del st.session_state["pipeline_future"]
            del st.session_state["pipeline_cancel_event"]
            del st.session_state["pipeline_start_time"]
            st.session_state["result"] = {
                "invalid": True, "cancelled": True,
                "reason": "Validation stopped. You can modify your idea above and validate again.",
            }
            st.rerun()
        if elapsed > MAX_PIPELINE_SECONDS:
            st.session_state["pipeline_cancel_event"].set()
        time.sleep(1)
        st.rerun()
    else:
        job_output = future.result()
        del st.session_state["pipeline_future"]
        del st.session_state["pipeline_cancel_event"]
        del st.session_state["pipeline_start_time"]

        if job_output["status"] == "cancelled":
            st.session_state["result"] = {
                "invalid": True, "cancelled": True,
                "reason": "Validation stopped. You can modify your idea above and validate again.",
            }
        elif job_output["status"] == "error":
            st.session_state["result"] = {"invalid": True, "reason": job_output["error"]}
        else:
            result = job_output["result"]
            result["idea_text"] = st.session_state.get("pipeline_idea_text_en", result.get("idea_text", ""))
            meta = {"budget": budget, "timeline": timeline, "submitted_at": datetime.now()}
            result["_meta"] = {**meta, "submitted_at": meta["submitted_at"].strftime("%Y-%m-%d %H:%M")}

            user_id = st.session_state["user"]["id"] if st.session_state.get("user") else None
            new_id = database.save_validation(result, meta, user_id=user_id)
            st.session_state["current_validation_id"] = new_id
            from agents.conversational_advisor import reset_advisor_memory
            reset_advisor_memory()
            st.session_state["result"] = result

if "result" in st.session_state and "pipeline_future" not in st.session_state:
    result = st.session_state["result"]

    if result.get("invalid"):
        if result.get("cancelled"):
            st.warning(result.get("reason"))
        else:
            st.error(result.get("reason", "Please enter a valid input."))
            st.info("Please modify your idea in the box above and click **Validate Idea** again - nothing is locked, you can edit and resubmit right away.")
    else:
        from tools.floating_chat_icon import render_floating_assistant
        render_floating_assistant(result)

        # Results are always shown in English - the idea text itself is
        # silently normalized to English before the pipeline runs (see
        # the validate-click handler above), so there is nothing to
        # translate or ask about here.
        display_result = result

        st.divider()
        top_col1, top_col2 = st.columns([3, 1])
        with top_col1:
            st.subheader("Quick Summary")
            st.info(display_result.get("quick_summary", "Summary not available."))
        with top_col2:
            st.write("")
            st.write("")
            st.download_button(
                "Download Full Report (PDF)",
                data=build_report_pdf(display_result, qa_history=_qa_history_for_pdf(st.session_state.get("current_validation_id"))),
                file_name=f"{result.get('extracted', {}).get('idea_name', 'validation') or 'validation'}_report.pdf",
                mime="application/pdf",
                key="top_download_button",
            )

        if display_result.get("improvement_suggestions"):
            st.subheader("\U0001F4A1 Suggestions to Improve This Idea")
            for suggestion in display_result["improvement_suggestions"]:
                st.write(f"- {suggestion}")


        st.divider()

        tabs = st.tabs([
            "Idea", "Web Search", "Market Analysis", "Competitors",
            "SWOT & Risk", "MVP", "GTM Strategy", "Viability Score",
            "Agent Scores", "Insights", "Report", "Advisor Chat", "History",
        ])

        with tabs[0]:
            st.subheader("Structured Idea Output")
            st.json(result["extracted"])

        with tabs[1]:
            st.subheader("Live Market & Competitor Data (Web Search Agent)")
            st.caption(f"Search query used: {result['search_results'].get('query', '')}")
            if not result["search_results"].get("results"):
                st.info("No live results found for this query.")
            for r in result["search_results"]["results"]:
                st.markdown(f"**[{r['title']}]({r['url']})**")
                st.write(r["content"][:200] + "...")
                st.divider()

        with tabs[2]:
            st.subheader("Market Analysis (Deep Search)")
            market = result["market_analysis"]
            st.write(f"**TAM:** {market.get('tam_estimate', '')}")
            st.write(f"**SAM:** {market.get('sam_estimate', '')}")
            st.write(f"**SOM:** {market.get('som_estimate', '')}")
            st.write(f"**Growth Trend:** {market.get('growth_trend', '')}")
            st.write(f"**Customer Segments:** {', '.join(market.get('customer_segments', []))}")

        with tabs[3]:
            st.subheader("Competitor Analysis")
            for c in result["competitors"].get("competitors", []):
                st.write(f"**{c.get('name', '')}** — Strength: {c.get('strength', '')} | Weakness: {c.get('weakness', '')}")
            st.write(f"**Market Gap:** {result['competitors'].get('market_gap', '')}")

        with tabs[4]:
            st.subheader("SWOT & Risk Analysis")
            swot = result["swot"]
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Strengths:**")
                for s in swot.get("strengths", []):
                    st.write(f"- {s}")
                st.write("**Opportunities:**")
                for o in swot.get("opportunities", []):
                    st.write(f"- {o}")
            with col2:
                st.write("**Weaknesses:**")
                for w in swot.get("weaknesses", []):
                    st.write(f"- {w}")
                st.write("**Threats:**")
                for t in swot.get("threats", []):
                    st.write(f"- {t}")

        with tabs[5]:
            st.subheader("MVP Recommendation")
            for f in result["mvp"].get("mvp_features", []):
                st.write(f"**[{f.get('priority', '')}]** {f.get('feature', '')}")
            st.write(f"**Estimated Timeline:** {result['mvp'].get('estimated_timeline', '')}")

        with tabs[6]:
            st.subheader("Go-To-Market Strategy")
            gtm = result["gtm"]
            st.write(f"**Positioning:** {gtm.get('positioning_statement', '')}")
            st.write(f"**Channels:** {', '.join(gtm.get('marketing_channels', []))}")
            st.write(f"**Pricing:** {gtm.get('pricing_strategy', '')}")

        with tabs[7]:
            st.subheader("Viability Score")
            viability = result["viability_score"]
            st.metric(label="Overall Score", value=f"{viability['overall_score']}/100")
            st.write(f"**Verdict:** {viability['verdict']}")
            with st.expander("See score breakdown"):
                breakdown = viability["breakdown"]
                st.write(f"- Idea Clarity: {breakdown['idea_clarity']}/10")
                st.write(f"- Competition Density: {breakdown['competition_density']}/10")
                st.write(f"- Market Analysis: {breakdown['market_analysis']}/10")
                st.write(f"- SWOT/Risk: {breakdown['swot_risk']}/10")

        with tabs[8]:
            st.subheader("Agent Scores (Graphical Comparison, /100)")
            st.caption("Every value here is read directly from the same numbers shown in the other tabs - this is a visual, not a separate calculation.")
            bar_fig, radar_fig = build_agent_score_figures(result)
            st.plotly_chart(bar_fig, width='stretch')
            st.plotly_chart(radar_fig, width='stretch')

        with tabs[9]:
            st.subheader("Honest Mentor Take")
            st.info(display_result.get("honest_summary", result.get("honest_summary", "")))
            st.subheader("What You Might Be Missing")
            for question in display_result.get("blind_spots", result.get("blind_spots", [])):
                st.warning(question)
            st.subheader("Elevator Pitch")
            pitch = display_result.get("elevator_pitch", result.get("elevator_pitch", {}))
            st.write(f"**Pitch:** {pitch.get('elevator_pitch', '')}")
            st.write(f"**Tagline:** _{pitch.get('tagline', '')}_")
            st.subheader("Suggested Funding Paths")
            for suggestion in result["funding_suggestions"]:
                st.write(f"**{suggestion.get('funding_type')}** — {suggestion.get('reason')}")

        with tabs[10]:
            st.subheader("Full Validation Report")
            st.markdown(display_result["report"])
            st.download_button(
                "Download Report (PDF)",
                data=build_report_pdf(display_result, qa_history=_qa_history_for_pdf(st.session_state.get("current_validation_id"))),
                file_name=f"{result.get('extracted', {}).get('idea_name', 'validation') or 'validation'}_report.pdf",
                mime="application/pdf",
                key="bottom_download_button",
            )

        with tabs[11]:
            st.subheader("Ask a Follow-up Question")
            st.caption("This advisor remembers earlier questions in this conversation. Only questions related to your idea are saved to your history - off-topic chat is answered but not stored.")

            from agents.conversational_advisor import _HISTORY_KEY
            from tools.chat_ui import render_chat_history
            if _HISTORY_KEY in st.session_state:
                render_chat_history(st.session_state[_HISTORY_KEY])

            with st.form(key="advisor_form", clear_on_submit=True):
                followup = st.text_input("Ask the Conversational Advisor about this report:")
                submitted = st.form_submit_button("Ask Advisor")

            if submitted and followup.strip():
                from agents.conversational_advisor import ask_advisor
                with st.spinner("Thinking..."):
                    ask_advisor(followup, result, st.session_state.get("current_validation_id"))
                st.rerun()
            elif submitted:
                st.warning("Please type a question first.")

        with tabs[12]:
            st.subheader("History")
            user = st.session_state.get("user")
            if not user:
                st.info(
                    "Log in from the sidebar to save this (and every future) validation to your "
                    "personal history and revisit it later. You can still download this report's "
                    "PDF right now without logging in."
                )
            else:
                user_id = user["id"]
                past_validations = database.list_validations(user_id) if st.session_state.get("db_ready") else []
                if past_validations:
                    st.session_state["history_tab_cache"] = past_validations
                    offline_cache.save_history_cache(user_id, past_validations)
                else:
                    # Database unreachable right now (or was unreachable
                    # at startup) - fall back to this session's copy,
                    # then to the on-disk cache from a previous visit,
                    # so history is still viewable rather than just
                    # showing a dead end (fixes: "history should be
                    # available anytime, even offline").
                    cached = st.session_state.get("history_tab_cache")
                    source_note = "this session"
                    if not cached:
                        cached = offline_cache.load_history_cache(user_id)
                        source_note = "your last visit"
                    if cached:
                        st.caption(f"\u26A0\uFE0F Showing history saved from {source_note} (couldn't reach the database right now).")
                        past_validations = cached
                    else:
                        st.warning(
                            "Could not connect to the PostgreSQL database, and there's no "
                            "previously-cached history for your account yet. Check your DB "
                            "settings in `.env` (PG_HOST / PG_PORT / PG_DB / PG_USER / "
                            "PG_PASSWORD or DATABASE_URL)."
                        )

                if not past_validations:
                    st.info("No validations saved yet.")
                else:
                    for row in past_validations:
                        score = row.get("viability_score")
                        score_label = f"{score}/100" if score is not None else "N/A"
                        submitted_at = row.get("submitted_at")
                        if hasattr(submitted_at, "strftime"):
                            submitted_label = submitted_at.strftime("%Y-%m-%d %H:%M")
                        else:
                            # Came back from the on-disk offline cache as
                            # a plain ISO string rather than a live
                            # datetime object - reformat it the same way
                            # so cached and fresh rows look identical.
                            try:
                                submitted_label = datetime.fromisoformat(str(submitted_at)).strftime("%Y-%m-%d %H:%M")
                            except (ValueError, TypeError):
                                submitted_label = str(submitted_at or "")
                        with st.expander(f"{row['idea_name']} - Score: {score_label} - {submitted_label}"):
                            st.write(f"**Verdict:** {row.get('verdict', 'N/A')}")
                            st.write(f"**Budget:** {row.get('budget') or 'N/A'}")
                            st.write(f"**Timeline:** {row.get('timeline') or 'N/A'}")
                            st.write(f"**Location:** {row.get('target_market') or 'N/A'}")
                            if row.get("quick_summary"):
                                st.info(row["quick_summary"])

                            chat_summary = database.get_chat_summary(row["id"])
                            chat_messages = database.get_advisor_messages(row["id"])
                            qa_for_pdf = chat_messages
                            if chat_summary:
                                qa_for_pdf = [{"role": "assistant", "content": f"(Summary of earlier conversation) {chat_summary}"}] + chat_messages
                            if chat_summary or chat_messages:
                                with st.expander("Advisor Chat History", expanded=False):
                                    if chat_summary:
                                        st.caption("Summary of earlier conversation:")
                                        st.write(chat_summary)
                                    from tools.chat_ui import render_chat_history
                                    render_chat_history(chat_messages)

                            past_full = database.get_validation(row["id"], user_id=user_id)
                            btn_col1, btn_col2, btn_col3 = st.columns(3)
                            with btn_col1:
                                if past_full:
                                    st.download_button(
                                        "Download PDF",
                                        data=build_report_pdf(past_full, qa_history=qa_for_pdf),
                                        file_name=f"{row['idea_name']}_report.pdf",
                                        mime="application/pdf",
                                        key=f"history_pdf_{row['id']}",
                                    )
                            with btn_col2:
                                if st.button("Continue Chatting", key=f"history_continue_{row['id']}"):
                                    if past_full:
                                        st.session_state["result"] = past_full
                                        st.session_state["current_validation_id"] = row["id"]
                                        from agents.conversational_advisor import load_advisor_history
                                        load_advisor_history(row["id"], past_full)
                                        st.rerun()
                            with btn_col3:
                                if st.button("Delete", key=f"history_delete_{row['id']}"):
                                    database.delete_validation(row["id"], user_id=user_id)
                                    st.rerun()

st.markdown(
    '<div class="app-footer">AI Startup Idea Validator &middot; Multi-Agent Validation Platform</div>',
    unsafe_allow_html=True,
)
