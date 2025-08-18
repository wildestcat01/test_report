# app.py
import os
import io
import re
import time
import requests
import pandas as pd
import streamlit as st
from typing import Dict, Any, Tuple, List, Optional

# ---------- .env ----------
from dotenv import load_dotenv
load_dotenv()

# ---------- ReportLab (PDF) ----------
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, Image as RLImage
)

# ---------- Matplotlib (math rendering) ----------
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def get_cfg(key: str, default: str = "") -> str:
    try:
        # Streamlit Cloud path
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    # Local dev (.env) fallback
    return os.getenv(key, default).strip()

# =========================
# Config from .env
# =========================
REDASH_URL      = get_cfg("REDASH_URL")
REDASH_API_KEY  = get_cfg("REDASH_API_KEY")
REDASH_QUERY_ID = get_cfg("REDASH_QUERY_ID")
TEST_API_URL    = get_cfg("TEST_API_URL")
TEACHER_NAME    = get_cfg("TEACHER_NAME", "Teacher")

# =========================
# Utilities
# =========================
class RedashError(RuntimeError):
    pass

def _require_json(r: requests.Response) -> Dict[str, Any]:
    ctype = r.headers.get("Content-Type", "")
    if "application/json" not in ctype.lower():
        snippet = (r.text or "")[:500]
        raise RedashError(f"Expected JSON, got {ctype}. Body snippet:\n{snippet}")
    return r.json()

# =========================
# 1) Redash: fetch token + assessment_id
# =========================
def redash_fetch_token_and_assessment(student_id: str) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    if not REDASH_URL or not REDASH_API_KEY or not REDASH_QUERY_ID:
        raise RedashError("Missing REDASH_URL / REDASH_API_KEY / REDASH_QUERY_ID in .env")

    base = REDASH_URL.rstrip("/")
    headers = {"Authorization": f"Key {REDASH_API_KEY}"}

    submit_url = f"{base}/api/queries/{REDASH_QUERY_ID}/results"
    payload = {"parameters": {"student_id": student_id}, "max_age": 0}
    r = requests.post(submit_url, headers=headers, json=payload, timeout=30)
    data = _require_json(r)
    job = data.get("job")
    if not job:
        rows = data.get("query_result", {}).get("data", {}).get("rows", [])
        if not rows:
            raise RedashError("No job and no rows in Redash response.")
        row = rows[0]
        return row.get("token"), row.get("assessment_id"), data

    job_id = job["id"]
    start = time.time()
    while True:
        jr = requests.get(f"{base}/api/jobs/{job_id}", headers=headers, timeout=30)
        j = _require_json(jr)
        status = j.get("job", {}).get("status")
        if status == 3:
            result_id = j["job"].get("query_result_id")
            if not result_id:
                raise RedashError("Job completed but no query_result_id.")
            break
        if status == 4:
            raise RedashError(f"Redash job failed: {j}")
        if time.time() - start > 90:
            raise RedashError("Timed out waiting for Redash.")
        time.sleep(1)

    rr = requests.get(f"{base}/api/query_results/{result_id}", headers=headers, timeout=30)
    result_json = _require_json(rr)
    rows = result_json.get("query_result", {}).get("data", {}).get("rows", [])
    if not rows:
        return None, None, result_json
    row = rows[0]
    return row.get("token"), row.get("assessment_id"), result_json

# =========================
# 2) Test API: GET with headers (your curl)
# =========================
def call_test_api(token: str, assessment_id: str, student_id: str) -> dict:
    if not TEST_API_URL:
        raise RuntimeError("Missing TEST_API_URL in .env")
    url = f"{TEST_API_URL.rstrip('/')}/assessment/{assessment_id}/report/detail"
    headers = {
        "Content-Type": "application/json",
        "user-id": str(student_id),
        "token": token,
        "user-type": "student"
    }
    r = requests.get(url, headers=headers, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Test API error {r.status_code}: {r.text}")
    return r.json()

# =========================
# 3) Normalizer: API schema → summary + table
# =========================
def _strip_html(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = (s.replace("&nbsp;", " ")
           .replace("&lt;", "<")
           .replace("&gt;", ">")
           .replace("&amp;", "&"))
    return re.sub(r"[ \t]+", " ", s).strip()

# ---------- Math helpers (ONLY $…$ or $$…$$ are math) ----------
_MATH_SPAN_RE = re.compile(r"(\${1,2})([^$]+?)\1")  # matches $...$ or $$...$$

def _balance_dollars_whole(s: str) -> Tuple[str, bool]:
    """If # of $ is odd, remove all $ and mark as not-math to avoid crashes."""
    if s.count("$") % 2 != 0:
        return s.replace("$", ""), False
    return s, True

def _sanitize_math_inner(inner: str) -> str:
    """Sanitize content INSIDE a math span for matplotlib mathtext."""
    # Map unsupported aliases
    inner = inner.replace(r"\cosec", r"\csc")
    inner = inner.replace(r"\arccosec", r"\csc")
    inner = inner.replace(r"\arcsec", r"\sec")
    inner = inner.replace(r"\tan^-1", r"\arctan")
    inner = inner.replace(r"\sin^-1", r"\arcsin")
    inner = inner.replace(r"\cos^-1", r"\arccos")

    # Normalize \sqrt usage:
    # \sqrt(2) -> \sqrt{2}
    inner = re.sub(r"\\sqrt\s*\(\s*([^)]+?)\s*\)", r"\\sqrt{\1}", inner)
    # \sqrt2, \sqrtx -> \sqrt{2}, \sqrt{x}
    inner = re.sub(r"\\sqrt(?!\{)\s*([A-Za-z0-9])", r"\\sqrt{\1}", inner)

    # Convert simple numeric a/b → \frac{a}{b}
    inner = re.sub(r"(?<!\\)(?<!\w)\b([0-9]{1,3})\s*/\s*([0-9]{1,3})\b(?!\w)", r"\\frac{\1}{\2}", inner)

    # Normalize already-escaped ampersands like \\& -> \&
    inner = re.sub(r"\\\\&", r"\\&", inner)

    # Escape special chars that break mathtext (incl. underscore and unescaped ampersand)
    # Also escape any literal $ inside math spans, which would prematurely terminate parsing.
    inner = inner.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_").replace("$", r"\$")

    # Fix dangling or malformed caret usages:
    # 1) Remove a caret at end or before end-of-math with only spaces after it
    inner = re.sub(r"\^\s*$", "", inner)
    # 2) If caret is followed by a space or a non-valid token, convert to empty superscript to avoid crash
    inner = re.sub(r"\^(?=\s|[^{\w\\])", r"^{ }", inner)

    return inner

def _prepare_candidate_text(text: str) -> Tuple[str, bool]:
    """
    Only treat $…$ or $$…$$ as math. Sanitize their inners. Return (candidate, has_math_spans).
    If dollars are unbalanced, we remove $ and return has_math_spans=False.
    """
    text = text or ""
    text, ok = _balance_dollars_whole(text)
    if not ok:
        return text, False

    has_math = False
    def repl(m):
        nonlocal has_math
        delim = m.group(1)     # $ or $$
        inner = m.group(2)
        has_math = True
        inner = _sanitize_math_inner(inner)
        # Normalize to single-$ for mathtext; $$…$$ becomes $…$
        return f"${inner}$"

    candidate = _MATH_SPAN_RE.sub(repl, text)
    # Escape any leftover stray dollar signs outside valid math spans
    candidate = candidate.replace("$", r"\$")
    return candidate, has_math

def normalize_test_json_sparkl(resp: Dict[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    data = resp.get("data", {}) or {}
    summary = {
        "test_name": data.get("test_name") or data.get("test_title") or "Test",
        "total_questions": data.get("total_questions"),
        "total_marks": data.get("total_marks"),
        "obtained_marks": data.get("marks_obtained"),
        "correct_count": data.get("correct_questions"),
        "assessment_date": data.get("assessment_date"),
        "time_spent": data.get("time_spent"),
        "percentage": data.get("percentage"),
        "percentile": data.get("percentile"),
    }
    rows: List[Dict[str, Any]] = []
    for q in data.get("questions", []) or []:
        q_text = _strip_html(q.get("question") or "")
        q_text, _ = _prepare_candidate_text(q_text)

        qno    = q.get("question_number") or q.get("question_label")
        marks_wt = q.get("marks")

        # correct option text
        correct_text = ""
        for opt in q.get("options", []) or []:
            if int(opt.get("is_correct", 0)) == 1:
                correct_text = _strip_html(str(opt.get("option")))
                correct_text, _ = _prepare_candidate_text(correct_text)
                break

        # student response
        chosen_text = ""
        is_correct = None
        marks_awarded = None
        resp_list = q.get("response")
        if isinstance(resp_list, list) and resp_list:
            r0 = resp_list[0]
            is_correct = bool(r0.get("is_correct"))
            marks_awarded = r0.get("marks")
            opt_field = r0.get("option")
            if isinstance(opt_field, list) and opt_field:
                chosen_text = _strip_html(str(opt_field[0].get("response")))
                chosen_text, _ = _prepare_candidate_text(chosen_text)
            elif isinstance(opt_field, dict):
                chosen_text = _strip_html(str(opt_field.get("response")))
                chosen_text, _ = _prepare_candidate_text(chosen_text)

        rows.append({
            "qno": qno,
            "question": q_text,
            "chosen": chosen_text,
            "correct": correct_text,
            "is_correct": bool(is_correct) if is_correct is not None else "",
            "marks_awarded": marks_awarded if marks_awarded is not None else "",
            "marks_weight": marks_wt if marks_wt is not None else "",
        })

    # --- Fill missing questions with "Skipped" entries ---
    def _to_int_qno(v):
        # Try to coerce question label/number to int, else return None
        try:
            if v is None or v == "":
                return None
            # Some labels may be strings like "1" or "Q1"
            s = str(v).strip()
            # strip leading 'Q' or 'q'
            if s.lower().startswith('q'):
                s = s[1:]
            return int(s)
        except Exception:
            return None

    present_nums = set(filter(lambda x: x is not None, (_to_int_qno(r.get("qno")) for r in rows)))

    # Prefer API reported total_questions; else infer from max present
    try:
        total_q = int(summary.get("total_questions") or 0)
    except Exception:
        total_q = 0
    if total_q <= 0:
        total_q = max(present_nums) if present_nums else 0

    # Add missing numbers as "Skipped"
    for n in range(1, (total_q or 0) + 1):
        if n not in present_nums:
            rows.append({
                "qno": str(n),
                "question": "Skipped",
                "chosen": "",
                "correct": "",
                "is_correct": "",
                "marks_awarded": "",
                "marks_weight": "",
            })

    # Sort rows by numeric qno when possible
    def _sort_key(r):
        qn = _to_int_qno(r.get("qno"))
        return (qn is None, qn if qn is not None else 10**9, str(r.get("qno")))
    rows = sorted(rows, key=_sort_key)

    df = pd.DataFrame(rows, columns=[
        "qno", "question", "chosen", "correct", "is_correct", "marks_awarded", "marks_weight"
    ])
    return summary, df

# =========================
# 4) Render text (math-aware) → Image
# =========================
def render_rich_text_image(text: str, max_width_cm: float) -> RLImage:
    """
    Render with matplotlib:
      - If text contains $…$/$$…$$, sanitize ONLY those spans and render with mathtext.
      - Otherwise, render as plain text.
    On any failure, gracefully fall back to plain text (no crash).
    """
    text = text or ""
    candidate, has_math = _prepare_candidate_text(text)

    def _draw(s: str):
        dpi = 200
        fig = plt.figure(figsize=(10, 0.01), dpi=dpi)
        fig.patch.set_alpha(0.0)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_frame_on(False)
        ax.text(0.01, 1.0, s, ha="left", va="top", fontsize=14, wrap=True, usetex=False)
        ax.set_xlim(0, 1)
        fig.tight_layout(pad=0.4)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, transparent=True, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img = RLImage(buf)
        max_w = max_width_cm * cm
        if img.drawWidth > max_w:
            scale = max_w / float(img.drawWidth)
            img.drawWidth *= scale
            img.drawHeight *= scale
        return img

    if has_math:
        try:
            return _draw(candidate)
        except Exception:
            return _draw(text)  # fallback
    else:
        return _draw(text)

# =========================
# 5) PDF (Teacher view) — card layout
# =========================
def build_teacher_pdf(summary: Dict[str, Any], df: pd.DataFrame, teacher_name: str,
                      student_id: str, assessment_id: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24
    )
    styles = getSampleStyleSheet()
    h1 = styles['Title']
    h2 = styles['Heading2']
    normal = styles['Normal']
    small = ParagraphStyle('small', parent=normal, fontSize=9, leading=12)

    story: List = []
    story.append(Paragraph("Teacher View: Test Report", h1))
    story.append(Spacer(1, 6))

    # Summary card
    summary_table_data = [[
        Paragraph(f"<b>Test:</b> {summary.get('test_name','')}", normal),
        Paragraph(f"<b>Student ID:</b> {student_id}", normal),
        Paragraph(f"<b>Assessment ID:</b> {assessment_id}", normal),
    ],[
        Paragraph(f"<b>Score:</b> {summary.get('obtained_marks','-')} / {summary.get('total_marks','-')}", normal),
        Paragraph(f"<b>Correct:</b> {summary.get('correct_count','-')} / {summary.get('total_questions','-')}", normal),
        Paragraph(f"<b>Teacher:</b> {teacher_name}", normal),
    ]]
    if summary.get("assessment_date"):
        pct = f"{round(float(summary['percentage']),2)}%" if summary.get("percentage") is not None else "-"
        summary_table_data.append([
            Paragraph(f"<b>Date:</b> {summary.get('assessment_date')}", normal),
            Paragraph(f"<b>Percentage:</b> {pct}", normal),
            Paragraph(f"<b>Time Spent (s):</b> {summary.get('time_spent','-')}", normal),
        ])
    sum_tbl = Table(summary_table_data, colWidths=[6.2*cm, 5.0*cm, 5.0*cm])
    sum_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
        ('BOX', (0,0), (-1,-1), 0.6, colors.grey),
        ('INNERGRID', (0,0), (-1,-1), 0.3, colors.lightgrey),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(sum_tbl)
    story.append(Spacer(1, 10))

    # Questions (card layout)
    if not df.empty:
        story.append(Paragraph("Responses", h2))
        story.append(Spacer(1, 4))

        for i, r in df.iterrows():
            qno = str(r.get("qno") or i+1)
            q_text = r.get("question") or ""
            chosen = r.get("chosen") or ""
            correct = r.get("correct") or ""
            awarded = r.get("marks_awarded") or ""
            weight  = r.get("marks_weight") or ""
            is_ok = bool(r.get("is_correct"))

            # Top line
            top_line = Table([[
                Paragraph(f"<b>Q{qno}</b>", normal),
                Paragraph(("✔ Correct" if is_ok else "✘ Incorrect"),
                          ParagraphStyle('chip', parent=normal,
                                         textColor=(colors.green if is_ok else colors.red))),
                Paragraph(f"<b>Awarded:</b> {awarded}", small),
                Paragraph(f"<b>Weight:</b> {weight}", small),
            ]], colWidths=[2.0*cm, 3.0*cm, 5.0*cm, 5.2*cm])
            top_line.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('ALIGN', (1,0), (3,0), 'LEFT'),
            ]))

            # Render math-aware images
            q_img  = render_rich_text_image(q_text, max_width_cm=14.5)
            # Render labels and values separately to avoid mixing text and math spans
            chosen_label_img  = render_rich_text_image("Chosen:", max_width_cm=2.0)
            chosen_value_img  = render_rich_text_image(chosen if chosen else "—", max_width_cm=5.3)
            correct_label_img = render_rich_text_image("Correct:", max_width_cm=2.0)
            correct_value_img = render_rich_text_image(correct if correct else "—", max_width_cm=5.3)

            ch_img = Table([[chosen_label_img, chosen_value_img]], colWidths=[2.0*cm, 5.5*cm])
            co_img = Table([[correct_label_img, correct_value_img]], colWidths=[2.0*cm, 5.5*cm])
            for t in (ch_img, co_img):
                t.setStyle(TableStyle([
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('LEFTPADDING', (0,0), (-1,-1), 0),
                    ('RIGHTPADDING', (0,0), (-1,-1), 0),
                    ('TOPPADDING', (0,0), (-1,-1), 0),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 0),
                ]))

            qc_table = Table([[q_img],
                              [Table([[ch_img, co_img]], colWidths=[7.5*cm, 7.5*cm])]],
                             colWidths=[15.0*cm])
            qc_table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]))

            card = Table([[top_line], [qc_table]], colWidths=[15.0*cm])
            card.setStyle(TableStyle([
                ('BOX', (0,0), (-1,-1), 0.6, colors.lightgrey),
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F7F9FC")),
                ('LEFTPADDING', (0,0), (-1,-1), 10),
                ('RIGHTPADDING', (0,0), (-1,-1), 10),
                ('TOPPADDING', (0,0), (-1,-1), 10),
                ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ]))

            story.append(KeepTogether([card]))
            story.append(Spacer(1, 6))
            if (i + 1) % 10 == 0:
                story.append(PageBreak())

    doc.build(story)
    return buffer.getvalue()

# =========================
# 6) Streamlit UI
# =========================
st.set_page_config(page_title="Teacher Test Report", page_icon="📄", layout="wide")
st.title("📄 Teacher View — Test Report Builder")

student_id = st.text_input("Enter Student ID", "", placeholder="e.g., 447")
go = st.button("Generate Report")

if go:
    try:
        st.info("🔎 Fetching token & assessment_id from Redash…")
        token, assessment_id, _ = redash_fetch_token_and_assessment(student_id)
        if not token or not assessment_id:
            st.error("Could not find both token and assessment_id from Redash. Check query columns or :student_id.")
            st.stop()
        st.success("✅ Redash output received.")

        st.info("🌐 Calling Test API…")
        api_json = call_test_api(token, str(assessment_id), str(student_id))
        st.success("✅ API Call made.")
        st.success("✅ API Response received.")

        summary, df = normalize_test_json_sparkl(api_json)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score", f"{summary.get('obtained_marks','-')}/{summary.get('total_marks','-')}")
        c2.metric("Correct", f"{summary.get('correct_count','-')}/{summary.get('total_questions','-')}")
        c3.metric("Percent", f"{round(float(summary['percentage']),2)}%" if summary.get("percentage") is not None else "-")
        c4.metric("Time (s)", summary.get("time_spent","-"))

        st.subheader("Responses (Teacher View)")
        st.dataframe(df, use_container_width=True)

        with st.expander("Raw Test API JSON"):
            st.json(api_json)

        st.info("🖨️ PDF generation in progress…")
        pdf_bytes = build_teacher_pdf(summary, df, TEACHER_NAME, student_id, assessment_id)
        st.success("✅ PDF generated.")

        fname = f"TeacherReport_{student_id}_{assessment_id}.pdf".replace(" ", "_")
        st.download_button("⬇️ Download Teacher Report (PDF)", data=pdf_bytes, file_name=fname, mime="application/pdf", use_container_width=True)

    except RedashError as e:
        st.error(f"Redash error: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
