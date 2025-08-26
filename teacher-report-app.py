
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

# ---------- Matplotlib (math rendering for PDF) ----------
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# =========================
# Config from .env
# =========================
REDASH_URL      = os.getenv("REDASH_URL", "").strip()
REDASH_API_KEY  = os.getenv("REDASH_API_KEY", "").strip()
REDASH_QUERY_ID = os.getenv("REDASH_QUERY_ID", "").strip()
TEST_API_URL    = os.getenv("TEST_API_URL", "").strip()
TEACHER_NAME    = os.getenv("TEACHER_NAME", "Teacher").strip()


# PDF font sizes (env-tunable)
PDF_QUESTION_FONT_SIZE = int(os.getenv("PDF_QUESTION_FONT_SIZE", "12"))
PDF_CHOICE_FONT_SIZE   = int(os.getenv("PDF_CHOICE_FONT_SIZE", "14"))

# =========================
# Utilities
# =========================
class RedashError(RuntimeError):
    pass

def _require_json(r: requests.Response) -> Dict[str, Any]:
    ctype = (r.headers.get("Content-Type") or "")
    if "application/json" not in ctype.lower():
        snippet = (r.text or "")[:500]
        raise RedashError(f"Expected JSON, got {ctype}. Body snippet:\n{snippet}")
    return r.json()

# =========================
# 1) Redash: fetch token + assessment_id
# =========================
def redash_fetch_options(student_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
        rows = data.get("query_result", {}).get("data", {}).get("rows", []) or []
        # Robust parse: student name + multiple assessment ids (csv/json/list)
        opts: List[Dict[str, Any]] = []
        for row in rows:
            # Token (fallback keys too)
            token = row.get("token") or row.get("lt_token") or row.get("user_token")

            # Student name (broad keys + first/last fallback)
            sname = (
                row.get("student_name") or row.get("name") or row.get("student") or
                row.get("student_full_name") or row.get("student_fullname") or row.get("full_name") or row.get("fullname") or row.get("studentname")
            )
            if not sname:
                fn = row.get("first_name")
                ln = row.get("last_name")
                if fn or ln:
                    sname = f"{fn or ''} {ln or ''}".strip() or None
            if not sname:
                sname = "Student"

            # Possible fields that may contain one/many assessment ids
            candidates: List[Any] = []
            for key in ("assessment_ids", "assessments", "assessment_list", "assessment_id", "id"):
                if key in row and row.get(key) is not None:
                    candidates.append(row.get(key))
            if not candidates:
                continue

            def _emit(aid_val: Any):
                aid_str = str(aid_val).strip()
                if not aid_str:
                    return
                opts.append({"token": token, "assessment_id": aid_str, "student_name": sname})

            for val in candidates:
                # If already a list/tuple → emit all
                if isinstance(val, (list, tuple)):
                    for x in val:
                        _emit(x)
                    continue
                # Strings: JSON-ish or CSV/space separated or plain
                if isinstance(val, str):
                    s = val.strip()
                    if not s:
                        continue
                    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
                        import re as _re
                        # Extract ids that look like uuids/ids, plus plain numbers
                        for m in _re.findall(r"[0-9a-fA-F-]{6,}", s):
                            _emit(m)
                        for m in _re.findall(r"\b\d+\b", s):
                            _emit(m)
                        continue
                    if "," in s or " " in s:
                        # Prefer comma split; if none, whitespace split
                        parts = [p.strip() for p in s.replace("\n", " ").replace("\t", " ").split(",")]
                        if len(parts) == 1:
                            parts = [p for p in s.split() if p]
                        for p in parts:
                            _emit(p)
                        continue
                    _emit(s)
                    continue
                # Numbers or other scalars
                _emit(val)

        # Dedup by (token, assessment_id)
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for o in opts:
            k = (o.get("token"), str(o.get("assessment_id")))
            if k in seen:
                continue
            seen.add(k)
            deduped.append(o)
        return deduped, data

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
    rows = result_json.get("query_result", {}).get("data", {}).get("rows", []) or []

    # Robust parse: student name + multiple assessment ids (csv/json/list)
    opts: List[Dict[str, Any]] = []
    for row in rows:
        # Token (fallback keys too)
        token = row.get("token") or row.get("lt_token") or row.get("user_token")

        # Student name (broad keys + first/last fallback)
        sname = (
            row.get("student_name") or row.get("name") or row.get("student") or
            row.get("student_full_name") or row.get("student_fullname") or row.get("full_name") or row.get("fullname") or row.get("studentname")
        )
        if not sname:
            fn = row.get("first_name")
            ln = row.get("last_name")
            if fn or ln:
                sname = f"{fn or ''} {ln or ''}".strip() or None
        if not sname:
            sname = "Student"

        # Possible fields that may contain one/many assessment ids
        candidates: List[Any] = []
        for key in ("assessment_ids", "assessments", "assessment_list", "assessment_id", "id"):
            if key in row and row.get(key) is not None:
                candidates.append(row.get(key))
        if not candidates:
            continue

        def _emit(aid_val: Any):
            aid_str = str(aid_val).strip()
            if not aid_str:
                return
            opts.append({"token": token, "assessment_id": aid_str, "student_name": sname})

        for val in candidates:
            # If already a list/tuple → emit all
            if isinstance(val, (list, tuple)):
                for x in val:
                    _emit(x)
                continue
            # Strings: JSON-ish or CSV/space separated or plain
            if isinstance(val, str):
                s = val.strip()
                if not s:
                    continue
                if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
                    import re as _re
                    # Extract ids that look like uuids/ids, plus plain numbers
                    for m in _re.findall(r"[0-9a-fA-F-]{6,}", s):
                        _emit(m)
                    for m in _re.findall(r"\b\d+\b", s):
                        _emit(m)
                    continue
                if "," in s or " " in s:
                    # Prefer comma split; if none, whitespace split
                    parts = [p.strip() for p in s.replace("\n", " ").replace("\t", " ").split(",")]
                    if len(parts) == 1:
                        parts = [p for p in s.split() if p]
                    for p in parts:
                        _emit(p)
                    continue
                _emit(s)
                continue
            # Numbers or other scalars
            _emit(val)

    # Dedup by (token, assessment_id)
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for o in opts:
        k = (o.get("token"), str(o.get("assessment_id")))
        if k in seen:
            continue
        seen.add(k)
        deduped.append(o)
    return deduped, result_json
    

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
_MATH_SPAN_RE = re.compile(r"(\${1,2})([^$]+?)\1")  # $...$ or $$...$$

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

    # Fix malformed sqrt:
    #   \sqrt(2)  -> \sqrt{2}
    inner = re.sub(r"\\sqrt\s*\(\s*([^)]+?)\s*\)", r"\\sqrt{\1}", inner)
    #   \sqrt2, \sqrt x, \sqrt3  -> \sqrt{2}, \sqrt{x}, \sqrt{3}
    inner = re.sub(r"\\sqrt(?!\{)\s*([A-Za-z0-9]+)", r"\\sqrt{\1}", inner)

    # Convert simple numeric a/b → \frac{a}{b}
    inner = re.sub(r"(?<!\\)(?<!\w)\b([0-9]{1,3})\s*/\s*([0-9]{1,3})\b(?!\w)", r"\\frac{\1}{\2}", inner)

    # Escape special chars that break mathtext (incl. underscore)
    inner = inner.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")

    return inner

def _prepare_candidate_text(text: str) -> Tuple[str, bool]:
    """
    Treat only $…$ or $$…$$ as math. Sanitize their inners. Return (candidate, has_math_spans).
    If dollars are unbalanced, remove $ and return has_math_spans=False.
    """
    text = text or ""
    text, ok = _balance_dollars_whole(text)
    if not ok:
        return text, False

    has_math = False
    def repl(m):
        nonlocal has_math
        inner = m.group(2)
        has_math = True
        inner = _sanitize_math_inner(inner)
        # Normalize $$…$$ to single-$ for mathtext
        return f"${inner}$"

    candidate = _MATH_SPAN_RE.sub(repl, text)
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

    df = pd.DataFrame(rows, columns=[
        "qno", "question", "chosen", "correct", "is_correct", "marks_awarded", "marks_weight"
    ])
    return summary, df

# -------------------------
# Helper: extract student name from API JSON (fallback if Redash lacks it)
# -------------------------

def extract_student_name_from_api(api_json: Dict[str, Any]) -> Optional[str]:
    try:
        data = api_json.get("data") or {}
        # Common direct keys
        for key in (
            "student_name", "studentName", "name", "student_full_name", "full_name", "fullname"
        ):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # Nested structures
        for key in ("student", "user", "profile"):
            node = data.get(key)
            if isinstance(node, dict):
                # first+last
                fn = node.get("first_name") or node.get("firstName") or node.get("given_name")
                ln = node.get("last_name") or node.get("lastName") or node.get("family_name")
                if fn or ln:
                    name = f"{(fn or '').strip()} {(ln or '').strip()}".strip()
                    if name:
                        return name
                # direct name fields
                for nk in ("name", "full_name", "fullname", "student_name"):
                    nv = node.get(nk)
                    if isinstance(nv, str) and nv.strip():
                        return nv.strip()
        # As a last resort, if an email is present, use local-part as a hint
        email = data.get("email") or (data.get("student") or {}).get("email") if isinstance(data.get("student"), dict) else None
        if isinstance(email, str) and "@" in email:
            return email.split("@")[0].replace(".", " ").replace("_", " ").title()
    except Exception:
        pass
    return None

# =========================
# 4) Render text (math-aware) → Image for PDF
# =========================
def render_rich_text_image(text: str, max_width_cm: float, font_size: int = 12) -> RLImage:
    """
    Render with matplotlib:
      - If text contains $…$/$$…$$, sanitize ONLY those spans and render with mathtext.
      - Otherwise, render as plain text.
    On any failure, gracefully fall back to plain text (no crash).
    """
    text = text or ""
    candidate, has_math = _prepare_candidate_text(text)

    # Always prepare a safe plain-text fallback with ALL '$' removed if odd or problematic
    # This avoids matplotlib's mathtext parser from choking on stray '$'.
    safe_plain, _ = _balance_dollars_whole(text)

    def _draw(s: str):
        dpi = 200
        fig = plt.figure(figsize=(8, 0.01), dpi=dpi)
        fig.patch.set_alpha(0.0)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_frame_on(False)
        ax.text(0.01, 1.0, s, ha="left", va="top", fontsize=font_size, wrap=True, usetex=False)
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
            # Fallback to safe plain text with stray '$' removed
            return _draw(safe_plain)
    else:
        # Even if we didn't detect proper math spans, we still pass a safe plain string
        return _draw(safe_plain)

# =========================
# 5) PDF (Teacher view) — card layout (bold labels + larger choice fonts)
# =========================
def build_teacher_pdf(summary: Dict[str, Any], df: pd.DataFrame, teacher_name: str,
                      student_id: str, student_name: str, assessment_id: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24
    )
    styles = getSampleStyleSheet()
    h1 = styles['Title']
    h2 = styles['Heading2']
    normal = styles['Normal']
    small = ParagraphStyle('small', parent=normal, fontSize=9, leading=12)
    bold_small = ParagraphStyle('bold_small', parent=small, fontName='Helvetica-Bold')

    story: List = []
    story.append(Paragraph("Teacher View: Test Report", h1))
    story.append(Spacer(1, 6))

    # Summary card
    pct = "-"
    if summary.get("percentage") is not None:
        try:
            pct = f"{round(float(summary['percentage']), 2)}%"
        except Exception:
            pct = str(summary.get("percentage"))
    summary_table_data = [[
        Paragraph(f"<b>Test:</b> {summary.get('test_name','')}", normal),
        Paragraph(f"<b>Student ID:</b> {student_id}", normal),
        Paragraph(f"<b>Student Name:</b> {student_name}", normal),
    ],[
        Paragraph(f"<b>Assessment ID:</b> {assessment_id}", normal),
        Paragraph(f"<b>Score:</b> {summary.get('obtained_marks','-')} / {summary.get('total_marks','-')}", normal),
        Paragraph(f"<b>Correct:</b> {summary.get('correct_count','-')} / {summary.get('total_questions','-')}", normal),
    ],[
        Paragraph(f"<b>Teacher:</b> {teacher_name}", normal),
        Paragraph(f"<b>Date:</b> {summary.get('assessment_date','-')}", normal),
        Paragraph(f"<b>Percentage:</b> {pct}", normal),
    ],[
        Paragraph(f"<b>Time Spent (s):</b> {summary.get('time_spent','-')}", normal),
        Paragraph("", normal),
        Paragraph("", normal),
    ]]
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

            # Render images
            q_img  = render_rich_text_image(q_text, max_width_cm=16.5, font_size=PDF_QUESTION_FONT_SIZE)

            # Labels bold, values larger via images
            chosen_val_img  = render_rich_text_image(chosen if chosen else "—", max_width_cm=8.0, font_size=PDF_CHOICE_FONT_SIZE)
            correct_val_img = render_rich_text_image(correct if correct else "—", max_width_cm=8.0, font_size=PDF_CHOICE_FONT_SIZE)

            ch_table = Table([
                [Paragraph("Chosen:", bold_small)],
                [chosen_val_img]
            ], colWidths=[8.25*cm])
            ch_table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING', (0,0), (-1,-1), 2),
                ('RIGHTPADDING', (0,0), (-1,-1), 2),
                ('TOPPADDING', (0,0), (-1,-1), 2),
                ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ]))

            co_table = Table([
                [Paragraph("Correct:", bold_small)],
                [correct_val_img]
            ], colWidths=[8.25*cm])
            co_table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING', (0,0), (-1,-1), 2),
                ('RIGHTPADDING', (0,0), (-1,-1), 2),
                ('TOPPADDING', (0,0), (-1,-1), 2),
                ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ]))

            qc_table = Table([[q_img],
                              [Table([[ch_table, co_table]], colWidths=[8.25*cm, 8.25*cm])]],
                             colWidths=[16.5*cm])
            qc_table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]))

            card = Table([[top_line], [qc_table]], colWidths=[16.5*cm])
            card.setStyle(TableStyle([
                ('BOX', (0,0), (-1,-1), 0.6, colors.lightgrey),
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F7F9FC")),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ]))

            story.append(KeepTogether([card]))
            story.append(Spacer(1, 6))
            if (i + 1) % 10 == 0:
                story.append(PageBreak())

    doc.build(story)
    return buffer.getvalue()

# =========================
# 6) Streamlit UI (stateful, two-step)
# =========================
st.set_page_config(page_title="Teacher Test Report", page_icon="📄", layout="wide")
st.title("📄 Teacher View — Test Report Builder")

# Enlarge math in UI a bit (optional)
st.markdown(
    """
    <style>
      .teacher-katex p {font-size: 18px; line-height: 1.45;}
    </style>
    """,
    unsafe_allow_html=True
)

# Session state keys
STATE_KEYS = {
    "student_id": "student_id",
    "options": "options",
    "ass_idx": "ass_idx",
    "chosen": "chosen",
    "generated": "generated",
}
for k in STATE_KEYS.values():
    st.session_state.setdefault(k, None)

# Inputs
student_id = st.text_input("Enter Student ID", value=st.session_state.get(STATE_KEYS["student_id"]) or "", placeholder="e.g., 447", key="student_id_input")
col_a, col_b = st.columns([1,1])
fetch_btn = col_a.button("🔎 Fetch Assessments")
generate_btn = col_b.button("🖨️ Generate Report")

# Step 1: Fetch (only when button pressed)
if fetch_btn:
    st.session_state[STATE_KEYS["generated"]] = None
    st.session_state[STATE_KEYS["chosen"]] = None
    st.session_state[STATE_KEYS["ass_idx"]] = 0
    st.session_state[STATE_KEYS["student_id"]] = student_id.strip()
    try:
        if not st.session_state[STATE_KEYS["student_id"]]:
            st.error("Please enter a Student ID first.")
        else:
            st.info("🔎 Fetching assessments from Redash…")
            options, _ = redash_fetch_options(st.session_state[STATE_KEYS["student_id"]])
            if not options:
                st.error("No assessments found for this student in Redash. Check query and :student_id.")
            else:
                st.session_state[STATE_KEYS["options"]] = options
                st.success(f"✅ Found {len(options)} assessment option(s).")
    except RedashError as e:
        st.error(f"Redash error: {e}")
    except Exception as e:
        st.error(f"Error while fetching: {e}")

# Step 2: Selection UI (persisted)
options = st.session_state.get(STATE_KEYS["options"]) or []
if options:
    labels = [f"{i+1}. Assessment {opt.get('assessment_id')} — {opt.get('student_name') or 'Unknown'}" for i, opt in enumerate(options)]
    st.selectbox(
        "Select an assessment:",
        options=list(range(len(options))),
        index=min(st.session_state.get(STATE_KEYS["ass_idx"]) or 0, max(len(options)-1, 0)),
        format_func=lambda i: labels[i],
        key=STATE_KEYS["ass_idx"],
    )
    with st.expander("Debug: Redash options (parsed)"):
        st.json(options)

# Step 3: Generate (only when button pressed)
if generate_btn:
    try:
        if not options:
            st.error("No options loaded. Click ‘Fetch Assessments’ first.")
        else:
            idx = st.session_state.get(STATE_KEYS["ass_idx"]) or 0
            idx = int(idx)
            if idx < 0 or idx >= len(options):
                st.error("Invalid selection index.")
            else:
                chosen = options[idx]
                st.session_state[STATE_KEYS["chosen"]] = chosen
                token = chosen.get("token")
                assessment_id = chosen.get("assessment_id")
                student_name = chosen.get("student_name") or "Student"
                if not token:
                    st.error("Redash did not return a token for this student. Ensure your query joins latest_token.")
                    st.stop()
                if not assessment_id:
                    st.error("No assessment_id found. If your query returns comma-separated assessment_ids, make sure the column name is one of: assessment_ids / assessments / assessment_list.")
                    st.stop()
                assessment_id = str(assessment_id)

                st.info("🌐 Calling Test API…")
                api_json = call_test_api(token, str(assessment_id), str(st.session_state[STATE_KEYS["student_id"]]))
                st.success("✅ API Call made.")
                st.success("✅ API Response received.")

                summary, df = normalize_test_json_sparkl(api_json)

                # Fallback: if student_name missing/placeholder, try to pull from API payload
                if not student_name or student_name.strip().lower() == "student":
                    api_name = extract_student_name_from_api(api_json)
                    if api_name:
                        student_name = api_name

                st.write(f"**Student:** {student_name}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Score", f"{summary.get('obtained_marks','-')}/{summary.get('total_marks','-')}")
                c2.metric("Correct", f"{summary.get('correct_count','-')}/{summary.get('total_questions','-')}")
                try:
                    pct_val = f"{round(float(summary['percentage']),2)}%" if summary.get("percentage") is not None else "-"
                except Exception:
                    pct_val = str(summary.get("percentage"))
                c3.metric("Percent", pct_val)
                c4.metric("Time (s)", summary.get("time_spent","-"))

                st.subheader("Responses (Teacher View)")
                st.dataframe(df, use_container_width=True)

                with st.expander("Formatted Teacher Responses (KaTeX)"):
                    for i, r in df.iterrows():
                        st.markdown(f"<div class='teacher-katex'><strong>Q{r.get('qno') or i+1}</strong> {'✔️' if r.get('is_correct') else '❌'}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='teacher-katex'>{r.get('question') or ''}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='teacher-katex'><em>Chosen:</em> {r.get('chosen') or '—'}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='teacher-katex'><em>Correct:</em> {r.get('correct') or '—'}</div>", unsafe_allow_html=True)
                        st.markdown("---")

                st.info("🖨️ PDF generation in progress…")
                pdf_bytes = build_teacher_pdf(summary, df, TEACHER_NAME, st.session_state[STATE_KEYS["student_id"]], student_name, assessment_id)
                st.success("✅ PDF generated.")

                fname = f"TeacherReport_{st.session_state[STATE_KEYS['student_id']]}_{assessment_id}.pdf".replace(" ", "_")
                st.download_button("⬇️ Download Teacher Report (PDF)", data=pdf_bytes, file_name=fname, mime="application/pdf", use_container_width=True)

                st.session_state[STATE_KEYS["generated"]] = True
    except RedashError as e:
        st.error(f"Redash error: {e}")
    except Exception as e:
        st.error(f"Error while generating: {e}")
