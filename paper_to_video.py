#!/usr/bin/env python3
"""
paper_to_video.py — local NotebookLM-style "video overview" generator, built
for Kurt (Mac Studio M3 Ultra, 96GB unified memory).

Pipeline:
  1. Extract text + embedded figures from a paper PDF (PyMuPDF).
  2. Ask a local LLM (Ollama) to turn the text into a beat-by-beat narration
     script, each beat optionally pointing at a figure to show.
  3. Synthesize narration per beat with Kokoro TTS running natively on
     Apple Silicon via mlx-audio (single narrator).
  4. Render a slide per beat — a real extracted figure with a caption, or a
     title/bullets card if no figure fits — and turn it into a Ken-Burns
     video clip timed to that beat's narration length.
  5. Concatenate all clips into the final mp4.

Install on Kurt:
    pip install -r requirements.txt --break-system-packages
    brew install ffmpeg   # if you don't already have it on PATH

Usage:
    python paper_to_video.py paper.pdf --output overview.mp4 \
        --model qwen2.5:32b-instruct --voice af_heart

To route the script-generation step through MLX-LM's OpenAI-compatible
server (port 10000) instead of Ollama, see the NOTE above generate_script().
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import pymupdf
import requests
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Config defaults — override via CLI flags where noted
# ---------------------------------------------------------------------------
VIDEO_W, VIDEO_H = 1920, 1080
FPS = 30
MIN_FIGURE_BYTES = 8_000    # skip tiny embedded images (journal logos, icons)
MAX_FIGURES = 24            # don't bother extracting more than this per paper
MAX_PROMPT_CHARS = 200_000  # ~50k tokens — override via --max-prompt-chars.
                             # Methods/Discussion often sit at the very end of a
                             # paper, so truncating here silently drops them —
                             # this default is sized to fit most full papers.
DEFAULT_NUM_CTX = 65536     # override via --num-ctx; must stay roughly in step
                             # with --max-prompt-chars (~4 chars/token, plus
                             # headroom for the prompt wrapper and JSON output)
ACCENT = (0, 122, 194)
BG = (250, 250, 248)

# Beat-count / narration-depth presets, selected via --detailed
STANDARD_DEPTH = dict(
    depth_desc="concise",
    beat_min=6, beat_max=10,
    narration_min=1, narration_max=3,
    bullets_min=1, bullets_max=3,
    methods_note="cover the method as a single beat or two",
    discussion_note="a closing beat noting what the results mean",
)
DETAILED_DEPTH = dict(
    depth_desc="in-depth, detailed",
    beat_min=16, beat_max=28,
    narration_min=3, narration_max=6,
    bullets_min=2, bullets_max=5,
    methods_note=(
        "break the Methods into AS MANY separate beats as needed — one per "
        "major technique, dataset, pipeline, or analysis step — and explain "
        "*how* each one actually works, not just that it was done"
    ),
    discussion_note=(
        "2-3 beats thoroughly summarizing the Discussion: what the findings "
        "mean, why they matter, how they fit prior work, and any limitations "
        "or caveats the authors themselves raise"
    ),
)


# ---------------------------------------------------------------------------
# 1. Extract paper text + figures
# ---------------------------------------------------------------------------
FIGURE_CAPTION_RE = re.compile(r"^(?:extended\s+data\s+)?fig(?:ure)?\.?\s*\d+[a-z]?\s*[\|:.]", re.IGNORECASE)
FIGURE_DPI = 200


def _find_caption_blocks(page):
    """Locate figure-caption text blocks on a page (e.g. 'Fig. 2 | Protein
    importance ranking...'), sorted top to bottom."""
    captions = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        text = text.strip()
        if text and FIGURE_CAPTION_RE.match(text):
            captions.append({"y0": y0, "y1": y1, "text": re.sub(r"\s+", " ", text)[:300]})
    captions.sort(key=lambda c: c["y0"])
    return captions


def _extract_page_figures(page, work_dir: Path, n_so_far: int):
    """Render the region around each detected figure caption directly from
    the PDF page. This is deliberately NOT based on page.get_images(): most
    journal figures (line plots, volcano plots, etc.) are drawn as vector
    graphics with no embedded raster image at all, so extracting embedded
    images misses the real figures and, on image-only/flattened pages, can
    instead pick up meaningless raster tiles. Rendering the page and cropping
    around the caption works for both vector and raster content."""
    captions = _find_caption_blocks(page)
    if not captions:
        return []

    figures = []
    page_rect = page.rect
    top = page_rect.y0
    for i, cap in enumerate(captions):
        bottom = (cap["y1"] + captions[i + 1]["y0"]) / 2 if i + 1 < len(captions) else page_rect.y1
        clip = pymupdf.Rect(page_rect.x0, top, page_rect.x1, bottom)
        pix = page.get_pixmap(dpi=FIGURE_DPI, clip=clip)
        fig_path = work_dir / f"figure_{n_so_far + len(figures):02d}.png"
        pix.save(fig_path)
        figures.append({"path": fig_path, "caption": cap["text"]})
        top = bottom
    return figures


MAX_RASTER_IMAGES_PER_PAGE = 8  # a real embedded figure is 1-4 images; more than
                                 # this on one page usually means a flattened/tiled
                                 # page (e.g. a scanned page split into strips), not
                                 # distinct content — skip it rather than extract slivers


def _extract_page_raster_images(page, doc, work_dir: Path, n_so_far: int):
    """Fallback for pages with no detected caption: pull embedded raster
    images directly (catches figures in papers whose PDF really does embed
    them as images, e.g. dense heatmaps in bioinformatics papers)."""
    images = page.get_images(full=True)
    if not images or len(images) > MAX_RASTER_IMAGES_PER_PAGE:
        return []
    figures = []
    for img in images:
        xref = img[0]
        try:
            base = doc.extract_image(xref)
        except Exception:
            continue
        if len(base["image"]) < MIN_FIGURE_BYTES:
            continue
        fig_path = work_dir / f"figure_{n_so_far + len(figures):02d}.{base['ext']}"
        fig_path.write_bytes(base["image"])
        figures.append({"path": fig_path, "caption": ""})
    return figures


def extract_paper_content(pdf_path: Path, work_dir: Path, start_page: int = 0, end_page: int = None):
    """Extract text + figures from pages [start_page, end_page) (0-indexed,
    end_page exclusive; None means to the end of the document). Figures are
    a list of {"path": Path, "caption": str} — caption is "" when a figure
    came from the embedded-image fallback rather than a detected caption."""
    doc = pymupdf.open(pdf_path)
    end_page = len(doc) if end_page is None else min(end_page, len(doc))
    full_text = []
    figures = []

    for page_index in range(start_page, end_page):
        page = doc[page_index]
        full_text.append(page.get_text())
        if len(figures) >= MAX_FIGURES:
            continue
        page_figures = _extract_page_figures(page, work_dir, len(figures))
        if not page_figures:
            page_figures = _extract_page_raster_images(page, doc, work_dir, len(figures))
        figures.extend(page_figures[:max(0, MAX_FIGURES - len(figures))])

    doc.close()
    return "\n".join(full_text), figures


# ---------------------------------------------------------------------------
# 1b. Chapter discovery (for study-guide mode) — uses the PDF's own bookmarks
# ---------------------------------------------------------------------------
def get_chapter_ranges(pdf_path: Path):
    """Return every top-level-bookmark section as {title, start, end} page
    indices (0-indexed, end exclusive), derived from the PDF's table of
    contents. Requires the PDF to have embedded bookmarks."""
    doc = pymupdf.open(pdf_path)
    toc = doc.get_toc()
    doc.close()
    if not toc:
        raise ValueError(
            "This PDF has no embedded table of contents/bookmarks, so chapters "
            "can't be auto-detected. Use the plain video mode instead, or split "
            "the PDF into per-chapter files yourself first."
        )
    chapters = []
    for i, (_level, title, page) in enumerate(toc):
        start = page - 1
        end = toc[i + 1][2] - 1 if i + 1 < len(toc) else None
        chapters.append({"title": title.strip(), "start": start, "end": end})
    return chapters


def is_content_chapter(title: str) -> bool:
    """True for actual chapters/appendices, filtering out front/back matter
    and Part dividers when iterating --all-chapters."""
    return bool(re.match(r"^(Chapter|Appendix)\s+\S+", title.strip(), re.IGNORECASE))


def select_chapter(chapters, selector: str):
    """Match a --chapter value against chapter titles: first try 'Chapter N'
    (or 'Appendix X') as a whole-token prefix, then fall back to substring."""
    selector = selector.strip()
    for c in chapters:
        if re.match(rf"^(chapter|appendix)\s+{re.escape(selector)}\b", c["title"], re.IGNORECASE):
            return c
    matches = [c for c in chapters if selector.lower() in c["title"].lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous --chapter {selector!r}, matches: {[c['title'] for c in matches]}")
    raise ValueError(f"No chapter found matching --chapter {selector!r}")


# ---------------------------------------------------------------------------
# 2. Generate the beat-by-beat script with a local LLM
# ---------------------------------------------------------------------------
SCRIPT_PROMPT = """You are producing a {depth_desc} narrated video overview of a research \
paper, for a scientist audience. Read the paper text below — it includes the full \
Methods and Discussion sections, don't assume anything got cut off — and return ONLY a \
JSON array (no markdown fences, no commentary) of {beat_min} to {beat_max} "beats" that \
walk through the paper in order:
  1. Motivation / background — why this work, what gap it fills
  2. Methods — {methods_note}
  3. Results — the key findings, in the order the paper presents them
  4. Discussion — {discussion_note}

Each beat is an object with:
  "title": short slide title (max 8 words)
  "bullets": {bullets_min}-{bullets_max} short bullet phrases (max 10 words each)
  "narration": {narration_min}-{narration_max} spoken sentences, plain conversational \
English that actually explains the mechanism or finding rather than just naming it, \
no jargon dumps
  "figure_hint": a short phrase describing which figure this beat should show, \
if any (e.g. "the CNV burden forest plot"), or null if it's a text-only beat

Paper text:
---
{paper_text}
---
Return the JSON array now."""

def _beats_schema(item_properties: dict, required: list, min_items: int, max_items: int) -> dict:
    """Array-of-beats JSON schema with minItems/maxItems — without this, a
    model can return a technically-valid but lazily short array (we've seen
    Ollama's structured-output mode do exactly that)."""
    return {
        "type": "array",
        "minItems": min_items,
        "maxItems": max_items,
        "items": {
            "type": "object",
            "properties": item_properties,
            "required": required,
        },
    }


BEAT_ITEM_PROPERTIES = {
    "title": {"type": "string"},
    "bullets": {"type": "array", "items": {"type": "string"}},
    "narration": {"type": "string"},
    "figure_hint": {"type": ["string", "null"]},
}


def _find_json_array(raw: str):
    """Locate the first top-level JSON array in a model reply, matching
    brackets by hand (not just first-'['-to-last-']') so trailing commentary
    or extra bracketed junk after the array doesn't get swept in."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)

    start = text.find("[")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]  # unterminated — let json.loads report the real error


def _repair_json(candidate: str) -> str:
    """Best-effort cleanup for the malformed-but-close-to-valid JSON local
    models tend to produce: raw control characters inside string literals
    (should be escaped) and trailing commas before a closing bracket."""
    repaired = []
    in_string = False
    escape = False
    for ch in candidate:
        if in_string:
            if escape:
                escape = False
                repaired.append(ch)
                continue
            if ch == "\\":
                escape = True
                repaired.append(ch)
                continue
            if ch == '"':
                in_string = False
                repaired.append(ch)
                continue
            if ch == "\n":
                repaired.append("\\n")
                continue
            if ch == "\t":
                repaired.append("\\t")
                continue
            repaired.append(ch)
            continue
        if ch == '"':
            in_string = True
        repaired.append(ch)
    text = "".join(repaired)
    text = re.sub(r",\s*([\]}])", r"\1", text)  # trailing commas
    return text


def _extract_beats_json(raw: str):
    """Pull the JSON array out of a model reply, tolerating markdown fences,
    stray commentary before/after, raw newlines in strings, trailing commas."""
    candidate = _find_json_array(raw)
    if candidate is None:
        raise ValueError(f"Couldn't find a JSON array in the model's reply:\n{raw[:500]}")

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(_repair_json(candidate))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model's reply wasn't valid JSON even after repair attempts "
            f"({exc}).\nFull reply:\n{raw}"
        ) from exc


def _call_ollama_for_beats(prompt: str, schema: dict, model: str, ollama_url: str,
                            num_ctx: int, timeout: int = 1800, min_items: int = 1,
                            max_attempts: int = 3):
    # NOTE: this hits Ollama's native /api/generate endpoint. To use MLX-LM's
    # OpenAI-compatible server on port 10000 instead, replace this POST with:
    #   requests.post(f"{base_url}/v1/chat/completions",
    #                 json={"model": model,
    #                       "messages": [{"role": "user", "content": prompt}]})
    # and read raw = response.json()["choices"][0]["message"]["content"].
    beats = None
    last_error = None
    for attempt in range(max_attempts):
        # nudge temperature up slightly on retries so a lazily-short reply
        # doesn't just get regenerated identically
        temperature = 0.4 + 0.15 * attempt
        resp = requests.post(
            f"{ollama_url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": schema,  # grammar-constrain output to this exact shape
                "options": {"temperature": temperature, "num_ctx": num_ctx},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json()["response"]
        try:
            candidate = _extract_beats_json(raw)
        except ValueError as exc:
            last_error = exc
            print(f"  Attempt {attempt + 1}/{max_attempts} produced invalid JSON, retrying...")
            continue

        if not isinstance(candidate, list):
            last_error = ValueError(f"Expected a JSON array of beats, got: {type(candidate).__name__}")
            print(f"  Attempt {attempt + 1}/{max_attempts}: {last_error}, retrying...")
            continue
        if not all(isinstance(b, dict) for b in candidate):
            last_error = ValueError("Expected each beat to be a JSON object")
            print(f"  Attempt {attempt + 1}/{max_attempts}: {last_error}, retrying...")
            continue

        beats = candidate
        if len(beats) >= min_items:
            break
        print(f"  Attempt {attempt + 1}/{max_attempts} returned only {len(beats)} beat(s) "
              f"(wanted >= {min_items}), retrying for fuller coverage...")

    if beats is None:
        raise last_error
    if len(beats) < min_items:
        print(f"  Warning: proceeding with only {len(beats)} beat(s) after {max_attempts} "
              f"attempts — coverage may be thin.")
    return beats


def generate_script(paper_text: str, model: str, ollama_url: str,
                     num_ctx: int = DEFAULT_NUM_CTX, max_prompt_chars: int = MAX_PROMPT_CHARS,
                     depth: dict = STANDARD_DEPTH, timeout: int = 1800):
    if len(paper_text) > max_prompt_chars:
        print(f"  Note: paper text is {len(paper_text)} chars, truncating to "
              f"{max_prompt_chars} (--max-prompt-chars) before sending to the model.")
    prompt = SCRIPT_PROMPT.format(paper_text=paper_text[:max_prompt_chars], **depth)
    schema = _beats_schema(BEAT_ITEM_PROPERTIES, ["title", "narration"],
                            depth["beat_min"], depth["beat_max"])
    beats = _call_ollama_for_beats(prompt, schema, model, ollama_url, num_ctx, timeout,
                                    min_items=depth["beat_min"])

    for beat in beats:
        beat.setdefault("bullets", [])
        beat.setdefault("figure_hint", None)
    return beats


# ---------------------------------------------------------------------------
# 2b. Study-guide script generation — one chapter, explained section by
#     section, with periodic quiz/answer beats
# ---------------------------------------------------------------------------
STUDY_GUIDE_PROMPT = """You are a patient, thorough tutor making a video study guide for one \
chapter of a textbook, to help a student genuinely learn the material (not just skim it). \
Book: {book_title}. Chapter: {chapter_title}.

This chapter has the following sections. YOU MUST PRODUCE AT LEAST ONE BEAT FOR EVERY ONE \
OF THEM, IN ORDER, ALL THE WAY TO THE LAST ONE — do not stop partway through the list even \
if it means using close to the maximum beat count:
{section_list}

Read the chapter text below and return ONLY a JSON array (no markdown fences, no \
commentary) of {beat_min} to {beat_max} beats that walk through every section listed above \
in order. For each concept, actually explain how and why it works, walk through any key \
derivations or examples in the text, and connect it to what came before — don't just \
name-drop terms. Give the later sections the same depth as the early ones; a common mistake \
is to explain the first few sections in detail and then rush or drop the rest — do not do \
that.

Roughly every 3-5 "concept" beats, insert one "quiz" beat that poses a short question \
testing the material just covered, immediately followed by one "answer" beat that gives \
the correct answer and explains the reasoning step by step. Spread quizzes across the \
whole chapter, not just the end — including the later sections.

Each beat is an object with:
  "type": "concept", "quiz", or "answer"
  "section": the section number/heading this beat belongs to (e.g. "1.3 Bayes' rule"), \
or null if not tied to one section
  "title": short slide title (max 8 words)
  "bullets": 1-5 short bullet phrases (max 12 words each) — key points for "concept", the \
question restated for "quiz", the key reasoning steps for "answer"
  "narration": for "concept", {narration_min}-{narration_max} sentences thoroughly \
explaining the idea in plain conversational English; for "quiz", clearly pose the question \
and tell the listener to pause the video and think it through; for "answer", state the \
correct answer and walk through the reasoning
  "figure_hint": a short phrase describing which figure or table this beat should show, \
if any, or null

Use as many beats as needed to genuinely cover the chapter — do not compress distinct \
concepts together just to hit a low count, and do not stop before the last section listed \
above.

Chapter text:
---
{chapter_text}
---
Return the JSON array now."""


def chapter_label_from_title(chapter_title: str):
    """'Chapter 5: ...' -> '5', 'Appendix A: ...' -> 'A' — the leading token
    real section numbers in this chapter should start with."""
    m = re.match(r"^chapter\s+(\d+)", chapter_title.strip(), re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r"^appendix\s+([A-Za-z])", chapter_title.strip(), re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _looks_like_heading(title: str) -> bool:
    """Reject table-of-numbers noise (e.g. a data table row like '0.19' /
    '1977') that would otherwise be mistaken for a section title."""
    title = title.strip()
    if not (3 <= len(title) <= 100):
        return False
    if not re.search(r"[A-Za-z]", title):
        return False
    if re.match(r"^[\d.,%$\-\s]+$", title):
        return False
    return True


def extract_section_headings(chapter_text: str, chapter_label: str = None):
    """Best-effort extraction of numbered section headings (e.g. '2.5' /
    'Estimating a normal mean...'), each with its character offset in
    chapter_text so the chapter can be split into per-section chunks. Tries
    two layouts: number and title on separate lines (common when a PDF
    renderer breaks the heading across lines), and 'N.M Title' on one line.
    When chapter_label is given (e.g. '2' or 'A'), only numbers starting with
    that label count — otherwise stray decimals in data tables (e.g. a '0.19'
    / '1977' row) get mistaken for section headings."""
    label = re.escape(chapter_label) if chapter_label else r"\d{1,2}"
    num_pattern = rf"({label}\.\d{{1,2}})"
    lines = chapter_text.split("\n")
    offsets = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    headings = []
    seen = set()
    for i, line in enumerate(lines):
        m = re.match(rf"^{num_pattern}\s*$", line.strip())
        if not m or m.group(1) in seen:
            continue
        title = next((lines[j].strip() for j in range(i + 1, min(i + 3, len(lines)))
                      if lines[j].strip()), None)
        if title and _looks_like_heading(title):
            headings.append({"num": m.group(1), "title": title, "offset": offsets[i]})
            seen.add(m.group(1))

    if len(headings) < 3:
        headings = []
        seen = set()
        for i, line in enumerate(lines):
            m = re.match(rf"^{num_pattern}\s+([A-Z][^\n]{{3,90}})$", line.strip())
            if m and m.group(1) not in seen and _looks_like_heading(m.group(2)):
                headings.append({"num": m.group(1), "title": m.group(2).strip(), "offset": offsets[i]})
                seen.add(m.group(1))

    return headings


def _is_content_heading(heading: dict) -> bool:
    return not re.search(r"bibliographic|exercise", heading["title"], re.IGNORECASE)


STUDY_BEAT_ITEM_PROPERTIES = {
    "type": {"type": "string", "enum": ["concept", "quiz", "answer"]},
    "section": {"type": ["string", "null"]},
    "title": {"type": "string"},
    "bullets": {"type": "array", "items": {"type": "string"}},
    "narration": {"type": "string"},
    "figure_hint": {"type": ["string", "null"]},
}

CHUNK_STUDY_PROMPT = """You are a patient, thorough tutor making a video study guide for one \
chapter of a textbook, to help a student genuinely learn the material (not just skim it). \
Book: {book_title}. Chapter: {chapter_title}.

This is one part of the chapter, covering section(s): {section_labels}. A separate part \
handles the rest of the chapter, so focus only on the text below — but do cover ALL of it.

Read the text below and return ONLY a JSON array (no markdown fences, no commentary) of \
{beat_min} to {beat_max} beats explaining this material thoroughly: walk through \
definitions, derivations, and worked examples in the text, actually explaining how and why \
things work rather than just naming them.

{quiz_instruction}

Each beat is an object with:
  "type": "concept", "quiz", or "answer"
  "section": the section number/heading this beat belongs to (e.g. "2.3 Summarizing \
posterior inference"), or null if not tied to one section
  "title": short slide title (max 8 words)
  "bullets": 1-5 short bullet phrases (max 12 words each) — key points for "concept", the \
question restated for "quiz", the key reasoning steps for "answer"
  "narration": for "concept", {narration_min}-{narration_max} sentences thoroughly \
explaining the idea in plain conversational English; for "quiz", clearly pose the question \
and tell the listener to pause the video and think it through; for "answer", state the \
correct answer and walk through the reasoning
  "figure_hint": a short phrase describing which figure or table this beat should show, \
if any, or null

Text:
---
{section_text}
---
Return the JSON array now."""


def _group_content_headings(headings, group_size=2):
    """Group consecutive content sections (skipping bibliographic notes/
    exercises) into chunks of group_size for separate LLM calls, each
    returning [start_offset, end_offset) into the chapter text."""
    content = [h for h in headings if _is_content_heading(h)]
    groups = []
    for i in range(0, len(content), group_size):
        group = content[i:i + group_size]
        end = content[i + group_size]["offset"] if i + group_size < len(content) else None
        groups.append({"headings": group, "start": group[0]["offset"], "end": end})
    return groups


def _generate_study_script_chunked(chapter_text: str, chapter_title: str, book_title: str,
                                    headings: list, model: str, ollama_url: str,
                                    num_ctx: int, narration_min: int, narration_max: int,
                                    timeout: int):
    groups = _group_content_headings(headings, group_size=2)
    # fold any chapter-opening text before the first heading into the first chunk
    if groups and groups[0]["start"] > 0:
        groups[0]["start"] = 0

    all_beats = []
    for gi, group in enumerate(groups):
        section_text = chapter_text[group["start"]:group["end"]]
        section_labels = ", ".join(f"{h['num']} {h['title']}" for h in group["headings"])
        n_sections = len(group["headings"])
        beat_min, beat_max = 3 * n_sections + 1, 6 * n_sections + 3
        quiz_instruction = (
            "End this part with exactly one \"quiz\" beat testing the material above, "
            "immediately followed by one \"answer\" beat with the correct answer and reasoning."
        )
        prompt = CHUNK_STUDY_PROMPT.format(
            book_title=book_title, chapter_title=chapter_title, section_labels=section_labels,
            section_text=section_text, beat_min=beat_min, beat_max=beat_max,
            narration_min=narration_min, narration_max=narration_max,
            quiz_instruction=quiz_instruction,
        )
        schema = _beats_schema(STUDY_BEAT_ITEM_PROPERTIES, ["type", "title", "narration"],
                                beat_min, beat_max)
        print(f"  Part {gi + 1}/{len(groups)}: {section_labels}")
        beats = _call_ollama_for_beats(prompt, schema, model, ollama_url, num_ctx, timeout,
                                        min_items=min(3, beat_min))
        all_beats.extend(beats)
    return all_beats


def _generate_study_script_wholechapter(chapter_text: str, chapter_title: str, book_title: str,
                                         model: str, ollama_url: str, num_ctx: int,
                                         beat_min: int, beat_max: int,
                                         narration_min: int, narration_max: int, timeout: int):
    """Fallback for books where numbered sections can't be reliably detected —
    one big generation call, best-effort. Less reliable at covering an entire
    chapter than the chunked path, since a single completion can quietly
    neglect later material; used only when chunking isn't possible."""
    prompt = STUDY_GUIDE_PROMPT.format(
        book_title=book_title, chapter_title=chapter_title, chapter_text=chapter_text,
        section_list="(no numbered sections detected — divide the chapter into logical sections yourself)",
        beat_min=beat_min, beat_max=beat_max,
        narration_min=narration_min, narration_max=narration_max,
    )
    schema = _beats_schema(STUDY_BEAT_ITEM_PROPERTIES, ["type", "title", "narration"],
                            beat_min, beat_max)
    return _call_ollama_for_beats(prompt, schema, model, ollama_url, num_ctx, timeout,
                                   min_items=beat_min)


def generate_study_script(chapter_text: str, chapter_title: str, book_title: str,
                           model: str, ollama_url: str,
                           num_ctx: int = DEFAULT_NUM_CTX, max_prompt_chars: int = MAX_PROMPT_CHARS,
                           beat_min: int = 20, beat_max: int = 40,
                           narration_min: int = 3, narration_max: int = 6,
                           timeout: int = 1800):
    if len(chapter_text) > max_prompt_chars:
        print(f"  Note: chapter text is {len(chapter_text)} chars, truncating to "
              f"{max_prompt_chars} (--max-prompt-chars) before sending to the model.")
    chapter_text = chapter_text[:max_prompt_chars]

    headings = extract_section_headings(chapter_text, chapter_label=chapter_label_from_title(chapter_title))
    content_headings = [h for h in headings if _is_content_heading(h)]

    if len(content_headings) >= 2:
        # Coverage is guaranteed by construction here — every content section
        # is the explicit subject of its own chunk's LLM call — so there's no
        # need to double-check it against the (inconsistently-formatted)
        # "section" field the model puts on each beat.
        print(f"  Detected {len(content_headings)} sections — generating per-section "
              f"for guaranteed coverage.")
        beats = _generate_study_script_chunked(
            chapter_text, chapter_title, book_title, headings, model, ollama_url,
            num_ctx, narration_min, narration_max, timeout)
    else:
        print("  No reliable section numbering detected — generating the chapter in one pass "
              "(coverage of later material isn't guaranteed here).")
        beats = _generate_study_script_wholechapter(
            chapter_text, chapter_title, book_title, model, ollama_url, num_ctx,
            beat_min, beat_max, narration_min, narration_max, timeout)
        expected = {h["num"] for h in content_headings}
        covered = {b["section"].split()[0] for b in beats
                   if b.get("section") and re.match(r"^\d+\.\d+", b["section"])}
        missing = sorted(expected - covered, key=lambda s: [int(p) for p in s.split(".")])
        if missing:
            print(f"  Warning: no beat tagged with section(s) {', '.join(missing)} — "
                  f"they may have been folded into an adjacent beat, or thinly covered.")

    for beat in beats:
        beat.setdefault("type", "concept")
        beat.setdefault("section", None)
        beat.setdefault("bullets", [])
        beat.setdefault("figure_hint", None)
    return beats


def _embed_text(text: str, model: str, ollama_url: str, timeout: int = 180):
    # The embedding model may need a cold-start swap in Ollama right after a
    # big script-generation call unloads (observed ~60s) — a short timeout
    # here just means every figure silently falls back to naive ordering.
    resp = requests.post(f"{ollama_url.rstrip('/')}/api/embeddings",
                          json={"model": model, "prompt": text}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["embedding"]


def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def match_figures_to_beats(beats, figures, ollama_url: str = None, embed_model: str = "nomic-embed-text"):
    """Assign extracted figures to beats that asked for one. When figures have
    real captions (from extract_paper_content's caption detection) and an
    embedding model is reachable, matches each beat's figure_hint to the
    caption it's semantically closest to. Falls back to naive first-come-
    first-served order when no captions were found or embedding fails —
    better than nothing, but a much weaker match."""
    for beat in beats:
        beat["figure_path"] = None
    wanting = [(i, b) for i, b in enumerate(beats) if b.get("figure_hint")]
    if not wanting or not figures:
        return beats

    if ollama_url and any(f.get("caption") for f in figures):
        try:
            hint_embeds = {i: _embed_text(b["figure_hint"], embed_model, ollama_url) for i, b in wanting}
            fig_embeds = [_embed_text(f["caption"] or f["path"].name, embed_model, ollama_url)
                          for f in figures]
        except Exception as exc:
            print(f"  Note: embedding-based figure matching unavailable ({exc}), "
                  f"falling back to naive first-come-first-served order.")
        else:
            pairs = sorted(
                ((_cosine_similarity(hint_embeds[i], fe), i, fi)
                 for i, _ in wanting for fi, fe in enumerate(fig_embeds)),
                key=lambda t: -t[0],
            )
            used_beats, used_figs = set(), set()
            for _score, i, fi in pairs:
                if i in used_beats or fi in used_figs:
                    continue
                beats[i]["figure_path"] = figures[fi]["path"]
                used_beats.add(i)
                used_figs.add(fi)
            return beats

    remaining = list(figures)
    for i, _ in wanting:
        if remaining:
            beats[i]["figure_path"] = remaining.pop(0)["path"]
    return beats


# ---------------------------------------------------------------------------
# 3. Narration via Kokoro (mlx-audio, native Apple Silicon — Mac only)
# ---------------------------------------------------------------------------
_tts_model = None

def synthesize_narration(text: str, output_wav: Path, voice: str = "af_heart", speed: float = 1.0):
    global _tts_model
    try:
        from mlx_audio.tts.generate import load_model
        from mlx_audio.audio_io import write as audio_write
    except ImportError as exc:
        raise RuntimeError(
            "mlx-audio isn't installed. This step only runs on Apple Silicon: "
            "pip install mlx-audio --break-system-packages"
        ) from exc

    if _tts_model is None:
        _tts_model = load_model("mlx-community/Kokoro-82M-bf16")

    results = list(_tts_model.generate(text=text, voice=voice, speed=speed, lang_code="a"))
    audio_write(str(output_wav), results[0].audio, 24000, format="wav")
    return output_wav


def wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def append_silence(wav_path: Path, seconds: float):
    """Pad a wav with trailing silence — used after a quiz question so the
    viewer has think-time before the answer beat plays."""
    if seconds <= 0:
        return wav_path
    with wave.open(str(wav_path), "rb") as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    silence = b"\x00" * int(params.framerate * seconds) * params.sampwidth * params.nchannels
    with wave.open(str(wav_path), "wb") as w:
        w.setparams(params)
        w.writeframes(frames + silence)
    return wav_path


# ---------------------------------------------------------------------------
# 4. Slide rendering (real figure > text card)
# ---------------------------------------------------------------------------
def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSDisplay.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


BEAT_TYPE_STYLE = {
    # Plain text, not emoji — Arial/Helvetica render emoji as empty boxes.
    "quiz": {"accent": (200, 90, 20), "tag": "QUIZ — PAUSE AND THINK"},
    "answer": {"accent": (30, 140, 90), "tag": "ANSWER"},
}


def _wrap_text(text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> list:
    """Greedy word-wrap so bullet text fits a fixed column width instead of
    running past it or (as a fixed-height reservation alone would risk)
    getting silently squeezed against a figure."""
    words = text.split()
    if not words:
        return [text]
    lines, current = [], words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def render_slide(beat: dict, output_png: Path):
    img = Image.new("RGB", (VIDEO_W, VIDEO_H), BG)
    draw = ImageDraw.Draw(img)
    style = BEAT_TYPE_STYLE.get(beat.get("type"))
    accent = style["accent"] if style else ACCENT
    draw.rectangle([0, 0, 24, VIDEO_H], fill=accent)

    title_font = _load_font(64, bold=True)
    bullet_font = _load_font(40, bold=True)
    tag_font = _load_font(30, bold=True)
    section_font = _load_font(30)

    margin = 100
    title_y = 70

    if style:
        draw.text((margin, title_y), style["tag"], font=tag_font, fill=accent)
        title_y += 46
    section = beat.get("section")
    if section:
        draw.text((margin, title_y), section, font=section_font, fill=(120, 120, 120))
        title_y += 42

    draw.text((margin, title_y), beat["title"], font=title_font, fill=(20, 20, 20))
    content_y = title_y + 150  # below the title, whatever header block preceded it
    content_bottom = VIDEO_H - 70

    figure_path = beat.get("figure_path")
    bullets = beat.get("bullets", [])
    has_figure = figure_path and Path(figure_path).exists()

    # Side-by-side layout when there's a figure: bullet text gets its own
    # fixed-width column so it's always sized for readability, regardless of
    # how tall the figure is — a big figure no longer squeezes text into a
    # sliver at the bottom (or off the bottom of the frame entirely).
    if has_figure:
        text_x, text_w = margin, int(VIDEO_W * 0.40) - margin
        fig_x0 = int(VIDEO_W * 0.46)
        fig_area_w, fig_area_h = VIDEO_W - margin - fig_x0, content_bottom - content_y

        fig = Image.open(figure_path).convert("RGB")
        scale = min(fig_area_w / fig.width, fig_area_h / fig.height, 1.0)
        fig = fig.resize((max(1, int(fig.width * scale)), max(1, int(fig.height * scale))))
        fx = fig_x0 + (fig_area_w - fig.width) // 2
        fy = content_y + (fig_area_h - fig.height) // 2
        img.paste(fig, (fx, fy))
    else:
        text_x, text_w = margin, VIDEO_W - 2 * margin

    bullets_y = content_y + 20
    line_height = 50
    for bullet in bullets:
        lines = _wrap_text(bullet, bullet_font, text_w - 44, draw)
        draw.ellipse([text_x, bullets_y + 15, text_x + 16, bullets_y + 31], fill=accent)
        for li, line in enumerate(lines):
            draw.text((text_x + 40, bullets_y + li * line_height), line,
                      font=bullet_font, fill=(35, 35, 35))
        bullets_y += line_height * len(lines) + 26

    img.save(output_png)
    return output_png


# ---------------------------------------------------------------------------
# 5. Ken-Burns clip per beat + final concat
# ---------------------------------------------------------------------------
def make_beat_clip(slide_png: Path, narration_wav: Path, output_mp4: Path,
                    zoom_rate: float = 0.0006, max_zoom: float = 1.15):
    duration = wav_duration_seconds(narration_wav)
    n_frames = max(int(duration * FPS), FPS)  # at least 1 second of motion

    zoompan = (
        f"scale={VIDEO_W}:{VIDEO_H},"
        f"zoompan=z='min(zoom+{zoom_rate},{max_zoom})':d={n_frames}:s={VIDEO_W}x{VIDEO_H},"
        f"format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-framerate", str(FPS), "-i", str(slide_png),
        "-i", str(narration_wav),
        "-filter_complex", f"[0:v]{zoompan}[v]",
        "-map", "[v]", "-map", "1:a",
        "-c:v", "libx264", "-r", str(FPS), "-c:a", "aac",
        "-t", f"{duration:.3f}",
        str(output_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_mp4


def concat_clips(clip_paths, output_path: Path, work_dir: Path):
    list_file = work_dir / "concat_list.txt"
    list_file.write_text("\n".join(f"file '{p.resolve()}'" for p in clip_paths))
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
           "-c", "copy", str(output_path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_video_overview(pdf_path: str, output_path: str, model: str,
                          ollama_url: str, voice: str, keep_temp: bool = False,
                          detailed: bool = False, num_ctx: int = DEFAULT_NUM_CTX,
                          max_prompt_chars: int = MAX_PROMPT_CHARS):
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    work_dir = Path(tempfile.mkdtemp(prefix="paper2video_"))
    print(f"Working directory: {work_dir}")

    print("Extracting text and figures...")
    paper_text, figures = extract_paper_content(pdf_path, work_dir)
    print(f"  {len(paper_text)} chars of text, {len(figures)} figures extracted")

    depth = DETAILED_DEPTH if detailed else STANDARD_DEPTH
    print(f"Generating {depth['depth_desc']} script with {model}...")
    beats = generate_script(paper_text, model, ollama_url, num_ctx=num_ctx,
                             max_prompt_chars=max_prompt_chars, depth=depth)
    beats = match_figures_to_beats(beats, figures, ollama_url=ollama_url)
    print(f"  {len(beats)} beats")

    clip_paths = []
    for i, beat in enumerate(beats):
        print(f"Beat {i + 1}/{len(beats)}: {beat['title']}")
        wav_path = work_dir / f"beat_{i:02d}.wav"
        png_path = work_dir / f"beat_{i:02d}.png"
        mp4_path = work_dir / f"beat_{i:02d}.mp4"

        synthesize_narration(beat["narration"], wav_path, voice=voice)
        render_slide(beat, png_path)
        make_beat_clip(png_path, wav_path, mp4_path)
        clip_paths.append(mp4_path)

    print("Concatenating final video...")
    concat_clips(clip_paths, output_path, work_dir)
    print(f"Done: {output_path}")

    if not keep_temp:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)

    return output_path


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug or "chapter"


def build_study_guide_chapter(pdf_path: Path, chapter: dict, output_path: Path, model: str,
                               ollama_url: str, voice: str, book_title: str,
                               keep_temp: bool = False, num_ctx: int = DEFAULT_NUM_CTX,
                               max_prompt_chars: int = MAX_PROMPT_CHARS,
                               quiz_pause_seconds: float = 6.0):
    """Build one study-guide video for a single chapter dict from get_chapter_ranges()."""
    work_dir = Path(tempfile.mkdtemp(prefix="studyguide_"))
    print(f"\n=== {chapter['title']} (pages {chapter['start'] + 1}-"
          f"{chapter['end'] if chapter['end'] else 'end'}) ===")
    print(f"Working directory: {work_dir}")

    print("Extracting chapter text and figures...")
    chapter_text, figures = extract_paper_content(
        pdf_path, work_dir, start_page=chapter["start"], end_page=chapter["end"])
    print(f"  {len(chapter_text)} chars of text, {len(figures)} figures extracted")

    print(f"Generating study-guide script with {model}...")
    beats = generate_study_script(chapter_text, chapter["title"], book_title, model, ollama_url,
                                   num_ctx=num_ctx, max_prompt_chars=max_prompt_chars)
    beats = match_figures_to_beats(beats, figures, ollama_url=ollama_url)
    print(f"  {len(beats)} beats "
          f"({sum(1 for b in beats if b['type'] == 'quiz')} quizzes)")

    clip_paths = []
    for i, beat in enumerate(beats):
        print(f"Beat {i + 1}/{len(beats)} [{beat['type']}]: {beat['title']}")
        wav_path = work_dir / f"beat_{i:03d}.wav"
        png_path = work_dir / f"beat_{i:03d}.png"
        mp4_path = work_dir / f"beat_{i:03d}.mp4"

        synthesize_narration(beat["narration"], wav_path, voice=voice)
        if beat["type"] == "quiz":
            append_silence(wav_path, quiz_pause_seconds)
        render_slide(beat, png_path)
        make_beat_clip(png_path, wav_path, mp4_path)
        clip_paths.append(mp4_path)

    print("Concatenating final video...")
    concat_clips(clip_paths, output_path, work_dir)
    print(f"Done: {output_path}")

    if not keep_temp:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)

    return output_path


def build_study_guide(pdf_path: str, model: str, ollama_url: str, voice: str,
                       chapter_selector: str = None, all_chapters: bool = False,
                       output_path: str = None, output_dir: str = "study_guide_videos",
                       book_title: str = None, keep_temp: bool = False,
                       num_ctx: int = DEFAULT_NUM_CTX, max_prompt_chars: int = MAX_PROMPT_CHARS,
                       quiz_pause_seconds: float = 6.0):
    pdf_path = Path(pdf_path)
    book_title = book_title or pdf_path.stem.replace("_", " ").replace("-", " ").title()
    chapters = get_chapter_ranges(pdf_path)

    if all_chapters:
        targets = [c for c in chapters if is_content_chapter(c["title"])]
        if not targets:
            raise ValueError("No 'Chapter N' / 'Appendix X' bookmarks found in this PDF's ToC.")
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Found {len(targets)} chapters/appendices. Output dir: {out_dir}")
        results, failures = [], []
        for idx, chapter in enumerate(targets):
            chapter_out = out_dir / f"{idx + 1:02d}_{_slugify(chapter['title'])}.mp4"
            print(f"\n[{idx + 1}/{len(targets)}] {chapter['title']}")
            try:
                results.append(build_study_guide_chapter(
                    pdf_path, chapter, chapter_out, model, ollama_url, voice, book_title,
                    keep_temp=keep_temp, num_ctx=num_ctx, max_prompt_chars=max_prompt_chars,
                    quiz_pause_seconds=quiz_pause_seconds))
            except Exception as exc:
                # One bad chapter (e.g. a model hiccup that exhausts all JSON
                # retries) shouldn't sink an unattended multi-hour batch —
                # log it and keep going so later chapters still get built.
                print(f"  FAILED: {chapter['title']}: {exc}")
                failures.append((chapter["title"], str(exc)))

        print(f"\nDone: {len(results)}/{len(targets)} chapters built into {out_dir}")
        if failures:
            print(f"{len(failures)} chapter(s) failed — rerun these individually with --chapter:")
            for title, err in failures:
                print(f"  - {title}: {err}")
        return results

    if not chapter_selector:
        raise ValueError("Pass --chapter <number or title> or use --all-chapters.")
    chapter = select_chapter(chapters, chapter_selector)
    out_path = Path(output_path) if output_path else Path(f"{_slugify(chapter['title'])}.mp4")
    return build_study_guide_chapter(
        pdf_path, chapter, out_path, model, ollama_url, voice, book_title,
        keep_temp=keep_temp, num_ctx=num_ctx, max_prompt_chars=max_prompt_chars,
        quiz_pause_seconds=quiz_pause_seconds)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", help="Path to the paper PDF")
    parser.add_argument("--output", default=None,
                         help="Output video path (default: overview.mp4, or "
                              "<chapter-slug>.mp4 in study-guide single-chapter mode)")
    parser.add_argument("--model", default="qwen2.5:32b-instruct",
                         help="Ollama model tag")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice id")
    parser.add_argument("--keep-temp", action="store_true",
                         help="Keep the working dir (figures, wavs, per-beat clips)")
    parser.add_argument("--detailed", action="store_true",
                         help="Produce a longer, more granular script: more beats, "
                              "deeper per-beat narration, explicit multi-beat Methods "
                              "coverage, and a thorough Discussion summary")
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX,
                         help=f"Ollama context window in tokens (default {DEFAULT_NUM_CTX})")
    parser.add_argument("--max-prompt-chars", type=int, default=MAX_PROMPT_CHARS,
                         help=f"Max chars of paper text sent to the model "
                              f"(default {MAX_PROMPT_CHARS}); raise together with --num-ctx")

    study_group = parser.add_argument_group("study guide mode (textbooks)")
    study_group.add_argument("--study-guide", action="store_true",
                              help="Treat the PDF as a textbook: split into chapters via its "
                                   "embedded table of contents, and produce a tutor-style "
                                   "study video per chapter with periodic quiz/answer beats")
    study_group.add_argument("--chapter",
                              help="Chapter to build, e.g. '3' or 'Hierarchical Models' "
                                   "(matched against the PDF's bookmarks). Use --list-chapters "
                                   "to see options")
    study_group.add_argument("--all-chapters", action="store_true",
                              help="Build a study video for every Chapter/Appendix in the ToC "
                                   "(can take a very long time for a full textbook)")
    study_group.add_argument("--list-chapters", action="store_true",
                              help="Print the chapters/sections found in the PDF's table of "
                                   "contents and exit")
    study_group.add_argument("--book-title",
                              help="Human-readable book title for the script prompt "
                                   "(default: derived from the PDF filename)")
    study_group.add_argument("--output-dir", default="study_guide_videos",
                              help="Output directory when using --all-chapters "
                                   "(default: study_guide_videos)")
    study_group.add_argument("--quiz-pause-seconds", type=float, default=6.0,
                              help="Silent think-time held after each quiz question, before "
                                   "the answer beat plays (default: 6.0)")
    args = parser.parse_args()

    if args.list_chapters:
        chapters = get_chapter_ranges(Path(args.pdf))
        for i, c in enumerate(chapters):
            pages = f"{c['start'] + 1}-{c['end'] if c['end'] else 'end'}"
            print(f"{i:>3}  p.{pages:<12} {c['title']}")
        return

    if args.study_guide:
        build_study_guide(
            args.pdf, args.model, args.ollama_url, args.voice,
            chapter_selector=args.chapter, all_chapters=args.all_chapters,
            output_path=args.output, output_dir=args.output_dir, book_title=args.book_title,
            keep_temp=args.keep_temp, num_ctx=args.num_ctx,
            max_prompt_chars=args.max_prompt_chars,
            quiz_pause_seconds=args.quiz_pause_seconds)
        return

    build_video_overview(args.pdf, args.output or "overview.mp4", args.model, args.ollama_url,
                          args.voice, args.keep_temp, detailed=args.detailed,
                          num_ctx=args.num_ctx, max_prompt_chars=args.max_prompt_chars)


if __name__ == "__main__":
    main()
