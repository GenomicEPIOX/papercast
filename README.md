# papercast

Turn a paper or textbook PDF into a narrated video. Entirely local — no cloud APIs, no rate limits, no upload of your PDF anywhere. Point it at a paper and get a NotebookLM-style video overview back; point it at a textbook and get a tutor-style study guide, chapter by chapter, that quizzes you as it goes.

Built to run end-to-end on a single Mac with Apple Silicon: a local LLM via [Ollama](https://ollama.com) writes the script, [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) narrates it natively on-device via `mlx-audio`, and `ffmpeg` assembles the final mp4.

## Example

[![Papercast demo](https://img.youtube.com/vi/u3SGV_XZ0-Y/maxresdefault.jpg)](https://youtu.be/u3SGV_XZ0-Y)

Click to watch on YouTube — a paper turned into a narrated video overview, generated entirely locally.

## What it does

**Paper mode** — feed it a research paper, get back a narrated video that walks through the motivation, methods, results, and discussion, illustrated with the paper's *actual* figures.

**Study-guide mode** — feed it a textbook, and it splits the book into chapters using the PDF's own table of contents, then builds one video per chapter: a patient walkthrough of every section, with worked examples explained in depth and a quiz-then-answer beat dropped in every few concepts to check you're actually retaining it.

Both modes generate a full script as structured JSON (not just a wall of narration), turn each beat into a slide — bold key-point text alongside the real figure when there is one — and stitch it all into a Ken-Burns-style video with narrated audio.

## Why this exists

Tools like NotebookLM's video overviews are great, but they're cloud-only, rate-limited, and your paper leaves your machine. This does the same job locally: your PDFs, your GPU, your model choice, no usage caps.

## How it works

1. **Extract** — pulls text out of the PDF, and extracts the *real* figures. Most journal figures are vector graphics with no embedded raster image at all, so naive "extract embedded images" approaches either miss them completely or pick up junk (a flattened supplementary page split into dozens of meaningless slivers, author headshots, logos). Instead, papercast finds each figure's caption in the page text (`Fig. 2 |`, `Extended Data Fig. 1 |`, ...) and renders the page region around it directly — which works for vector and raster content alike.
2. **Script** — sends the text to a local LLM via Ollama, with the response grammar-constrained to a JSON schema so it can't return malformed or wrong-shaped output. Study-guide mode splits a chapter into per-section chunks and generates one part at a time, so coverage of the whole chapter is guaranteed by construction rather than hoping one giant completion remembers to get to the end.
3. **Match figures to beats** — each beat's figure request is embedded (via a local `nomic-embed-text` model) and matched to whichever figure's real caption is the closest semantic fit, instead of just handing figures out in page order.
4. **Narrate** — synthesizes every beat's narration with Kokoro TTS, running natively on Apple Silicon via `mlx-audio`.
5. **Render** — draws a slide per beat: bold bullet text in its own column, the matched figure alongside it, quiz beats get a distinct visual treatment and a silent think-time pause before the answer.
6. **Assemble** — turns each slide into a pan/zoom video clip timed to its narration length, then concatenates everything into the final mp4.

## Requirements

- macOS on Apple Silicon — the narration step (`mlx-audio`) is Mac-only.
- [ffmpeg](https://ffmpeg.org): `brew install ffmpeg`
- [espeak-ng](https://github.com/espeak-ng/espeak-ng): `brew install espeak-ng` — Kokoro's `misaki` text-processing backend falls back to it for out-of-vocabulary words.
- [Ollama](https://ollama.com) running locally, with:
  - a capable instruct model for script generation, e.g. `ollama pull qwen2.5:32b-instruct`
  - `nomic-embed-text` for figure matching: `ollama pull nomic-embed-text` (matching just falls back to naive order without it)
- Python packages — see `requirements.txt`.

## Install

```bash
git clone https://github.com/GenomicEPIOX/papercast.git
cd papercast
pip install -r requirements.txt --break-system-packages
```

> **Note:** install plain `misaki`, not `misaki[en]` — the `[en]` extra pulls in `spacy`'s `blis`/`thinc` dependencies, which try to build from source and can fail on newer Python. `requirements.txt` already has this right.

## Quickstart

```bash
# a paper -> one narrated overview video
python paper_to_video.py your_paper.pdf --output overview.mp4 \
    --model qwen2.5:32b-instruct --voice af_heart

# ...or go deep: more beats, more detail on methods and discussion
python paper_to_video.py your_paper.pdf --detailed --output overview.mp4 \
    --model qwen2.5:32b-instruct --voice af_heart
```

```bash
# a textbook -> see what chapters were detected
python paper_to_video.py textbook.pdf --list-chapters

# ...one chapter as a study guide, with quizzes
python paper_to_video.py textbook.pdf --study-guide --chapter 3 \
    --model qwen2.5:32b-instruct --voice af_heart --book-title "My Textbook"

# ...or the whole book, one video per chapter (can take hours)
python paper_to_video.py textbook.pdf --study-guide --all-chapters \
    --model qwen2.5:32b-instruct --voice af_heart --book-title "My Textbook" \
    --output-dir my_textbook_study_guide
```

## Usage — paper mode

| Flag | Default | Description |
|---|---|---|
| `pdf` | — | Path to the paper PDF (positional) |
| `--output` | `overview.mp4` | Output video path |
| `--model` | `qwen2.5:32b-instruct` | Ollama model tag used for script generation |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL |
| `--voice` | `af_heart` | Kokoro voice id |
| `--keep-temp` | off | Keep the working dir (figures, wavs, per-beat clips) for debugging |
| `--detailed` | off | Longer, more granular script: 16-28 beats instead of 6-10, 3-6 narration sentences per beat instead of 1-3, the Methods broken into one beat per technique/step, and 2-3 beats thoroughly covering the Discussion |
| `--max-prompt-chars` | `200000` | Max chars of paper text sent to the model. Methods/Discussion often sit at the very end of a paper — if this truncates before them, raise it (and `--num-ctx` with it) |
| `--num-ctx` | `65536` | Ollama context window in tokens. Keep roughly in step with `--max-prompt-chars` (~4 chars/token, plus headroom for the prompt and JSON output) |

For a thorough walkthrough of a long paper: `--detailed --max-prompt-chars 300000 --num-ctx 98304` (adjust down if your model's max context or RAM can't cover that).

## Usage — study-guide mode (textbooks)

Splits a textbook into chapters using its embedded table of contents/bookmarks, then produces a tutor-style video per chapter: section-by-section walkthroughs following the real headings in the text, with a quiz beat roughly every 3-5 concepts — a question, a silent pause to think, then an answer beat with the reasoning.

| Flag | Default | Description |
|---|---|---|
| `--study-guide` | off | Switch to study-guide mode |
| `--chapter` | — | Chapter to build, e.g. `3` or `Hierarchical Models` (matched against the PDF's bookmarks) |
| `--all-chapters` | off | Build a video for every `Chapter`/`Appendix` bookmark. A single chapter failing (e.g. a model hiccup that exhausts its JSON retries) is logged and skipped rather than aborting the batch — failures are listed at the end so you can rerun just those with `--chapter` |
| `--list-chapters` | — | Print the chapters/sections found in the PDF's table of contents and exit (needs the PDF to have embedded bookmarks) |
| `--book-title` | derived from filename | Human-readable book title used in the script prompt |
| `--output-dir` | `study_guide_videos` | Output directory when using `--all-chapters` |
| `--quiz-pause-seconds` | `6.0` | Silent think-time held after each quiz question, before the answer beat plays |
| `--output` | `<chapter-slug>.mp4` | Output path for single-chapter mode |

Also respects `--model`, `--ollama-url`, `--voice`, `--keep-temp`, `--max-prompt-chars`, `--num-ctx` from paper mode.

### How full chapter coverage is guaranteed

Asking a local model to "explain this whole chapter" in one completion reliably drifts: it explains the first few sections in real depth, then quietly thins out or stops before the end — even when explicitly told the full section list and told not to. So study-guide mode instead:

1. Detects the chapter's real numbered section headings from the extracted text (e.g. `2.1` ... `2.9`), scoped to that chapter's own leading number so a stray decimal in a data table doesn't get mistaken for a heading.
2. Groups sections into chunks of ~2 and makes a **separate LLM call per chunk**, each covering only its own sections — coverage is guaranteed by construction, not by hoping one giant completion remembers to finish.
3. Falls back to a single whole-chapter prompt only when no reliable section numbering could be detected.

## Known limitations

- **Figure crops include page chrome** — a figure is rendered as the page region around its caption, so the crop can include the header/footer or, on a page with multiple figures, whatever sits directly beside them.
- **Caption detection is regex-based** — tuned for common styles (`Fig. 2 |`, `Figure 3:`, `Extended Data Fig. 1 |`). A very different caption style may fall back to embedded-raster extraction or plain page-order matching.
- **Study-guide mode needs embedded bookmarks** — a textbook PDF without a table of contents in its metadata can't be auto-split into chapters.
- **Figure matching needs `nomic-embed-text`** — without it (or if the call fails), figures fall back to naive first-come-first-served order.
- **Local-model output quality varies** — schema-constrained JSON and retries push hard against thin or oddly-organized output, but nothing guarantees perfection. Spot-check a new paper or book before trusting a long `--all-chapters` run to it unattended.

## Extending

- **Two-host dialogue instead of single narrator** — [Dia-1.6B](https://github.com/nari-labs/dia) is built for tagged multi-speaker scripts and runs on MPS via `transformers`, at the cost of speed versus the native MLX Kokoro path.
- **Route through MLX-LM instead of Ollama** — swap the request in `generate_script()` for MLX-LM's OpenAI-compatible endpoint (`/v1/chat/completions`); the exact change is commented inline in the script.
- **Vision-model figure matching** — the current match compares hint text to caption text, not to the figure's actual pixels. A vision-capable Ollama model (`qwen2.5-vl`, `llava`) could compare a hint directly against a rendered thumbnail for captions that don't describe the figure well.

## License

[MIT](LICENSE)
