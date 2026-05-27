"""Three-platform caption generation via Claude SDK.

Two-phase flow matching the Curator agent's approach:

  1) `scout_clip(file_path)` — extracts 12-30 keyframes + transcript,
     asks Claude to do PART A (factual read) + PART B (angle question
     with 2-3 concrete options). No captions yet — gives the user a
     chance to pick the angle / specify text preferences.

  2) `generate_captions(file_path, angle)` — with the user's angle hint,
     re-runs the same frame extraction (cached on disk) and asks Claude
     to compose three captions (Instagram / TikTok / YouTube Shorts) in
     the curator's exact format.

Uses Claude Agent SDK with allowed_tools=["Read"] so the model can open
each JPEG keyframe by absolute path. Subscription auth — no API key.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from claude_agent_sdk import (  # type: ignore
    ClaudeAgentOptions,
    ClaudeSDKClient,
)

from idea import prepare_inputs as extract_frames_and_transcript

# ────────────────────────────────────────────────────────────────────
# Curator's CAPTION_SPEC — ported verbatim. Kept inline so clipper is
# self-contained and doesn't depend on agent/v1 being on the PYTHONPATH.
# ────────────────────────────────────────────────────────────────────
CAPTION_SPEC = """Compose THREE captions for the SAME video — one per
platform — in a SINGLE reply, in this exact order: INSTAGRAM →
TIKTOK → YOUTUBE SHORTS. End with one Russian refinement line.
ENGLISH ONLY for caption bodies + hashtags. Russian only on the
refinement line at the very bottom.

══════════════════ STYLE RULE — NO DASHES ══════════════════

Caption bodies and hooks must NOT contain dashes:
  ✗ em-dash  —
  ✗ en-dash  –
  ✗ hyphen-as-pause (e.g. "great clip - watch this")

Replace with: comma, period, semicolon, colon, or just split into
another sentence. Hyphens INSIDE words (`pop-flash`, `1v4`) and inside
hashtags (`#1v4Retake`) are FINE — only standalone dashes-as-punctuation
are banned.

This rule applies to: IG hook, IG body (all 3 paragraphs), TikTok body,
YouTube Shorts description. The Figma title may also not contain dashes.

The Russian refinement line at the very bottom IS allowed to have a
dash — it's a fixed template string outside the user-facing caption.

══════════════════ EVIDENCE-ONLY RULE (READ FIRST) ══════════════════

ANALYZE THIS CLIP ALONE. Your training memory about CS2 tournaments,
team rosters, match outcomes, player histories, event locations,
sponsor abbreviations — IGNORE ALL OF IT for this task.

EVERY specific fact in the caption must come from EXACTLY ONE of:
  (S1) the spoken transcript provided below
  (S2) text visible on the frames you READ (scoreboard, overlays,
       watermarks like `PGL CAC` / `BLAST` / `IEM`, player tags,
       map name in the bottom corner, kill feed)
  (S3) the filename of the clip

If a fact is not in S1/S2/S3, DO NOT WRITE IT. Specifically forbidden:
  ✗ Expanding abbreviations from training memory (`CAC` → `Astana`).
  ✗ Naming a tournament/event/stage/year not explicitly on-screen.
  ✗ Naming a city/venue not on-screen.
  ✗ Naming opposing team if not on-screen.
  ✗ Inventing a series score not shown on screen.
  ✗ Player history claims ("third ace this Major").

Better to write less specifically than to write wrong specifically.

══════════════════ INSTAGRAM ══════════════════

  IG-1  HOOK HEADER
    One line, max ~60 chars, ONE emoji, ends with `👇`.
  IG-2  BODY
    EXACTLY 3 paragraphs separated by ONE blank line.
    Each paragraph 70-110 words.
    ¶1 context / setup
    ¶2 what actually happens in the clip
    ¶3 takeaway / why it matters
    Max 2 emoji TOTAL across the 3 paragraphs.
    NO hashtags in the body. NO bullet lists. NO `🔑 Keywords`.
  IG-3  HASHTAG LINE
    ONE line, EXACTLY 5 hashtags separated by single spaces.

══════════════════ TIKTOK ══════════════════

  TT-1  BODY
    Single block (1-3 short sentences). HARD CAP: 200 characters
    (body only — hashtags do NOT count).
    Hook-forward — punchy first 5 words; ends with a tease/question.
  TT-2  HASHTAG LINE
    ONE line, EXACTLY 5 hashtags.

══════════════════ YOUTUBE SHORTS ══════════════════

  YT-1  DESCRIPTION + TAGS (ONE block)
    `<hype phrase> <#tag1> <#tag2> <#tag3> <#tag4> <#tag5>`
    HARD CAP: 100 characters TOTAL.

══════════════════ FIGMA TITLE ══════════════════

  FT-1  FIGMA TITLE
    A short 3-5 word title for use in Figma / thumbnail design — punchy,
    title-case, no emoji, no hashtags, no quotes, no period.
    Distills the clip's essence into a thumbnail-ready phrase.
    Examples (shape only):
      "Donk's 1v4 Mirage Heist"
      "NiKo Refuses 1-9 Collapse"
      "Ace From Eight HP"

══════════════════ RU REFINEMENT (always last line) ══════════════════

  `(Готов переделать — скажи как: короче, юморнее, под одну платформу и т.д.)`

══════════════════ HASHTAG TIER LADDER ══════════════════

Pick 5 per platform, in priority order. Skip a tier if evidence absent.
  T1 PLAY — hyper-niche compound (`#donk1v4`, `#deagleAce`).
  T2 TEAM(S) — visible on scoreboard / logo / overlay only.
  T3 MAP — visible from HUD / loading screen.
  T4 TOURNAMENT — literal on-screen text only (no expansion).
  T5 GAME — always `#CS2` or `#CounterStrike2`.

──── FORBIDDEN HASHTAGS ────
  #FPSGaming #PCGaming #VideoGames #Gamer #Gaming #GamingContent
  #Esports (alone) #PC #viral #fyp #foryou #trending #shorts
  #reels #explore #insta #instagram

══════════════════ OUTPUT FORMAT ══════════════════

Use these EXACT section headers (they let the UI render as separate
copyable cards):

**Figma**

<3-5 word title, no quotes/emoji/period>

**Instagram**

<IG hook>

<IG ¶1>

<IG ¶2>

<IG ¶3>

<#5 hashtags>

**TikTok**

<TT body>

<#5 hashtags>

**YouTube Shorts**

<line with description + 5 hashtags ≤ 100 chars total>

(Готов переделать — скажи как: короче, юморнее, под одну платформу и т.д.)
"""


SCOUT_SPEC = """Ты анализируешь короткий клип из CS2 (≤90 секунд)
для боса-контент-криейтора. Тебе пришли:
  • Список абсолютных путей к JPEG-кадрам клипа
  • Транскрипт речи (если есть)
  • Имя файла

Открой КАЖДЫЙ кадр через Read и прочитай. Затем напиши ОДНО короткое
сообщение на русском в формате:

  PART A — Что вижу (2-4 предложения, фактически):
  • Карта (из HUD / лоадинга если видно)
  • Команды на overlay / scoreboard
  • Что произошло (главное действие в клипе)
  • Необычные оверлеи / водяные знаки (литерально, БЕЗ расшифровки)

  PART B — Какой акцент? (2-3 варианта, конкретно, на основе PART A)
  Пример: "(а) ace-серия игрока X, (б) последний пик с pop-flash,
  (в) общий highlight, или скажи своё (тон/длина/юмор/без эмодзи)."

EVIDENCE-only: то что не видно на кадрах и не упомянуто в транскрипте —
НЕ пиши. Не расшифровывай аббревиатуры (PGL CAC не в Astana 2024).
Никаких caption'ов в этом сообщении — только scout."""


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
async def _claude_oneshot(system_prompt: str, user_msg: str,
                          allowed_tools: list[str]) -> str:
    """Run a single Claude SDK round-trip; collect all assistant text."""
    opts = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
    )
    chunks: list[str] = []
    async with ClaudeSDKClient(options=opts) as client:
        await client.query(user_msg)
        async for msg in client.receive_response():
            blocks = getattr(msg, "content", None)
            if isinstance(blocks, list):
                for b in blocks:
                    t = getattr(b, "text", None)
                    if isinstance(t, str):
                        chunks.append(t)
            elif isinstance(blocks, str):
                chunks.append(blocks)
    return "".join(chunks).strip()


def _build_user_payload(file_path: Path, frames: list[str],
                        transcript: str, angle: str = "") -> str:
    parts: list[str] = []
    parts.append(f"Filename: {file_path.name}")
    parts.append("")
    parts.append("Frame paths (read EACH with the Read tool):")
    for p in frames:
        parts.append(f"  - {p}")
    parts.append("")
    parts.append("Transcript:")
    parts.append(transcript.strip() or "(silent clip — describe from frames alone)")
    if angle:
        parts.append("")
        parts.append("Angle / preferences chosen by the boss:")
        parts.append(angle.strip())
    return "\n".join(parts)


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────
async def scout_clip(file_path: Path) -> dict:
    """Phase 1 — extract frames + transcript, ask Claude for PART A
    (factual read) + PART B (angle question with 2-3 options).
    Returns: { duration_s, frame_count, transcript_preview, scout_text }
    """
    data = extract_frames_and_transcript(file_path)
    frames = data["frames"]
    transcript = data["transcript"]
    payload = _build_user_payload(file_path, frames, transcript)
    payload += "\n\nWrite PART A + PART B ONLY (no captions yet)."
    scout_text = await _claude_oneshot(
        system_prompt=SCOUT_SPEC,
        user_msg=payload,
        allowed_tools=["Read"],
    )
    return {
        "duration_s":         data.get("duration_s", 0),
        "frame_count":        len(frames),
        "transcript_chars":   len(transcript),
        "transcript_preview": transcript[:400],
        "scout_text":         scout_text,
    }


async def generate_captions(file_path: Path, angle: str = "") -> dict:
    """Phase 2 — with optional angle, produce 4 cards (Figma title +
    3 platforms). Result is also appended to per-clip history.

    Returns: { duration_s, frame_count, captions_text, ts }
    """
    data = extract_frames_and_transcript(file_path)
    frames = data["frames"]
    transcript = data["transcript"]
    payload = _build_user_payload(file_path, frames, transcript, angle)
    payload += ("\n\nNow READ each frame and compose: Figma title (3-5 words) "
                "then all three platform captions per the spec. Use the exact "
                "section headers `**Figma** / **Instagram** / **TikTok** / "
                "`**YouTube Shorts**`.")
    captions_text = await _claude_oneshot(
        system_prompt=CAPTION_SPEC,
        user_msg=payload,
        allowed_tools=["Read"],
    )
    ts = _save_history_entry(file_path, angle, captions_text)
    return {
        "duration_s":    data.get("duration_s", 0),
        "frame_count":   len(frames),
        "captions_text": captions_text,
        "ts":            ts,
    }


# ────────────────────────────────────────────────────────────────────
# History — per-clip JSON of past generations.
# Stored at `<media-parent>/.captions/<stem>.json` so it travels with
# the media; when the clip is deleted, the history sidecar can go too.
# ────────────────────────────────────────────────────────────────────
import json
import time


def _history_path(file_path: Path) -> Path:
    return file_path.parent / ".captions" / f"{file_path.stem}.json"


def _save_history_entry(file_path: Path, angle: str, text: str) -> float:
    """Append a fresh entry, return its ts (used as id)."""
    f = _history_path(file_path)
    f.parent.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    if f.exists():
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(d, dict) and isinstance(d.get("items"), list):
                items = d["items"]
        except Exception:
            items = []
    ts = time.time()
    items.append({
        "ts":    ts,
        "angle": angle or "",
        "text":  text or "",
    })
    # Cap at 50 most recent so the file never grows unbounded.
    items = items[-50:]
    f.write_text(
        json.dumps({"items": items}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return ts


def load_history(file_path: Path) -> list[dict]:
    f = _history_path(file_path)
    if not f.exists():
        return []
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(d, dict) and isinstance(d.get("items"), list):
            return d["items"]
    except Exception:
        pass
    return []


def delete_history_entry(file_path: Path, ts: float) -> bool:
    items = load_history(file_path)
    new_items = [it for it in items if abs(it.get("ts", 0) - ts) > 0.001]
    if len(new_items) == len(items):
        return False
    f = _history_path(file_path)
    f.write_text(
        json.dumps({"items": new_items}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return True
