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

Owns `extract_frames_and_transcript()` (the EXTRACT + TRANSCRIBE half
of the pipeline) — this was previously imported from the legacy Gemini
module `idea.py`, which has been removed.
"""
from __future__ import annotations

import asyncio
import json as _json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from claude_agent_sdk import (  # type: ignore
    ClaudeAgentOptions,
    ClaudeSDKClient,
)


# ────────────────────────────────────────────────────────────────────
# Lalo (marketer agent at :8771) HTTP client — generate-phase backend.
# Scout phase stays on local Claude SDK; generate phase delegates to
# Lalo so we get the marketer-grade persona + tool access + reasoning.
# Auto-creates a dedicated "Clipper captions" project on first call.
# ────────────────────────────────────────────────────────────────────
LALO_BASE = "http://127.0.0.1:8771"
LALO_PROJECT_NAME = "Clipper captions"           # default persona
LALO_PROJECT_NAME_PROKOP = "Prokop captions"     # 77prokop77-voice persona
# Cache is keyed by project name so both personas can coexist.
_LALO_PROJECT_ID_CACHE: dict[str, str] = {}


def _lalo_http_json(method: str, path: str, payload: Optional[dict] = None,
                    timeout: int = 30) -> dict:
    """Sync HTTP — wrapped in to_thread by async callers."""
    url = LALO_BASE + path
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = _json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="ignore")
            return _json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="ignore")
        except Exception:
            detail = ""
        return {"_error": f"HTTP {e.code}: {detail[:200]}"}
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


async def _ensure_lalo_project(name: str = LALO_PROJECT_NAME) -> str:
    """Return the project id of the given Lalo project, creating it if
    needed. Cached per-name so default + prokop personas coexist without
    stepping on each other."""
    cached = _LALO_PROJECT_ID_CACHE.get(name)
    if cached:
        return cached
    listing = await asyncio.to_thread(_lalo_http_json, "GET", "/api/projects")
    if "_error" in listing:
        raise RuntimeError(f"Lalo unreachable: {listing['_error']}")
    for p in listing.get("items", []):
        if p.get("name") == name:
            pid = p.get("id") or p.get("pid") or p.get("_id")
            if pid:
                _LALO_PROJECT_ID_CACHE[name] = pid
                return pid
    # Not found — create
    created = await asyncio.to_thread(
        _lalo_http_json, "POST", "/api/projects",
        {"name": name})
    if "_error" in created:
        raise RuntimeError(f"Lalo project create failed: {created['_error']}")
    proj = created.get("project") or {}
    pid = proj.get("id") or proj.get("pid")
    if not pid:
        raise RuntimeError(f"Lalo project create returned no id: {created}")
    _LALO_PROJECT_ID_CACHE[name] = pid
    return pid


async def call_lalo(message: str, *, poll_interval: float = 2.0,
                    timeout: float = 300.0,
                    project_name: str = LALO_PROJECT_NAME) -> str:
    """Send a message to Lalo and wait for the reply text.
    Returns the assistant's reply_text from the new ui_turn.
    `project_name` picks the persona (default vs "Prokop captions")."""
    pid = await _ensure_lalo_project(project_name)
    # Snapshot current state so we know which ui_turn is new
    pre_state = await asyncio.to_thread(
        _lalo_http_json, "GET", f"/api/state?project={pid}&since=0")
    if "_error" in pre_state:
        raise RuntimeError(f"Lalo /api/state failed: {pre_state['_error']}")
    pre_turn_count = len(pre_state.get("ui_turns", []))
    # POST message
    posted = await asyncio.to_thread(
        _lalo_http_json, "POST", "/api/chat",
        {"project": pid, "message": message, "images": []})
    if "_error" in posted:
        raise RuntimeError(f"Lalo /api/chat failed: {posted['_error']}")
    # Poll for completion
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        await asyncio.sleep(poll_interval)
        st = await asyncio.to_thread(
            _lalo_http_json, "GET", f"/api/state?project={pid}&since=0")
        if "_error" in st:
            continue  # transient
        if st.get("turn_error"):
            raise RuntimeError(f"Lalo turn error: {st['turn_error']}")
        turns = st.get("ui_turns", [])
        if len(turns) > pre_turn_count and not st.get("turn_active"):
            new = turns[-1]
            text = (new.get("reply_text") or "").strip()
            if text:
                return text
            return _json.dumps(new)[:500]  # fallback
    raise RuntimeError(f"Lalo response timed out after {timeout}s")

import core
import subs_burn


# ────────────────────────────────────────────────────────────────────
# Frame extraction + transcript prep (formerly in idea.py::prepare_inputs)
# ────────────────────────────────────────────────────────────────────
def _vtt_to_plain(vtt: Path) -> str:
    """Strip a WebVTT file to plain text — drops timecodes, indices,
    inline tags, and the WEBVTT header line."""
    lines: list[str] = []
    try:
        for raw in vtt.read_text("utf-8", errors="ignore").splitlines():
            s = raw.strip()
            if not s: continue
            if s.upper().startswith("WEBVTT"): continue
            if "-->" in s: continue
            if s.isdigit(): continue
            s = re.sub(r"<[^>]+>", "", s)
            lines.append(s)
    except Exception:
        return ""
    return " ".join(lines).strip()


def extract_frames_and_transcript(file_path: Path,
                                  n_frames: int = 0,
                                  language: Optional[str] = None) -> dict:
    """Run the EXTRACT + TRANSCRIBE half of the caption pipeline and
    stop — no LLM call. Returns {transcript, frames, duration_s}.

    `language` pins the Whisper language for the transcript pass.
      None       → legacy path: reuse ANY .vtt sidecar, fall back to
                    English-pinned Whisper (historical default for CS2
                    English streams).
      "uk"/"en"/… → persona-scoped: prefer a sidecar matching this
                    language; else run Whisper pinned to `language`.

    `n_frames=0` → auto: clamp(12, dur_sec / 2, 30). Floor 12 keeps short
    reels dense (every beat visible); ~1 frame per 2 sec in the middle;
    cap 30 prevents long matches from blowing up the caller's vision
    budget. Explicit positive `n_frames` overrides the formula.

    Frames are written into a sibling `.frames/<stem>/fN.jpg` cache
    folder so they don't pollute the user's "Open folder" view — the
    media file stays alone in its parent directory."""
    if not file_path.exists():
        raise ValueError(f"file missing: {file_path}")
    # 1. Frames — JPEG cache in a per-media subfolder.
    probe = subprocess.run(
        [core.FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)],
        capture_output=True, text=True)
    try:
        dur = float((probe.stdout or "0").strip())
    except ValueError:
        dur = 0.0
    if dur <= 0:
        raise RuntimeError("ffprobe failed to read duration")
    if n_frames <= 0:
        n_frames = max(12, min(30, int(dur / 2)))
    frames_dir = file_path.parent / ".frames" / file_path.stem
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[str] = []
    for i in range(1, n_frames + 1):
        t = dur * i / (n_frames + 1)
        out_p = frames_dir / f"f{i}.jpg"
        r = subprocess.run(
            [core.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
             "-ss", f"{t:.2f}", "-i", str(file_path),
             "-vframes", "1", "-vf", "scale='min(1280,iw)':-2",
             "-q:v", "4", str(out_p)],
            capture_output=True, timeout=30)
        if r.returncode == 0 and out_p.exists():
            frame_paths.append(str(out_p.resolve()))
    if not frame_paths:
        raise RuntimeError("frame extraction yielded nothing")
    # 2. Transcript — prefer .vtt sidecar, fall back to Whisper.
    # When `language` is set, prefer a sidecar tagged with that language
    # (`.whisper.<lang>.*.vtt`) so a stale English sidecar from an
    # earlier IDEA scout doesn't get reused for a Ukrainian Prokop pass.
    transcript = ""
    existing = subs_burn.find_subtitle_files(file_path)
    if language:
        lang = language.lower()
        matching = [p for p in existing if f".whisper.{lang}." in p.name
                                          or p.name.endswith(f".{lang}.vtt")]
        if matching:
            transcript = _vtt_to_plain(matching[0])
    elif existing:
        transcript = _vtt_to_plain(existing[0])
    if not transcript.strip():
        try:
            if language:
                subs_burn.generate_subs_via_whisper(
                    file_path, language=language)
            else:
                subs_burn.generate_subs_via_whisper(file_path)
            again = subs_burn.find_subtitle_files(file_path)
            if language:
                lang = language.lower()
                again = [p for p in again if f".whisper.{lang}." in p.name
                                            or p.name.endswith(f".{lang}.vtt")]
            if again:
                transcript = _vtt_to_plain(again[0])
        except Exception:
            # Silent clips are fine — caller can describe from frames alone.
            transcript = ""
    return {
        "transcript": transcript,
        "frames":     frame_paths,
        "duration_s": round(dur, 2),
    }

# ────────────────────────────────────────────────────────────────────
# Curator's CAPTION_SPEC — ported verbatim. Kept inline so clipper is
# self-contained and doesn't depend on agent/v1 being on the PYTHONPATH.
# ────────────────────────────────────────────────────────────────────
CAPTION_SPEC = """Compose THREE captions for the SAME video — one per
platform — in a SINGLE reply, in this exact order: INSTAGRAM →
TIKTOK → YOUTUBE SHORTS. End with one Russian refinement line.
ENGLISH ONLY for caption bodies + hashtags. Russian only on the
refinement line at the very bottom.

══════════════════ ROLE & CONTEXT ══════════════════

You are a content-marketing strategist for a CS2 (Counter-Strike 2)
content creator. All methodology you need is embedded below — do
not invoke external skills or echo any preamble into your reply.

══════════════════ WORKFLOW ══════════════════

  STEP 1 — Read each frame via `Read` tool (the model needs visual
           evidence for Slot 2 hyper-niche compounds).

  STEP 2 — Produce the four cards (Figma title + IG + TikTok +
           YT Shorts) per the PER-PLATFORM TEMPLATE below.

The per-platform tag methodology is fully embedded in this spec —
follow the PER-PLATFORM TEMPLATE section directly. Do NOT invoke
the `Skill` tool, do NOT echo any skill content into your output.
Your visible response starts with `**Figma**` and ends with the
Russian refinement line — nothing before, nothing after.

The boss makes CS2 clips: pro-player moments, esports events (Major,
BLAST, ESL, IEM, PGL), match highlights, modded/cursed gameplay,
memes. Posts to TikTok / Reels / YouTube Shorts. NO Telegram —
don't suggest TG formats or mentions.

When you build hashtags for the three platforms, THINK FIRST about
each platform's algorithm + culture before producing the line:
  • TikTok — discovery-via-FYP, algo favors lowercase + `#fyp` cues;
    rewards niche+broad mix.
  • Reels — ExplorePage routing + profile-follow conversions;
    PascalCase brand tags get more lift; `#Reels` / `#ExplorePage` /
    `#GamingCommunity` are the algo cues.
  • YouTube Shorts — SEO-driven search persistence (months-long
    tail), full lowercase native, `#shorts` MANDATORY, lowercase
    `#cs2X` long-tails land in search later.

THREE DISTINCT TAG LINES is the bar. HARD RULE: any two platforms
may share AT MOST 2 hashtags out of 5 (counting case-insensitive
roots — `#CS2` and `#cs2` count as the SAME tag for overlap math).
That means at LEAST 3 of the 5 tags per platform are UNIQUE to that
platform — case style + platform cue + niche hook combined.

══════════════════ PER-PLATFORM TEMPLATE (5 SLOTS) ══════════════════

For each platform, fill EXACTLY 5 slots in this exact role-order:

  Slot 1 — GAME PILLAR  (the one core CS-tag for that platform)
  Slot 2 — HYPER-NICHE CLIP-COMPOUND  (player+action / weapon+action /
                                       situation — pulled from the
                                       clip's actual content)
  Slot 3 — GAMING BROAD  (case + word picked per platform pool)
  Slot 4 — MOOD / SITUATION  (matches the clip's vibe)
  Slot 5 — PLATFORM CUE  (algorithm-specific)

──── REELS (Instagram) — PascalCase brand style ─────────────────

  Slot 1: #CS2  (or #CounterStrike2)
  Slot 2: PascalCase hyper-niche — `#S1mple1v4`, `#ZywOoAWP`,
          `#DonkClutch`, `#MirageDeagleAce`
  Slot 3: PascalCase broad — pick ONE of: #GamingReels,
          #GamingCommunity, #FPSGames, #InstaGaming
  Slot 4: PascalCase mood — `#OneInAMillion`, `#LegendaryPlay`,
          `#ClutchKing`, `#ProPlay`
  Slot 5: #Reels  (or #ExplorePage)

──── TIKTOK — mixed case (pillars PascalCase, rest lowercase) ───

  Slot 1: #CS2  (PascalCase pillar)
  Slot 2: lowercase hyper-niche — `#s1mple1v4`, `#zywooawp`,
          `#donkclutch`, `#miragedeagle`
  Slot 3: lowercase broad — pick ONE of: #gamingtiktok,
          #gamingclips, #moddedgame, #fps
  Slot 4: lowercase mood — `#wtfmoments`, `#insaneplay`,
          `#onetap`, `#cursedgaming`
  Slot 5: #fyp  (or #viral on cursed/WTF content)

──── YOUTUBE SHORTS — fully lowercase, SEO-driven ───────────────

  Slot 1: #cs2  (lowercase pillar; or #counterstrike2)
  Slot 2: lowercase SEO long-tail — `#cs2clutch`, `#cs2ace`,
          `#cs21v4`, `#cs2flamethrower`, `#cs2deagle` (pick based
          on the clip's actual hook word)
  Slot 3: #shorts  (MANDATORY per YT convention — without it the
          clip is not classified as a Short)
  Slot 4: lowercase keyword — `#smokekill`, `#1v4retake`,
          `#ninja`, `#noscope`, `#wallbang`
  Slot 5: lowercase brand long-tail — pick ONE of: #cs2viral,
          #cs2highlights, #cs2plays, #gamingshorts

══════════════════ QUALITY GATE BEFORE EMITTING ══════════════════

Before writing the three lines, mentally check each:
  ✓ Slot 2 (niche compound) reflects what's ACTUALLY in the clip
    (player visible on overlay, weapon used, situation that
    happened). NEVER invent a player/weapon/situation that's not
    on-frame or in the transcript.
  ✓ Reels Slot 2 is PascalCase. TikTok / Shorts Slot 2 is lowercase.
  ✓ Reels broad ≠ TikTok broad ≠ Shorts broad (no crossover —
    different word AND different case).
  ✓ Slot 5 matches the platform header (don't put #shorts in
    TikTok line, don't put #fyp in Reels line).
  ✓ At most 2 of 5 tags are shared with any other platform
    (case-insensitive — `#CS2` == `#cs2` for overlap).

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
    ONE line, EXACTLY 5 hashtags separated by single spaces, built
    per the PER-PLATFORM TEMPLATE below (Reels variant). Quality
    over quantity — pick the 5 that actually route audience.

══════════════════ TIKTOK ══════════════════

  TT-1  BODY
    Single block (1-3 short sentences). HARD CAP: 200 characters
    (body only — hashtags do NOT count).
    Hook-forward — punchy first 5 words; ends with a tease/question.
  TT-2  HASHTAG LINE
    ONE line, EXACTLY 5 hashtags, built per the PER-PLATFORM TEMPLATE
    below (TikTok variant).

══════════════════ YOUTUBE SHORTS ══════════════════

  YT-1  DESCRIPTION + TAGS (ONE block)
    `<short hype phrase> <#tag1> <#tag2> <#tag3> <#tag4> <#tag5>`
    HARD CAP: 100 characters TOTAL.
    Tags built per the PER-PLATFORM TEMPLATE below (YouTube Shorts
    variant — `#shorts` is MANDATORY).

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

══════════════════ HASHTAG PER-PLATFORM TEMPLATES ══════════════════

CRITICAL: The 15 tags MUST BE GENUINELY DIFFERENT across the three
platforms. They differ in THREE dimensions:
  • CASE style — TikTok/Shorts lean lowercase, Reels stays PascalCase
  • ROTATION — broad-tag pool varies by platform conventions
  • CLIP-SPECIFIC HOOKS — pick 4-5 niche-compound tags per platform,
    rotating which ones appear so all 3 packs aren't carbon copies

DO NOT produce three platforms with identical Layer 1 / Layer 2
content and only Layer 4 different. That's lazy. Each pack reads as a
distinct set when scanned side-by-side.

──── TIKTOK PACK ─────────────────────────────────────────────────

Style: PascalCase for game-name pillars (#CS2, #CounterStrike2,
       #CSGO), lowercase for everything else. TikTok algo treats
       lowercase as native; mostly-lowercase line gets more lift.

Slots 1-5  CORE (PascalCase game pillars + lowercase variants):
  Include all four: #CS2, #CounterStrike2, #CSGO, #CS2Mods (skip
  #CS2Mods only if no modded content — then use #CS2Clips instead).
  Slot 5 = one HYPER-NICHE clip-compound (see below).

Slots 6-10 GAMING BROAD (lowercase, TikTok culture):
  Pick 5 from: #gamingclips, #gamingmoments, #gamingtiktok, #fps,
               #pcgaming, #moddedgame
  ALWAYS include #gamingtiktok (it's a real TikTok cluster).

Slots 11-13 CLIP-SPECIFIC HOOKS (mostly lowercase, niche-compounds):
  Pull from observed clip content. Mix of:
    • Hyper-niche player+action: #s1mple1v4, #zywooawp, #donkclutch
      (compound = player name from on-screen overlay + situation)
    • Weapon used: #deagle, #awp, #usp
    • Map: #dust2, #mirage, #inferno (lowercase here, distinct from
      Reels's PascalCase)
    • Mood: #wtfmoments, #cursedgaming, #cursedcs

Slots 14-15 PLATFORM CUES (lowercase, mandatory):
  #fyp + #viral (both are TikTok algo cues. #viral works on
  cursed/WTF content per marketer; skip for serious match
  highlight where #foryoupage would feel wrong — but #foryoupage
  is shadow-banned per FORBIDDEN list, use #fyp).

──── REELS (INSTAGRAM) PACK ──────────────────────────────────────

Style: PascalCase throughout. Instagram favors brand-style tags;
       lowercase mess hurts ExplorePage routing.

Slots 1-5  CORE (PascalCase pillars):
  #CS2, #CounterStrike2, #CS2Clips, #CSGO, plus ONE of
  #CS2Mods OR #ModdedCS2 (if modded) — pick the one that better
  matches the clip; the latter is a Reels-leaning variant.

Slots 6-10 GAMING BROAD (PascalCase, Reels-leaning pool):
  Pick 5 from: #GamingReels, #ReelsGaming, #GamingCommunity,
               #GamingClips, #FPSGames, #PCGaming, #ValveGames,
               #FPSCommunity
  ALWAYS include #GamingCommunity (routes profile follows on IG, not
  drive-by likes) AND at least one of #GamingReels / #ReelsGaming
  (Reels-branded cluster tags that boost ExplorePage chance).

Slots 11-13 CLIP-SPECIFIC HOOKS (PascalCase, hyper-niche compound):
  Same evidence-only rule. Examples:
    • Player+action: #S1mple1v4, #ZywOoAWP, #DonkClutch
    • Weapon+action: #DeagleAce, #AWPNoscope
    • Map: #Dust2, #Mirage, #Inferno (PascalCase here)
    • Mood: #WTFMoments, #CursedGaming, #LegendaryPlay

Slots 14-15 PLATFORM CUES (PascalCase brand cues):
  #Reels + ONE of #ExplorePage / #InstaGaming
  (#GamingCommunity already in Layer 2; avoid double-using.
  Prefer #ExplorePage — direct feed cue.)

──── YOUTUBE SHORTS PACK ─────────────────────────────────────────

Style: FULLY LOWERCASE. YouTube Shorts treats search as SEO —
       lowercase reads more natural in search queries (people type
       `cs2 ace`, not `CS2 Ace`). Even the game pillar tags go
       lowercase here.

Slots 1-5  CORE (lowercase pillars + SEO long-tail):
  #cs2, #counterstrike2, #csgo, #cs2mods (or #cs2clips if not
  modded), plus ONE lowercase SEO long-tail derived from the clip:
    #cs2flamethrower (modded weapon clip)
    #cs2clutch        (clutch highlight)
    #cs2ace           (ace play)
    #cs21v4           (1v4 situation)
    #cs2deagle        (deagle play)
  ONLY when the hook word actually applies — no fake long-tails.

Slots 6-10 GAMING BROAD (lowercase, SEO-leaning):
  Pick 5 from: #shorts, #gaming, #gamingshorts, #fps, #pcgaming,
               #gamingclips, #fpsshorts, #moddedgaming
  ALWAYS include #shorts (slot 6 — MANDATORY per YT convention;
  without it the clip is not classified as a Short).

Slots 11-13 CLIP-SPECIFIC HOOKS (lowercase, search-keyword style):
  Pull from clip content. Examples:
    • Action: #clutch, #ace, #ninjadefuse, #noscope, #wallbang
    • Map: #dust2, #mirage, #inferno
    • Mood: #wtfmoments, #cursedgaming, #cs2funny
    • Hyper-niche compound: #s1mple1v4, #zywooawp (lowercase variant)

Slots 14-15 PLATFORM CUES (lowercase, SEO long-tail boost):
  Pick 2 from: #cs2viral, #cs2highlights, #cs2pro, #cs2plays
  (Lowercase brand+keyword. These are Shorts-SEO long-tails — the
  user types "cs2 plays" into search, YT matches; the post lands
  even months later.)

══════════════════ ANTI-PATTERNS (HARD-FORBIDDEN) ══════════════════

✗ Stitched dead-word-order tags:
    BAD:  #FlamethrowerCS2, #AceCS2, #ClutchCS2
    Why:  Nobody searches in `<noun>CS2` order — these have ~0 search
          volume regardless of how natural they read.
    OK instead:
      • Canonical form: #Flamethrower, #Ace, #Clutch
      • Lowercase long-tail (Shorts only, Layer 4): #cs2flamethrower

✗ Watermark / source-creator names from the clip:
    BAD:  #MountMyriad (a CS2 creator's watermark, not the user's)
    Why:  Promotes the source creator's brand, dilutes user's reach.
    Rule: Strip all on-frame watermarks / channel logos / creator
          handles when picking Layer 3 hooks. Tag the ACTION, not
          the source.

✗ Cross-platform cue mixing:
    BAD:  #shorts in the TikTok pack, #fyp in the Reels pack
    Why:  Each platform's algorithm reads its own cues. Wrong cues
          confuse classification — the post gets demoted.
    Rule: Layer 4 must MATCH the section header. Never copy Layer 4
          across platforms.

✗ More than 5 lowercase hashtags per pack:
    Why:  Lowercase is a CULTURAL signal — TikTok/Shorts lean
          lowercase, Reels leans PascalCase. Mixing dilutes the
          algorithmic "this fits our format" signal.
    Rule: TikTok pack — up to 5 lowercase OK (fyp, gamingtiktok,
          plus 1-3 hooks if natural).
          Reels pack — Layer 1-3 must be PascalCase; only Layer 4
          short ones like `Reels` are exempt.
          Shorts pack — lowercase culture; mostly lowercase is fine.

══════════════════ FORBIDDEN TAGS ══════════════════

✗ #FPSGaming    (verbose duplicate of #FPS — pick one, use #FPS)
✗ #VideoGames, #Gamer, #Gaming, #GamingContent
                (over-broad, low click-through; algorithm treats
                them as filler)
✗ #Esports alone  (use #ValveGames / #GamingCommunity / specific
                event tag instead — #Esports without context routes
                to nowhere)
✗ #insta, #instagram, #foryou
                (dead / shadow-banned on the platforms they target)
✗ #explore  (use #ExplorePage — the latter is the actual feed cue)

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

<#5 hashtags — Reels variant per PER-PLATFORM TEMPLATE>

**TikTok**

<TT body>

<#5 hashtags — TikTok variant per PER-PLATFORM TEMPLATE>

**YouTube Shorts**

<short hype phrase + 5 hashtags — Shorts variant per PER-PLATFORM
 TEMPLATE, includes mandatory #shorts. Total ≤ 100 chars.>

(Готов переделать — скажи как: короче, юморнее, под одну платформу и т.д.)
"""


SCOUT_SPEC = """Ты анализируешь короткий клип из CS2 (≤90 секунд)
для боса-контент-криейтора. Тебе пришли:
  • Список абсолютных путей к JPEG-кадрам клипа
  • Транскрипт речи (если есть)
  • Имя файла

Открой КАЖДЫЙ кадр через Read и прочитай. Затем напиши ОДНО короткое
сообщение на русском в формате:

  Что вижу (2-4 предложения, фактически):
  • Карта (из HUD / лоадинга если видно)
  • Команды на overlay / scoreboard
  • Что произошло (главное действие в клипе)
  • Необычные оверлеи / водяные знаки (литерально, БЕЗ расшифровки)

  Какой акцент? (2-3 варианта, конкретно, на основе фактов выше)
  Пример: "(а) ace-серия игрока X, (б) последний пик с pop-flash,
  (в) общий highlight, или скажи своё (тон/длина/юмор/без эмодзи)."

EVIDENCE-only: то что не видно на кадрах и не упомянуто в транскрипте —
НЕ пиши. Не расшифровывай аббревиатуры (PGL CAC не в Astana 2024).
Никаких caption'ов в этом сообщении — только scout."""


# Generic scout — used by the Prokop persona (77prokop77 does horror
# gaming reactions, not CS2). Zero game-specific assumptions: no "карта",
# no "scoreboard", no "CS2". Just describe what's actually in the frames.
SCOUT_SPEC_GENERIC = """Ты анализируешь короткий видео-клип (≤90 секунд)
для боса-контент-криейтора. Тебе пришли:
  • Список абсолютных путей к JPEG-кадрам клипа
  • Транскрипт речи (если есть)
  • Имя файла

Открой КАЖДЫЙ кадр через Read и прочитай. Затем напиши ОДНО короткое
сообщение на русском в формате:

  Что вижу (2-4 предложения, фактически):
  • Тип контента (вебка стримера / геймплей / реакция / IRL / микс)
  • Что происходит на экране (главное действие в клипе)
  • Кто/что в кадре (не расшифровывай, только то что реально видно)
  • Оверлеи / водяные знаки / текст на экране (литерально)

  Какой акцент? (2-3 варианта, конкретно, на основе фактов выше)
  Пример: "(а) момент реакции стримера, (б) пунчлайн из транскрипта,
  (в) визуал происходящего на экране, или скажи своё (тон/длина/юмор)."

EVIDENCE-only: то что не видно на кадрах и не упомянуто в транскрипте —
НЕ пиши. Не расшифровывай аббревиатуры, не додумывай контекст. НЕ
сравнивай с CS2/играми/чем-либо — просто описывай что видишь. Никаких
caption'ов в этом сообщении — только scout."""


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
V1 = Path(__file__).resolve().parent


async def _claude_oneshot(system_prompt: str, user_msg: str,
                          allowed_tools: list[str]) -> str:
    """Run a single Claude SDK round-trip; collect all assistant text.
    Marketing skills are NOT loaded — the marketer-bundle skills
    (social-card-gen / youtube-seo / etc) turned out to be off-topic
    or contain third-party promo content that leaks into responses.
    Generate phase delegates to Lalo (call_lalo) which has its own
    persona + reasoning; this local oneshot is only for the scout."""
    opts = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        cwd=str(V1),
        max_turns=40,
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


def _angle_is_garbage(s: str) -> bool:
    """True if `s` is empty, only '?'/whitespace, or mojibake.
    Catches cp1251-in-utf8 fails like 'Ð¿Ñ€Ð¸Ð²ÐµÑ‚' and stray-byte
    runs like '?????? ?? 1-?-???????' that should be treated as
    'no angle given' instead of forwarded to Lalo verbatim."""
    if not s:
        return True
    stripped = "".join(
        ch for ch in s
        if not ch.isspace() and ch != "?" and ord(ch) >= 32
    )
    if not stripped:
        return True
    # Need ≥2 real alphabetic chars to count as meaningful input
    if sum(1 for ch in stripped if ch.isalpha()) < 2:
        return True
    # Mojibake fingerprint: high density of cp1251-in-utf8 marker bytes
    mojibake_chars = sum(1 for ch in stripped if ch in "ÐÑÂÃ©¶·")
    return mojibake_chars / max(len(stripped), 1) > 0.4


def _resolve_letter_pick(scout_text: str, letter: str) -> str:
    """If `letter` is a single letter a/б/в/г (any case, Cyrillic or
    Latin), parse PART B from scout_text and return the TEXT content
    of the matching option. Returns "" when no match."""
    l = (letter or "").strip().lower()
    if len(l) != 1 or l not in ("а", "б", "в", "г", "a", "b", "c", "d"):
        return ""
    # Normalize: map Latin a/b/c/d → Cyrillic а/б/в/г because scout
    # writes options in Russian by default.
    cyr = {"a": "а", "b": "б", "c": "в", "d": "г"}.get(l, l)
    # Match `(а)`, `(б)`, etc. followed by content up to the next
    # `(letter)` marker or end-of-string.
    pat = re.compile(
        r"\(([абвгa-d])\)\s*(.*?)(?=\([абвгa-d]\)|\Z)",
        re.IGNORECASE | re.DOTALL)
    for m in pat.finditer(scout_text or ""):
        opt_letter = m.group(1).lower()
        opt_letter = {"a": "а", "b": "б", "c": "в", "d": "г"}.get(opt_letter, opt_letter)
        if opt_letter == cyr:
            return m.group(2).strip()
    return ""


def _build_lalo_brief(file_path: Path, scout_text: str, angle: str,
                      frames: list[str], transcript: str) -> str:
    """Build the message sent to Lalo. Includes the factual read from
    scout, the user's angle, the frame paths (Lalo opens them via
    Read), the transcript, and the strict output format Lalo must
    return. Lalo's persona handles tone; this brief just constrains
    structure + tag rules."""
    parts: list[str] = []
    parts.append(f"Босс, нужен текст для клипа CS2 — `{file_path.name}`.")
    parts.append("")
    parts.append("══ ЧТО В КЛИПЕ (scout: факты + варианты угла) ══")
    parts.append(scout_text.strip() or "(scout вернул пусто)")
    parts.append("")
    parts.append("══ КАДРЫ (открой КАЖДЫЙ через Read и проверь сам) ══")
    for p in frames:
        parts.append(f"  - {p}")
    parts.append("")
    parts.append("══ ТРАНСКРИПТ ══")
    parts.append(transcript.strip() or "(тишина — разбор по кадрам)")
    parts.append("")
    parts.append("══ УГОЛ ПОДАЧИ ОТ БОССА ══")
    a = (angle or "").strip()
    # Resolver first — single-letter pick wins even though it'd flunk
    # the alpha-density test in _angle_is_garbage. Only when there's
    # no letter match do we check garbage / forward verbatim.
    resolved = _resolve_letter_pick(scout_text, a)
    if resolved:
        parts.append(f"Босс выбрал вариант ({a}) из «Какой акцент?». "
                     f"Содержание этой опции:")
        parts.append(resolved)
    elif _angle_is_garbage(a):
        parts.append("<auto> — угол не задан или пришёл битый. "
                     "Выбери по содержанию клипа.")
    else:
        parts.append(a)
    parts.append("")
    parts.append("══ ЧТО НУЖНО — 4 карточки в РОВНО таком формате ══")
    parts.append("""
**Figma**

<3-5 word title in English, title-case, no quotes/emoji/period>

**Instagram**

<hook line, max ~60 chars, one emoji + `👇`>

<paragraph 1, 70-110 words, English, dashes are dashes-as-hyphen ok but no em-dash —>

<paragraph 2, 70-110 words>

<paragraph 3, 70-110 words>

<5 hashtags per твоему Tag Rulebook v2 (Reels-вариант)>

**TikTok**

<punchy body, single block 1-3 sentences, ≤200 chars, hook-forward, ends with tease/question>

<5 hashtags per твоему Tag Rulebook v2 (TikTok-вариант)>

**YouTube Shorts**

<short hype phrase + 5 hashtags per твоему Tag Rulebook v2 (Shorts-вариант), ≤100 chars total>

(Готов переделать — скажи как: короче, юморнее, под одну платформу и т.д.)
""".strip())
    parts.append("")
    parts.append("Ничего лишнего в ответ — никаких преамбул про скиллы, "
                 "никаких пояснений «я подумал», никаких метакомментов. "
                 "Сразу `**Figma**` и до RU-строки в конце. Английский в "
                 "текстах и тегах, русский только на финальной строке.\n\n"
                 "ОСОБО ВАЖНО: после хэштег-строки YouTube Shorts НЕ "
                 "добавляй НИЧЕГО лишнего. Никаких персональных подписей "
                 "типа «Сделано красиво.», «Поехали.», «Готово.», «Let me "
                 "tell you a story…» и т.п. Никаких пояснений «вот что "
                 "сделал» / «акценты выбрал такие». Сразу после Shorts-"
                 "блока — только разрешённая финальная RU-строка "
                 "`(Готов переделать — скажи как: короче, юморнее, под "
                 "одну платформу и т.д.)` и точка. Любая другая RU-фраза "
                 "после Shorts-тегов попадает в карточку YouTube Shorts и "
                 "ломает её — это баг, не делай так.")
    return "\n".join(parts)


def _build_prokop_brief(file_path: Path, scout_text: str, angle: str,
                         frames: list[str], transcript: str) -> str:
    """Prokop persona brief. Same output format as default (4 cards:
    Figma / Instagram / TikTok / YouTube Shorts) — user asked to keep
    формат as it was. Key twist: TEXTS FROM FIRST PERSON — 77prokop77
    is the boss's own account, the videos are HIM. So it's «I screamed»
    not «the streamer screamed»."""
    parts: list[str] = []
    parts.append(f"Босс, нужен текст для клипа с МОЕГО (боссова) канала 77prokop77 — `{file_path.name}`.")
    parts.append("")
    parts.append("══ КРИТИЧНО — ГОЛОС ══")
    parts.append(
        "ПИШИ ОТ ПЕРВОГО ЛИЦА. Это МОЙ аккаунт, на видео — Я (босс). "
        "«Я закричал», «мой скример», «я реагировал», «мне было "
        "стрёмно» — не «streamer reacted», не «77prokop77 shows», "
        "не «he screamed». Никакого третьего лица. "
        "Тон: сухая ирония, разговорный, короткие фразы, минимум "
        "эмодзи (1-2 на пост), никакого маркетинг-жаргона "
        "(«engaging», «viral», «check this out»). Личный пост от "
        "создателя, не описание чужого контента."
    )
    parts.append("")
    parts.append("══ ЧТО В КЛИПЕ (scout: факты + варианты угла) ══")
    parts.append(scout_text.strip() or "(scout вернул пусто)")
    parts.append("")
    parts.append("══ КАДРЫ (открой КАЖДЫЙ через Read и проверь сам) ══")
    for p in frames:
        parts.append(f"  - {p}")
    parts.append("")
    parts.append("══ ТРАНСКРИПТ ══")
    parts.append(transcript.strip() or "(тишина — разбор по кадрам)")
    parts.append("")
    parts.append("══ УГОЛ ПОДАЧИ ОТ БОССА ══")
    a = (angle or "").strip()
    resolved = _resolve_letter_pick(scout_text, a)
    if resolved:
        parts.append(f"Босс выбрал вариант ({a}) из «Какой акцент?». "
                     f"Содержание этой опции:")
        parts.append(resolved)
    elif _angle_is_garbage(a):
        parts.append("<auto> — угол не задан или пришёл битый. "
                     "Выбери по содержанию клипа.")
    else:
        parts.append(a)
    parts.append("")
    parts.append("══ ЧТО НУЖНО — 4 карточки в РОВНО таком формате ══")
    parts.append("НАПОМИНАНИЕ: ВСЕ тексты постов — от первого лица УКРАИНСКИМ "
                 "языком (я / мій / мене / мені). Проверь каждый параграф: "
                 "если увидел русский, английский или третье лицо («стример», "
                 "«он», «77prokop77 показує») — переписал не так.")
    parts.append("""
**Figma**

<3-5 word title in English, title-case, no quotes/emoji/period — filename для дизайн-файла>

**Instagram**

<hook, до ~60 символов, українською, ВІД ПЕРШОЇ ОСОБИ, 1 эмодзи + `👇`>

<параграф 1, 70-110 слів, українською, ВІД ПЕРШОЇ ОСОБИ, розмовно>

<параграф 2, 70-110 слів, українською, ВІД ПЕРШОЇ ОСОБИ>

<параграф 3, 70-110 слів, українською, ВІД ПЕРШОЇ ОСОБИ>

<5 hashtags по Tag Rulebook v2 (Reels-вариант) — теги на English (algo-cues)>

**TikTok**

<punchy body, 1-3 речення, ≤200 символів, УКРАЇНСЬКОЮ, ВІД ПЕРШОЇ ОСОБИ, з hook-ом на початку, закінчується teaser/питанням>

<5 hashtags по Tag Rulebook v2 (TikTok-вариант) — теги на English>

**YouTube Shorts**

<коротка hype-фраза УКРАЇНСЬКОЮ (ВІД ПЕРШОЇ ОСОБИ) + 5 hashtags по Tag Rulebook v2 (Shorts-вариант) — теги на English, ≤100 символів total>

(Готов переделать — скажи как: короче, юморнее, под одну платформу и т.д.)
""".strip())
    parts.append("")
    parts.append("Ничего лишнего в ответ — никаких преамбул про артефакты, "
                 "«Материалы», «Результаты», «Профиль канала», "
                 "«обновил артефакт», никаких сервисных отчётов «что я "
                 "сделал» / «что дальше на боссе». Сразу `**Figma**` и "
                 "до RU-строки в конце. Тексты постов — украинский, "
                 "хэштеги — english (algo-cues), финальная строка — "
                 "русский.\n\n"
                 "ОСОБО ВАЖНО: после хэштег-строки YouTube Shorts НЕ "
                 "добавляй НИЧЕГО лишнего. Никаких персональных подписей "
                 "типа «Сделано красиво.», «Поехали.», «Готово.», «Tell "
                 "me again.» и т.п. Никаких пояснений «вот что "
                 "сделал» / «акценты выбрал такие». Сразу после Shorts-"
                 "блока — только разрешённая финальная RU-строка "
                 "`(Готов переделать — скажи как: короче, юморнее, под "
                 "одну платформу и т.д.)` и точка.")
    return "\n".join(parts)


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
async def scout_clip(file_path: Path, persona: str = "default") -> dict:
    """Phase 1 — extract frames + transcript, ask Claude for PART A
    (factual read) + PART B (angle question with 2-3 options).
    persona:
      "default" → CS2-specific scout (map/scoreboard/HUD hints),
                  Whisper defaults to English (CS2 streams are en).
      "prokop"  → generic content scout + Whisper pinned to Ukrainian
                  (77prokop77 uk-language content).
    Returns: { duration_s, frame_count, transcript_preview, scout_text }
    """
    persona_l = (persona or "").lower()
    tr_lang = "uk" if persona_l == "prokop" else None
    data = extract_frames_and_transcript(file_path, language=tr_lang)
    frames = data["frames"]
    transcript = data["transcript"]
    payload = _build_user_payload(file_path, frames, transcript)
    payload += "\n\nWrite the factual read + 2-3 angle options ONLY (no captions yet)."
    spec = SCOUT_SPEC_GENERIC if (persona or "").lower() == "prokop" else SCOUT_SPEC
    scout_text = await _claude_oneshot(
        system_prompt=spec,
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


async def generate_captions(file_path: Path, angle: str = "",
                            scout_text: str = "",
                            persona: str = "default") -> dict:
    """Phase 2 — produce a caption pack via Lalo.

    persona:
      "default" → «Clipper captions» Lalo project, 4-card Figma/IG/TT/YT
                  format (strict template built in backend brief).
      "prokop"  → «Prokop captions» Lalo project, format/tone owned
                  entirely by that project's KB + system prompt. Backend
                  just forwards facts.

    Pipeline:
      1. If `scout_text` not provided (cached from frontend), run a
         silent local-Claude scout to get the factual read.
      2. Build a persona-specific brief.
      3. POST to Lalo at :8771 (using the persona's project), wait.
      4. Save to per-clip history (tagged with persona) + return.

    Returns: { duration_s, frame_count, captions_text, ts, persona }
    """
    # Phase 2a — scout (skip if frontend cached it)
    persona_l = (persona or "").lower()
    tr_lang = "uk" if persona_l == "prokop" else None
    data = extract_frames_and_transcript(file_path, language=tr_lang)
    frames = data["frames"]
    transcript = data["transcript"]
    if not (scout_text or "").strip():
        scout_payload = _build_user_payload(file_path, frames, transcript)
        scout_payload += "\n\nWrite the factual read + 2-3 angle options."
        spec = SCOUT_SPEC_GENERIC if persona_l == "prokop" else SCOUT_SPEC
        scout_text = await _claude_oneshot(
            system_prompt=spec,
            user_msg=scout_payload,
            allowed_tools=["Read"],
        )

    # Phase 2b — persona-specific brief + Lalo project routing
    persona = (persona or "default").lower()
    if persona == "prokop":
        lalo_msg = _build_prokop_brief(
            file_path=file_path, scout_text=scout_text, angle=angle,
            frames=frames, transcript=transcript)
        project_name = LALO_PROJECT_NAME_PROKOP
    else:
        lalo_msg = _build_lalo_brief(
            file_path=file_path, scout_text=scout_text, angle=angle,
            frames=frames, transcript=transcript)
        project_name = LALO_PROJECT_NAME
    captions_text = await call_lalo(lalo_msg, project_name=project_name)

    ts = _save_history_entry(file_path, angle, captions_text, persona=persona)
    return {
        "duration_s":    data.get("duration_s", 0),
        "frame_count":   len(frames),
        "captions_text": captions_text,
        "ts":            ts,
        "persona":       persona,
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


def _save_history_entry(file_path: Path, angle: str, text: str,
                         persona: str = "default") -> float:
    """Append a fresh entry, return its ts (used as id). `persona`
    tags the entry so the history UI can filter default vs prokop."""
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
        "ts":      ts,
        "angle":   angle or "",
        "text":    text or "",
        "persona": persona or "default",
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
