"""Parse AMI transcript files into speaker-turn records, preserving exact text."""
from pathlib import Path
import re

TURN_RE = re.compile(r"(?m)^(?P<time>\d{2}:\d{2}:\d{2})\s+(?P<speaker>Speaker [A-D])\s*$")

def parse_transcript(path: Path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    meeting = re.search(r"(?m)^Meeting:\s*(\S+)", raw)
    team = re.search(r"(?m)^Team:\s*(\S+)", raw)
    if not meeting or not team:
        raise ValueError(f"Missing Meeting/Team header in {path.name}")
    matches = list(TURN_RE.finditer(raw))
    records=[]
    for i,m in enumerate(matches):
        start=m.end()
        end=matches[i+1].start() if i+1<len(matches) else len(raw)
        text=raw[start:end].strip()
        if not text:
            continue
        records.append({
            "meeting_id": meeting.group(1), "team": team.group(1),
            "speaker": m.group("speaker"), "timestamp": m.group("time"),
            "turn_id": f"{meeting.group(1)}_{i+1}", "text": text,
            "source_file": path.name
        })
    return records

def split_long_turn(turn, max_chars=1100):
    """Split long speaker turns into exact text fragments; fragments remain quotable."""
    text=turn["text"]
    if len(text)<=max_chars:
        return [dict(turn, chunk_id=turn["turn_id"]+"_0")]
    # Keep every character and do not rewrite the source. Prefer whitespace boundaries.
    chunks=[]; pos=0; part=0
    while pos<len(text):
        end=min(pos+max_chars, len(text))
        if end<len(text):
            boundary=text.rfind(" ", pos, end)
            if boundary>pos+max_chars//2:
                end=boundary
        fragment=text[pos:end]
        if fragment.strip():
            chunks.append(dict(turn, text=fragment, chunk_id=f"{turn['turn_id']}_{part}"))
            part+=1
        pos=end
        while pos<len(text) and text[pos].isspace():
            pos+=1
    return chunks
