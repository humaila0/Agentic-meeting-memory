
"""Independent citation checker for answer files."""
import json
import re
import sys
from pathlib import Path
from .config import DATA_DIR
from .transcripts import parse_transcript


def norm(s):
    return re.sub(r"\s+", " ", s).strip().casefold()


def main():
    answer_file = Path(
        sys.argv[1] if len(sys.argv) > 1 else "answers.json"
    )

    turns = {}
    for p in Path(DATA_DIR).glob("*.txt"):
        for t in parse_transcript(p):
            turns.setdefault(t["meeting_id"], []).append(t["text"])

    data = json.loads(answer_file.read_text(encoding="utf-8"))
    errors = []

    for a in data.get("answers", []):
        mids = set()
        qid = a.get("question_id")

        for c in a.get("citations", []):
            mid = c.get("meeting_id", "")
            quote = c.get("quote", "")
            mids.add(mid)

            valid = (
                len(re.findall(r"\b\w+\b", quote)) >= 6
                and any(
                    norm(quote) in norm(t)
                    for t in turns.get(mid, [])
                )
            )

            if not valid:
                errors.append({
                    "question_id": qid,
                    "meeting_id": mid,
                    "problem": "quote not exact or shorter than six words"
                })

        if len(mids) < 2:
            errors.append({
                "question_id": qid,
                "problem": "fewer than two distinct meetings cited"
            })

    print(f"File checked: {answer_file}")
    print(f"Answers checked: {len(data.get('answers', []))}")
    print(f"Citation issues: {len(errors)}")

    for e in errors:
        print(e)

    audit_file = answer_file.with_name(
        answer_file.stem + "_citation_audit.json"
    )
    audit_file.write_text(
        json.dumps({"issues": errors}, indent=2),
        encoding="utf-8"
    )
    print(f"Audit saved: {audit_file}")


if __name__ == "__main__":
    main()
