
"""Multi-step AMI question answering with exact, Python-generated citations."""
import json
import re
from collections import defaultdict
from pathlib import Path

from .config import QUESTIONS_FILE, DATA_DIR
from .store import TranscriptStore
from .gemini import Gemini
from .transcripts import parse_transcript, split_long_turn


def normalize(value):
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def words(value):
    return re.findall(r"\b\w+\b", str(value))


STOP_WORDS = {
    "what", "which", "when", "where", "who", "how", "did", "does",
    "they", "their", "team", "from", "with", "into", "about", "that",
    "this", "then", "than", "have", "were", "was", "are", "the",
    "and", "for", "one", "more", "over", "first", "later", "final",
    "will", "would", "could", "should", "between", "there", "them",
    "some", "also", "been", "being", "because",
}


class MeetingAgent:
    def __init__(self):
        self.store = TranscriptStore()
        self.llm = Gemini()
        self.source_turns = {}
        self.team_meetings = defaultdict(set)

        for path in Path(DATA_DIR).glob("*.txt"):
            for turn in parse_transcript(path):
                self.source_turns.setdefault(
                    turn["meeting_id"], []
                ).append(turn)
                self.team_meetings[turn["team"]].add(
                    turn["meeting_id"]
                )

    def _question_terms(self, question):
        return {
            term
            for term in re.findall(r"[a-zA-Z0-9]+", question.casefold())
            if len(term) > 2 and term not in STOP_WORDS
        }

 
    def retrieve(self, team, question, extra=False):
        """Retrieve evidence for separate parts of a question."""
        meeting_ids = sorted(self.team_meetings.get(team, set()))

        if not meeting_ids:
            return [], []

        # Split multi-part questions so each part can retrieve
        # its own relevant meetings.
        aspects = [
            part.strip(" ,?!.")
            for part in re.split(r"\band\b", question, flags=re.IGNORECASE)
            if len(self._question_terms(part)) >= 2
        ]

        if not aspects:
            aspects = [question]

        # Avoid excessively large retrieval prompts.
        aspects = aspects[:4]

        meeting_data = {}

        for meeting_id in meeting_ids:
            combined = {}
            aspect_scores = []

            for aspect in aspects:
                hits = self.store.search(
                    aspect,
                    team=team,
                    meeting_ids=[meeting_id],
                    n_results=10 if extra else 8,
                )

                best_distance = min(
                    (item.get("distance", 2.0) for item in hits),
                    default=2.0,
                )

                terms = self._question_terms(aspect)
                lexical = []

                for turn in self.source_turns.get(meeting_id, []):
                    turn_terms = set(
                        re.findall(
                            r"[a-zA-Z0-9]+",
                            turn["text"].casefold()
                        )
                    )

                    overlap = (
                        len(terms & turn_terms) / max(1, len(terms))
                    )

                    if overlap > 0:
                        for fragment in split_long_turn(turn):
                            lexical.append({
                                **fragment,
                                "distance": 1.0 - overlap,
                            })

                lexical.sort(
                    key=lambda item: item.get("distance", 2.0)
                )

                for item in hits:
                    key = item.get(
                        "chunk_id",
                        item.get("turn_id", item.get("text", "")),
                    )
                    combined[key] = item

                for item in lexical[:5]:
                    key = item.get(
                        "chunk_id",
                        item.get("turn_id", item.get("text", "")),
                    )
                    combined.setdefault(key, item)

                best_lexical = max(
                    (1.0 - item["distance"] for item in lexical),
                    default=0.0,
                )

                aspect_scores.append(
                    best_distance - (0.20 * best_lexical)
                )

            meeting_data[meeting_id] = {
                "score": min(aspect_scores),
                "items": list(combined.values()),
                "aspect_scores": aspect_scores,
            }

        # Select the best meeting separately for each question aspect.
        selected = []

        for aspect_index in range(len(aspects)):
            ranked = sorted(
                meeting_ids,
                key=lambda mid: (
                    meeting_data[mid]["aspect_scores"][aspect_index]
                ),
            )

            best_meeting = ranked[0]

            if best_meeting not in selected:
                selected.append(best_meeting)

        # If different aspects selected the same meeting, add another
        # strong candidate to provide evidence from another meeting.
        overall_ranked = sorted(
            meeting_ids,
            key=lambda mid: meeting_data[mid]["score"],
        )

        for meeting_id in overall_ranked:
            if len(selected) >= 4:
                break

            if meeting_id not in selected:
                selected.append(meeting_id)

        evidence = []

        for meeting_id in selected:
            items = meeting_data[meeting_id]["items"]

            items.sort(
                key=lambda item: item.get("distance", 2.0)
            )

            for item in items[:8 if extra else 6]:
                evidence.append(item)

        return evidence, selected


    def _make_exact_citations(self, evidence_ids, evidence_map, team):
        """
        Convert Gemini-selected evidence IDs into exact transcript quotes.
        Gemini never writes the final citation quote.
        """
        citations = []
        used_meetings = set()

        for evidence_id in evidence_ids:
            item = evidence_map.get(evidence_id)

            if not item:
                continue

            meeting_id = str(item.get("meeting_id", ""))

            if not meeting_id.startswith(team):
                continue

            if meeting_id in used_meetings:
                continue

            fragment = str(item.get("text", "")).strip()

            if not fragment:
                continue

            original_turns = self.source_turns.get(meeting_id, [])

            # Prefer the retrieved text if it exactly occurs in a source turn.
            exact_quote = None

            for turn in original_turns:
                if normalize(fragment) in normalize(turn["text"]):
                    exact_quote = fragment
                    break

            # If the retrieved text is not an exact source substring,
            # use the original full speaker turn when speaker/timestamp match.
            if exact_quote is None:
                for turn in original_turns:
                    same_speaker = (
                        str(turn.get("speaker", ""))
                        == str(item.get("speaker", ""))
                    )
                    same_timestamp = (
                        str(turn.get("timestamp", ""))
                        == str(item.get("timestamp", ""))
                    )

                    if same_speaker and same_timestamp:
                        exact_quote = str(turn["text"]).strip()
                        break

            if not exact_quote:
                continue

            if len(words(exact_quote)) < 6:
                continue

            # Final deterministic source check.
            if not any(
                normalize(exact_quote) in normalize(turn["text"])
                for turn in original_turns
            ):
                continue

            citations.append({
                "meeting_id": meeting_id,
                "quote": exact_quote,
            })

            used_meetings.add(meeting_id)

            if len(citations) == 3:
                break

        return citations

    def verify(self, answer_obj, team):
        issues = []
        citations = answer_obj.get("citations", [])

        if not isinstance(citations, list) or not (
            1 <= len(citations) <= 3
        ):
            issues.append("Need 1 to 3 citations")
            citations = citations if isinstance(citations, list) else []

        meeting_ids = set()

        for citation in citations:
            if not isinstance(citation, dict):
                issues.append("Citation must be an object")
                continue

            meeting_id = str(citation.get("meeting_id", ""))
            quote = str(citation.get("quote", ""))

            if meeting_id not in self.source_turns:
                issues.append(f"Unknown meeting {meeting_id}")
                continue

            if not meeting_id.startswith(team):
                issues.append(
                    f"Citation {meeting_id} is not from requested team"
                )

            if len(words(quote)) < 6:
                issues.append(f"Quote too short in {meeting_id}")

            if not any(
                normalize(quote) in normalize(turn["text"])
                for turn in self.source_turns[meeting_id]
            ):
                issues.append(
                    f"Quote is not an exact substring of one speaker turn in {meeting_id}"
                )

            meeting_ids.add(meeting_id)

        if len(meeting_ids) < 2:
            issues.append(
                "Evidence must cover at least two distinct meetings of the team"
            )

        answer = str(answer_obj.get("answer", "")).strip()

        sentence_count = len([
            part
            for part in re.split(r"(?<=[.!?])\s+", answer)
            if part.strip()
        ])

        if not answer or sentence_count > 3:
            issues.append("Answer must be 1 to 3 sentences")

        return issues

    def answer_one(self, question_record):
        team = question_record["team"]
        question = question_record["question"]

        evidence, selected_meetings = self.retrieve(team, question)

        if len(selected_meetings) < 2:
            return {
                "question_id": question_record["question_id"],
                "answer": "Insufficient transcript meetings were found; manual review required.",
                "citations": [],
                "verification_issues": [
                    "Fewer than two source meetings available"
                ],
            }

        # Assign IDs to evidence. Gemini selects IDs, not citation quotes.
        evidence_map = {}
        grouped = []
        counter = 1

        for meeting_id in selected_meetings:
            meeting_items = [
                item for item in evidence
                if item.get("meeting_id") == meeting_id
            ]

            group = {
                "meeting_id": meeting_id,
                "evidence": [],
            }

            for item in meeting_items:
                evidence_id = f"E{counter:03d}"
                counter += 1

                evidence_map[evidence_id] = item

                group["evidence"].append({
                    "evidence_id": evidence_id,
                    "speaker": item.get("speaker", ""),
                    "timestamp": item.get("timestamp", ""),
                    "text": item.get("text", ""),
                })

            grouped.append(group)

        last_obj = None
        last_issues = []

        for attempt in range(2):
            retry_note = ""

            if attempt:
                retry_note = (
                    "Your previous response failed these checks: "
                    + "; ".join(last_issues)
                    + ". Select different evidence IDs if necessary."
                )

            prompt = f"""
You answer questions using ONLY the supplied AMI transcript evidence.

STRICT REQUIREMENTS:
- Answer in 1 to 3 sentences.
- Use only facts supported by the supplied evidence.
- Do not invent facts or guess.
- Select evidence from at least TWO DIFFERENT meeting IDs.
- Return 2 or 3 evidence IDs from different meetings.
- Do NOT write citations or quotes yourself.
- Return only evidence IDs that directly support your answer.
- If evidence is insufficient, clearly say what is unclear.

Return valid JSON only:
{{"answer":"your answer","evidence_ids":["E001","E002"]}}

Question ID: {question_record['question_id']}
Team: {team}
Question: {question}

Evidence:
{json.dumps(grouped, ensure_ascii=False)}

{retry_note}
"""

            try:
                raw_obj = self.llm.ask_json(prompt)

                if not isinstance(raw_obj, dict):
                    raise ValueError("Gemini did not return a JSON object")

                evidence_ids = raw_obj.get("evidence_ids", [])

                if not isinstance(evidence_ids, list):
                    evidence_ids = []

                # Python constructs citations from original transcript text.
                citations = self._make_exact_citations(
                    evidence_ids,
                    evidence_map,
                    team,
                )

                last_obj = {
                    "answer": str(raw_obj.get("answer", "")).strip(),
                    "citations": citations,
                }

            except Exception as exc:
                last_obj = {
                    "answer": f"Unable to generate an answer because Gemini returned an error: {exc}",
                    "citations": [],
                }
                last_issues = [
                    "Gemini API request failed; rerun this question later"
                ]
                break

            last_issues = self.verify(last_obj, team)

            if not last_issues:
                return {
                    "question_id": question_record["question_id"],
                    "answer": last_obj["answer"],
                    "citations": last_obj["citations"],
                }

        return {
            "question_id": question_record["question_id"],
            "answer": (last_obj or {}).get(
                "answer", "Evidence insufficient; manual review required."
            ),
            "citations": (last_obj or {}).get("citations", []),
            "verification_issues": last_issues,
        }

    def run_all(self, out_path="answers.json", fresh=False):
        data = json.loads(
            Path(QUESTIONS_FILE).read_text(encoding="utf-8")
        )
        questions = data["questions"]
        output = Path(out_path)

        existing = {}

        if output.exists() and not fresh:
            try:
                old_data = json.loads(
                    output.read_text(encoding="utf-8")
                )

                for item in old_data.get("answers", []):
                    team = next(
                        (
                            q["team"] for q in questions
                            if q["question_id"] == item.get("question_id")
                        ),
                        "",
                    )

                    if not self.verify(item, team):
                        existing[item["question_id"]] = item

            except Exception:
                existing = {}

        results = []

        for index, question_record in enumerate(questions, 1):
            question_id = question_record["question_id"]

            print(
                f"\n[{index}/{len(questions)}] "
                f"{question_id} ({question_record['team']})"
            )

            if question_id in existing:
                result = existing[question_id]
                print("Already verified; keeping previous answer.")
            else:
                result = self.answer_one(question_record)
                print(
                    "Verified"
                    if "verification_issues" not in result
                    else "REVIEW REQUIRED"
                )

            results.append(result)

            output.write_text(
                json.dumps(
                    {"answers": results + [
                        item for item in existing.values()
                        if item.get("question_id") not in {
                            r.get("question_id") for r in results
                        }
                    ]},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        by_id = {item["question_id"]: item for item in results}

        for item in existing.values():
            by_id.setdefault(item["question_id"], item)

        ordered = [
            by_id[q["question_id"]]
            for q in questions
            if q["question_id"] in by_id
        ]

        output.write_text(
            json.dumps(
                {"answers": ordered},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(f"\nSaved {out_path}. Manually inspect every answer.")
        print(
            "Run: python -m src.check_answers "
            + str(out_path)
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Answer Catalist transcript questions."
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Regenerate all answers.",
    )

    args = parser.parse_args()
    MeetingAgent().run_all(fresh=args.fresh)


if __name__ == "__main__":
    main()
