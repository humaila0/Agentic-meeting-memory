'Create TS3012 pre-final-meeting brief using meetings a, b and c only.'
import json
from pathlib import Path
from .store import TranscriptStore
from .gemini import Gemini
from .config import DATA_DIR
from .transcripts import parse_transcript
from .agent import normalize, words

def main():
    team="TS3012"; allowed=["TS3012a","TS3012b","TS3012c"]
    store=TranscriptStore(); llm=Gemini(); evidence={}
    topics=["design decisions materials components colours and rationale", "decisions that changed between meetings and why", "open tasks commitments owners and follow up", "unresolved disagreements risks and questions", "costing budget constraints and final design tradeoffs"]
    for topic in topics:
        for x in store.search(topic,team=team,meeting_ids=allowed,n_results=15): evidence[x["chunk_id"]]=x
    # Include selected source turns, not meeting d: the brief is for the team before its final meeting.
    context=[{"meeting_id":x["meeting_id"],"speaker":x["speaker"],"timestamp":x["timestamp"],"text":x["text"]} for x in sorted(evidence.values(),key=lambda x:x["distance"])[:60]]
    prompt=f'''Write a concise, useful pre-final-meeting brief for team TS3012. Use ONLY the supplied evidence from meetings a, b, c. Do NOT use meeting d because the brief must be prepared before the final meeting. Cover: (1) decisions already made, (2) what changed and why, (3) open actions and named owners, (4) unresolved disagreements/questions, (5) cost/budget implications. Every factual bullet must include one or more citations in this exact form [TS3012a: "at least six exact consecutive words"] and the quote must be copied from a single supplied speaker turn. If an owner or status is not clear, say "not clear in the transcript". Do not invent completion or consensus. Return plain text with headings and bullets. Evidence: {json.dumps(context,ensure_ascii=False)}'''
    text=llm.ask_text(prompt)
    # Lightweight citation audit: flag quote-looking snippets not found in original speaker turns.
    originals={}
    for p in Path(DATA_DIR).glob("TS3012[abc].txt"):
        for t in parse_transcript(p): originals.setdefault(t["meeting_id"],[]).append(t["text"])
    import re
    citations=re.findall(r'\[(TS3012[abc]):\s*["“](.+?)["”]\]',text)
    bad=[]
    for mid,quote in citations:
        if len(words(quote))<6 or not any(normalize(quote) in normalize(turn) for turn in originals.get(mid,[])):
            bad.append((mid,quote))
    Path("TS3012_brief.txt").write_text(text+"\n",encoding="utf-8")
    Path("TS3012_brief_audit.json").write_text(json.dumps({"citations_found":len(citations),"invalid_or_unmatched":bad},indent=2),encoding="utf-8")
    print(text); print("\nAudit:", "PASS" if not bad and citations else f"REVIEW {bad}")
if __name__=="__main__": main()
