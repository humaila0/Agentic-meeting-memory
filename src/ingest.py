"""Read all source transcripts and build the local searchable memory."""
from pathlib import Path
from tqdm import tqdm
from .config import DATA_DIR
from .transcripts import parse_transcript, split_long_turn
from .store import TranscriptStore

def main():
    paths=sorted(Path(DATA_DIR).glob("*.txt"))
    if not paths: raise SystemExit(f"No .txt transcripts found in {DATA_DIR}")
    all_chunks=[]; meetings=set(); teams=set(); turns=0
    for path in tqdm(paths,desc="Parsing meetings"):
        records=parse_transcript(path); turns+=len(records)
        for r in records:
            meetings.add(r["meeting_id"]); teams.add(r["team"])
            all_chunks.extend(split_long_turn(r))
    store=TranscriptStore(); store.reset(); store.add_records(all_chunks)
    print(f"\nFiles: {len(paths)} | Meetings: {len(meetings)} | Teams: {len(teams)}")
    print(f"Speaker turns: {turns} | Searchable chunks: {len(all_chunks)}")
    print("Local vector database created in chroma_db/")
if __name__=="__main__": main()
