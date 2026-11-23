"""Extract structured decisions/actions/numbers per meeting; resumable and citation-audited."""
import json, time
from pathlib import Path
from tqdm import tqdm
from .config import DATA_DIR
from .transcripts import parse_transcript
from .gemini import Gemini

OUT=Path('temporal_memory.json')

def main():
    llm=Gemini()
    existing={}
    if OUT.exists():
        try: existing={x['meeting_id']:x for x in json.loads(OUT.read_text(encoding='utf-8')).get('meetings',[])}
        except Exception: existing={}
    paths=sorted(Path(DATA_DIR).glob('*.txt'))
    if not paths: raise SystemExit(f'No transcript files in {DATA_DIR}')
    for path in tqdm(paths,desc='Extracting meeting memory'):
        turns=parse_transcript(path)
        if not turns: continue
        mid=turns[0]['meeting_id']
        if mid in existing: continue
        transcript='\n'.join(f"{t['timestamp']} {t['speaker']}: {t['text']}" for t in turns)
        prompt=f'''Extract structured project memory from this ONE meeting transcript. Do not infer facts absent from it.
Return JSON with exactly these top-level keys: meeting_id, team, decisions, actions, numbers, open_questions.
Each decisions item: topic, decision, status (proposed/agreed/rejected/changed/unclear), reason, speaker, quote.
Each actions item: task, owner (speaker letter/name only if stated; otherwise unknown), status (open/completed/unclear), speaker, quote.
Each numbers item: subject, value_and_unit, speaker, quote.
Each open_questions item: question, owner_if_known, speaker, quote.
All quotes MUST be copied exactly from one speaker turn in this transcript and be at least six words. If no supported items in a category, return an empty array. Keep speaker labels as Speaker A/B/C/D. Meeting: {mid}; Team: {turns[0]['team']}.
TRANSCRIPT:\n{transcript}'''
        last_error=None
        for attempt in range(3):
            try:
                item=llm.ask_json(prompt)
                item['meeting_id']=mid; item['team']=turns[0]['team']
                # Deterministic quote audit. Keep only extracted facts with valid exact quotes.
                for key in ('decisions','actions','numbers','open_questions'):
                    clean=[]
                    for fact in item.get(key,[]):
                        quote=str(fact.get('quote',''))
                        wc=len(quote.split())
                        exact=wc>=6 and any(' '.join(quote.casefold().split()) in ' '.join(t['text'].casefold().split()) for t in turns)
                        if exact: clean.append(fact)
                    item[key]=clean
                existing[mid]=item
                OUT.write_text(json.dumps({'meetings':[existing[k] for k in sorted(existing)]},indent=2,ensure_ascii=False),encoding='utf-8')
                last_error=None
                break
            except Exception as e:
                last_error=str(e); time.sleep(3*(attempt+1))
        if last_error:
            print(f'\nFAILED {mid}: {last_error}')
        # Free-tier friendly pacing; resume file prevents repeating completed meetings.
        time.sleep(4.2)
    print(f'\nSaved {len(existing)} meeting records to {OUT}. Review extracted facts and statuses.')

if __name__=='__main__': main()
