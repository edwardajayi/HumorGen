import json
from typing import List, Dict, Any, Tuple

def extract_jokes(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Extract (joke_id, joke_text) pairs from a data record.
    Auto-detects between:
    1. Flattened columns: keys like 'kimi_joke_1', 'qwen32b_joke_2', 'kimi_v1_joke'
    2. Candidates array: list under 'candidates' key
    Returns tuples of (joke_id, joke_text). Skips empty/ERROR jokes.
    """
    results = []
    
    # 1. Check for 'candidates' array format
    if 'candidates' in record and isinstance(record['candidates'], list) and record['candidates']:
        counts = {}
        for cand in record['candidates']:
            model = cand.get('sm', 'unknown')
            counts[model] = counts.get(model, 0) + 1
            joke_text = cand.get('joke', '')
            if joke_text and "ERROR" not in joke_text:
                joke_id = f"{model}_{counts[model]}"
                results.append((joke_id, joke_text))
        return results

    # 2. Check for flattened column format
    for key, value in record.items():
        # Handle legacy format with explicit _joke_ separator
        if "_joke_" in key and isinstance(value, str) and value.strip():
            if "ERROR" in value:
                continue
            parts = key.rsplit("_joke_", 1)
            if len(parts) == 2:
                model = parts[0]
                number = parts[1]
                joke_id = f"{model}_{number}"
                results.append((joke_id, value))
            else:
                results.append((key, value))
            continue
            
        # Handle format ending in _joke (e.g. kimi_v1_joke)
        if key.endswith("_joke") and isinstance(value, str) and value.strip():
            if "ERROR" in value:
                continue
            joke_id = key[:-5] # remove _joke
            results.append((joke_id, value))
                    
    return results

def save_jsonl(data: List[Dict[str, Any]], file_path: str, append: bool = False):
    mode = 'a' if append else 'w'
    with open(file_path, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        return []
    return data

def save_json(data: Any, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
