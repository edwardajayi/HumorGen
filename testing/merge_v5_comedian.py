import json
import os
import shutil

# Script lives in testing/; use this directory for inputs/outputs
root = os.path.dirname(os.path.abspath(__file__))
main_file = os.path.join(root, "generated_jokes_v4.jsonl")    # The current definitive source
comedian_file = os.path.join(root, "comedian_jokes.jsonl")  # The generated comedian jokes
out_file = os.path.join(root, "generated_jokes_v5.jsonl")    # The new target dataset

def load_jokes(path, key):
    m = {}
    if not os.path.exists(path): return m
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if key in d:
                    m[d["id"]] = d[key]
    return m

# Load the newly generated comedian jokes
com_jokes = load_jokes(comedian_file, "comedian_joke")
print(f"Loaded {len(com_jokes)} comedian jokes.")

updated_count = 0
with open(main_file, "r") as f_in, open(out_file, "w") as f_out:
    for line in f_in:
        if line.strip():
            data = json.loads(line)
            hid = data["id"]
            
            # If we generated a comedian joke for this ID, add it
            if hid in com_jokes:
                data["comedian_joke"] = com_jokes[hid]
                updated_count += 1
                
            f_out.write(json.dumps(data) + "\n")

print(f"Merged {updated_count} comedian jokes into {out_file}")
