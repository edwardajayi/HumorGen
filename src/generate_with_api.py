#!/usr/bin/env python3
"""
Cognitive Synergy Framework - API Edition (Groq + Kimi K2)
Restoring the exact logic that produced high-quality jokes, but using Groq API.
Includes:
- Intelligent Rate Limiting (Sliding Window)
- Token Usage Tracking (token_usage.json)
- Selective Judging (to save tokens)
- Resume Support
"""

import os
import sys
import json
import re
import time
import random 
from collections import deque
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from repo root
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ==============================================================================
# 0. HELPERS: Rate Limiter & Token Tracker
# ==============================================================================
class RateLimiter:
    """Manages Token Per Minute (TPM) limits using a sliding window."""
    def __init__(self, limit_tpm=10000, buffer_ratio=0.98):
        self.limit = int(limit_tpm * buffer_ratio)
        self.history = deque() # Stores (timestamp, tokens_used)
        print(f"🚦 Rate Limiter initialized: {self.limit} TPM limit")

    def estimate_tokens(self, text: str) -> int:
        return len(str(text)) // 4 + 5

    def wait_for_capacity(self, needed_tokens: int):
        while True:
            now = time.time()
            while self.history and self.history[0][0] < now - 60:
                self.history.popleft()
            
            current_usage = sum(count for _, count in self.history)
            
            if current_usage + needed_tokens <= self.limit:
                return
            
            if self.history:
                oldest_time = self.history[0][0]
                wait_time = (oldest_time + 61) - now
                if wait_time < 0.1: wait_time = 0.1
            else:
                wait_time = 1.0
                
            print(f"⏳ Rate limit approaching ({current_usage}/{self.limit}). Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)

    def add_usage(self, tokens: int):
        self.history.append((time.time(), tokens))

class TokenTracker:
    """Tracks total token usage and saves to file."""
    def __init__(self, output_path="token_usage.json"):
        self.output_path = output_path
        self.usage = {"total_tokens": 0, "calls": 0, "breakdown": {}}
        self.load()

    def load(self):
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r') as f:
                    self.usage = json.load(f)
            except: pass

    def update(self, model: str, tokens: int):
        self.usage["total_tokens"] += tokens
        self.usage["calls"] += 1
        self.usage["breakdown"][model] = self.usage["breakdown"].get(model, 0) + tokens
        
        # Save every update (safe for sequential)
        with open(self.output_path, 'w') as f:
            json.dump(self.usage, f, indent=2)

# ==============================================================================
# 1. THE ENGINE WRAPPER (Groq API)
# ==============================================================================
class HumorEngineGroq:
    def __init__(self, model_name="moonshotai/kimi-k2-instruct-0905", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature
        
        # API keys from environment: GROQ_API_KEY (single) or GROQ_API_KEYS (comma-separated for rotation)
        keys_env = os.getenv("GROQ_API_KEYS") or os.getenv("GROQ_API_KEY") or ""
        self.api_keys = [k.strip() for k in keys_env.split(",") if k.strip()]
        if not self.api_keys:
            raise ValueError(
                "Set GROQ_API_KEY or GROQ_API_KEYS (comma-separated) in environment or .env"
            )
        self.current_key_idx = 0
        self.rate_limiter = RateLimiter(limit_tpm=10000)
        self.token_tracker = TokenTracker()
            
        print(f"init Groq Agent with model: {self.model_name}")
        print(f"Loaded {len(self.api_keys)} API key(s). Starting with Key #{self.current_key_idx + 1}")
        
        self._init_client()

    def _init_client(self):
        """Initialize the ChatGroq client with the current key."""
        current_key = self.api_keys[self.current_key_idx]
        # Mask key for logging
        masked_key = current_key[:4] + "..." + current_key[-4:] if len(current_key) > 8 else "***"
        print(f"Connecting with Key #{self.current_key_idx + 1} ({masked_key})")
        
        self.client = ChatGroq(
            api_key=current_key, 
            model=self.model_name, 
            temperature=self.temperature, 
            max_retries=3
        )

    def _rotate_key(self):
        """Switch to the next API key."""
        self.current_key_idx += 1
        if self.current_key_idx >= len(self.api_keys):
            print(f"\nALL {len(self.api_keys)} API keys exhausted.")
            print(f"Total tokens used so far: {self.token_tracker.usage['total_tokens']}")
            print("Stopping execution safely.")
            sys.exit(1)
        
        print(f"\nRotating to Key #{self.current_key_idx + 1}...")
        self._init_client()

    def generate(self, prompt, system_prompt):
        """Single generation with key rotation."""
        input_tokens = self.rate_limiter.estimate_tokens(prompt + system_prompt)
        self.rate_limiter.wait_for_capacity(input_tokens + 150)
        
        max_key_retries = len(self.api_keys)
        
        while True:
            try:
                messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
                response = self.client.invoke(messages)
                content = response.content.strip()
                
                output_tokens = self.rate_limiter.estimate_tokens(content)
                total_tokens = input_tokens + output_tokens
                
                self.rate_limiter.add_usage(total_tokens)
                self.token_tracker.update(self.model_name, total_tokens)
                
                return content
                
            except Exception as e:
                error_str = str(e).lower()
                if "quota" in error_str or "limit reached" in error_str:
                     print(f"⚠️ Quota hit on Key #{self.current_key_idx + 1}. Attempting rotation...")
                     self._rotate_key()
                     # Loop continues with new key
                elif "429" in error_str or "rate limit" in error_str:
                    print(f"⚠️ 429 Hit. Waiting 10s...")
                    time.sleep(10)
                else:
                    self._handle_error(e)
                    return ""

    def generate_batch(self, prompts, system_prompts, temperature=None):
        """Batch generation using sequential calls with key rotation."""
        # Note: LangChain ChatGroq binding for temp is cleaner but re-init is safer for now
        
        # If temp changes, we need a temp client, but we must respect the CURRENT KEY
        # To avoid complexity, we'll just check if temp differs, and if so, 
        # we might need to rely on the main client's update if we want rotation to persist nicely.
        # However, for simplicity, let's just use the current client if temp matches,
        # or create a temporary one if it doesn't (BUT this temp one won't rotate the main self.current_key_idx globally easily).
        # BETTER STRATEGY: Update self.client's temp if needed, or instantiate a new one with current key.
        
        current_client = self.client
        if temperature is not None and temperature != self.temperature:
             # Create ad-hoc client with CURRENT key
             current_client = ChatGroq(
                 api_key=self.api_keys[self.current_key_idx], 
                 model=self.model_name, 
                 temperature=temperature, 
                 max_retries=3
            )

        results = []
        for p, sp in zip(prompts, system_prompts):
            input_tokens = self.rate_limiter.estimate_tokens(p + sp)
            self.rate_limiter.wait_for_capacity(input_tokens + 150)

            # Retry Loop
            max_retries = 5
            base_wait = 2
            
            # Key Rotation Loop
            while True: 
                # Inner loop for standard retries (network/500s)
                success = False
                for attempt in range(max_retries + 1):
                    try:
                        # Ensure we use the latest key for ad-hoc clients too
                        if temperature is not None and temperature != self.temperature:
                            current_client = ChatGroq(
                                api_key=self.api_keys[self.current_key_idx], 
                                model=self.model_name, 
                                temperature=temperature, 
                                max_retries=3
                            )
                        else:
                            current_client = self.client # Always use updated self.client

                        messages = [SystemMessage(content=sp), HumanMessage(content=p)]
                        response = current_client.invoke(messages)
                        content = response.content.strip()
                        results.append(content)
                        
                        output_tokens = self.rate_limiter.estimate_tokens(content)
                        total_tokens = input_tokens + output_tokens
                        
                        self.rate_limiter.add_usage(total_tokens)
                        self.token_tracker.update(self.model_name, total_tokens)
                        success = True
                        break # Success, exit retry loop
                        
                    except Exception as e:
                        error_str = str(e).lower()
                        
                        if "quota" in error_str or "limit reached" in error_str:
                            print(f"\n⚠️ QUOTA HIT on Key #{self.current_key_idx + 1}. initiating rotation...")
                            self._rotate_key()
                            # Break out of the retry loop to restart with new key (the outer while loop)
                            break 
                            
                        elif "429" in error_str or "rate limit" in error_str:
                            wait = 15 * (attempt + 1)
                            print(f"⚠️ 429 Hit. Waiting {wait}s...")
                            time.sleep(wait)
                        elif "500" in error_str or "503" in error_str or "capacity" in error_str or "internal" in error_str:
                            wait = base_wait * (2 ** attempt) + random.uniform(0, 1)
                            print(f"⚠️ Service Over Capacity (503). Retrying in {wait:.1f}s... (Attempt {attempt+1}/{max_retries})")
                            time.sleep(wait)
                        else:
                            print(f"⚠️ Groq Gen Error: {e}")
                            # Treat unknown errors as fatal for this item after retries
                            if attempt == max_retries:
                                break
                
                if success:
                    break # Next item
                
                # If we exhausted retries on non-quota errors, give up
                if not success and "quota" not in error_str:
                     print(f"❌ Failed to generate after retries.")
                     results.append("") 
                     break
        
        return results

    def _handle_error(self, e):
        error_str = str(e).lower()
        if "quota" in error_str or "limit reached" in error_str:
            print(f"\n🚨 API QUOTA EXHAUSTED or HARD LIMIT for {self.model_name}! 🚨")
            print(f"Total tokens used so far: {self.token_tracker.usage['total_tokens']}")
            print("Stopping execution safely. Update API key in .env and restart script to resume.")
            sys.exit(1)
        elif "429" in error_str or "rate limit" in error_str:
            print(f"⚠️ 429 Hit. Waiting 10s...")
            time.sleep(10)
        else:
            print(f"⚠️ Groq Gen Error: {e}")
            time.sleep(1)

# ==============================================================================
# 2. THE "SUPER ENSEMBLE" MINER
# ==============================================================================
class EnsembleMinerGroq:
    def __init__(self, engine):
        self.engine = engine
        
        # === THE ORIGINAL 3 (SAFE & RELIABLE) ===
        self.P1_OBSERVER = """You are an Observational Comedian (Style: Jerry Seinfeld).
        Task: Write a GENUINELY HILARIOUS joke. This must make people laugh out loud.
        BE BOLD. BE SURPRISING. Take creative risks. Mediocre jokes are failures.
        SAFETY: NO racism, sexism, slurs, or punching down at vulnerable groups. Dark humor is OK but never mean-spirited.
        Technique: 'The Relatable Truth'. Ask "What's the deal with this?" and find the mundane absurdity.
        Constraint: {constraint_instruction}
        Input: "{input_text}"
        Output Format:
        <THOUGHT> [Your observation] </THOUGHT>
        <JOKE> [The joke - make it MEMORABLE and QUOTABLE] </JOKE>"""

        self.P2_WORDSMITH = """You are a Witty Wordsmith - MASTER of wordplay.
        Task: Write a BRILLIANTLY clever joke. The wordplay must be sharp and surprising.
        BE CREATIVE. Push boundaries. Obvious puns are lazy - find the unexpected twist.
        SAFETY: NO racism, sexism, slurs, or punching down at vulnerable groups. Clever wordplay is always clean.
        Technique: 'The Linguistic Twist'. Use double meanings, puns, or precise vocabulary to flip the meaning.
        Constraint: {constraint_instruction}
        Input: "{input_text}"
        Output Format:
        <THOUGHT> [Your wordplay logic] </THOUGHT>
        <JOKE> [The joke - make it CLEVER and SURPRISING] </JOKE>"""

        self.P3_OPTIMIST = """You are a Cheerful Optimist with INFECTIOUS humor.
        Task: Write a joke so funny it makes people smile uncontrollably.
        BE ABSURDLY POSITIVE. Find the most ridiculous silver lining possible.
        SAFETY: NO racism, sexism, slurs, or punching down at vulnerable groups. Keep it wholesome but hilarious.
        Technique: 'The Innocent Interpretation'. Take things literally or find a silly silver lining in a bad situation.
        Constraint: {constraint_instruction}
        Input: "{input_text}"
        Output Format:
        <THOUGHT> [Your innocent logic] </THOUGHT>
        <JOKE> [The joke - make it DELIGHTFULLY ABSURD] </JOKE>"""

        self.P4_ABSURDIST = """You are an Absurdist Comedian (Style: Mitch Hedberg) - MASTER of the unexpected.
        Task: Write a WILDLY FUNNY joke that catches people completely off guard.
        GO WEIRD. The more surreal and unexpected, the better. Safe jokes are boring.
        SAFETY: NO racism, sexism, slurs, or punching down at vulnerable groups. Absurd ≠ offensive.
        Technique: 'The Non-Sequitur'. Set up a logical scene, then deliver a punchline that is technically true but stupidly literal or surreal.
        Constraint: {constraint_instruction}
        Input: "{input_text}"
        Output Format:
        <THOUGHT> [Surreal logic] </THOUGHT>
        <JOKE> [Joke - make it BIZARRE and UNFORGETTABLE] </JOKE>"""

        self.P5_CYNIC = """You are a Cynical Satirist (Style: Ricky Gervais) - VICIOUSLY funny.
        Task: Write a DEVASTATINGLY funny joke that makes people laugh AND wince.
        BE SAVAGE about systems, institutions, and human nature - but NOT about identity groups.
        SAFETY: NO racism, sexism, slurs, or punching down at vulnerable groups. Punch UP at the powerful, not DOWN.
        Technique: 'The Brutal Truth'. What is the selfish, dark, or depressing reality behind this? Make us laugh at the misery.
        Constraint: {constraint_instruction}
        Input: "{input_text}"
        Output Format:
        <THOUGHT> [Dark logic] </THOUGHT>
        <JOKE> [Joke - make it BITING and PAINFULLY TRUE] </JOKE>"""

        self.P6_NEUROTIC = """You are a Neurotic Overthinker (Style: George Costanza) - HILARIOUSLY anxious.
        Task: Write a joke so relatable it makes people say "That's so true!"
        GO DEEP on the anxiety. Find the most ridiculous thing to worry about.
        SAFETY: NO racism, sexism, slurs, or punching down at vulnerable groups. Anxiety comedy is always self-directed.
        Technique: 'The Spiraling Anxiety'. Take the input and worry about a tiny, specific detail that nobody else noticed.
        Constraint: {constraint_instruction}
        Input: "{input_text}"
        Output Format:
        <THOUGHT> [Anxious logic] </THOUGHT>
        <JOKE> [Joke - make the worry ABSURDLY SPECIFIC and RELATABLE] </JOKE>"""

        self.JUDGE_PROMPT = """You are an experienced comedy and humor judge. Evaluate the following joke.
SCORING CRITERIA:
- 90-100: Very funny. The joke lands well, is memorable, and fits the constraint.
- 75-89: Good. Coherent and funny, but maybe a bit standard.
- 60-74: Mediocre. Makes sense, but lacks "spark" or is cliché.
- 0-59: Poor. Confusing, unfunny, or misses the constraint.
IMPORTANT: Do not just give 80 or 82 to everything. Be critical. If it's generic, give it a 65-75.
Constraint: {input_desc}
Candidate Joke: "{joke}"
Step 1: Write a 1-sentence critique in <ANALYSIS> tags.
Step 2: Assign a score in <SCORE> tags.
Output Format:
<ANALYSIS> ... </ANALYSIS>
<SCORE> ... </SCORE>"""

    def extract_tag(self, text, tag):
        # Robust regex extraction with case insensitivity
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            return re.sub(r"^\[.*?\]\s*", "", content)
        
        # Fallback 1: Text headers like "JOKE:" or "**JOKE**:"
        alt_pattern = f"(?:_|\\*\\*|\\b){tag}(?:\\*\\*)?:\\s*(.*)"
        alt_match = re.search(alt_pattern, text, re.IGNORECASE | re.DOTALL)
        if alt_match:
            raw_content = alt_match.group(1).strip()
            if tag == "THOUGHT" and "JOKE" in raw_content.upper():
                 split_match = re.split(r"(?:_|\\*\\*|\\b|S)JOKE(?:\\*\\*)?:", raw_content, flags=re.IGNORECASE)
                 return split_match[0].strip()
            elif tag == "ANALYSIS" and "SCORE" in raw_content.upper():
                 split_match = re.split(r"(?:_|\\*\\*|\\b|S)SCORE(?:\\*\\*)?:", raw_content, flags=re.IGNORECASE)
                 return split_match[0].strip()
            else:
                return raw_content

        # Fallback 2: Simple string splitting if regex failed (e.g. malformed attributes)
        if f"<{tag}>" in text:
             return text.split(f"<{tag}>")[-1].split(f"</{tag}>")[0].strip()
        
        # Fallback 3: Return empty string instead of full text to prevent pollution
        return ""

    def mine(self, prompt_text, task_type, drafts_per_persona=1, judge_target="all"):
        candidates = []
        clean_input = prompt_text.replace("Headline: ", "").replace("Words: ", "").strip('"')
        
        # --- Prompt Customization ---
        if task_type == 'headline':
            # Added brevity constraint as requested
            constraint_desc = "Write a monologue joke related to this headline. Keep it concise (under 280 characters if possible)."
        else:
            constraint_desc = f"Include EXACTLY these words: {clean_input}. Keep it concise."

        # 1. GENERATION PHASE
        personas = [
            (self.P1_OBSERVER, "Observer"),
            (self.P2_WORDSMITH, "Wordsmith"),
            (self.P3_OPTIMIST, "Optimist"),
            (self.P4_ABSURDIST, "Absurdist"),
            (self.P5_CYNIC, "Cynic"),
            (self.P6_NEUROTIC, "Neurotic")
        ]
        
        batch_prompts = []
        batch_systems = []
        batch_meta = []

        for prompt_temp, p_name in personas:
            full_prompt = prompt_temp.format(constraint_instruction=constraint_desc, input_text=clean_input)
            user_msg = f"Input: {clean_input}"
            for _ in range(drafts_per_persona):
                batch_prompts.append(user_msg)
                batch_systems.append(full_prompt)
                batch_meta.append(p_name)
        
        try:
            results = self.engine.generate_batch(batch_prompts, batch_systems, temperature=0.95)
            for raw, p_name in zip(results, batch_meta):
                if not raw: continue
                joke = self.extract_tag(raw, "JOKE")
                reasoning = self.extract_tag(raw, "THOUGHT")
                if len(joke) > 900: joke = joke[:900]
                candidates.append({
                    "persona": p_name, "joke": joke, "reasoning": reasoning, "score": 0
                })
        except Exception as e:
            print(f"Error in batch generation: {e}")

        # 2. JUDGING PHASE (Selective)
        should_judge = False
        if judge_target == "all":
            should_judge = True
        elif judge_target == "words" and task_type == 'words':
            should_judge = True
        elif judge_target == "headline" and task_type == 'headline':
            should_judge = True
            
        if candidates and should_judge:
            judge_prompts = []
            judge_systems = []
            for cand in candidates:
                user_msg = f"Constraint: {prompt_text}\nCandidate Joke: \"{cand['joke']}\""
                judge_prompts.append(user_msg)
                judge_systems.append(self.JUDGE_PROMPT)
                
            try:
                score_strs = self.engine.generate_batch(judge_prompts, judge_systems, temperature=0.6)
                for cand, output in zip(candidates, score_strs):
                    score_text = self.extract_tag(output, "SCORE")
                    nums = re.findall(r'\\d+', score_text)
                    if not nums: nums = re.findall(r'\\d+', output)
                    cand['score'] = int(nums[0]) if nums else 0
            except Exception as e:
                print(f"Error in batch judging: {e}")
        else:
             # If skipped judging, give random or placeholder scores so we can still output
             # Maybe rely on the generator strength.
             if candidates:
                 pass # Scores remain 0

        # 3. SORTING
        sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        return sorted_candidates

# ==============================================================================
# 3. PIPELINE RUNNER
# ==============================================================================
def run_generation_pipeline_api(
    input_file: str, 
    output_jsonl: str, 
    model_name: str,
    max_items: int = 50,
    drafts_per_persona: int = 1,
    judge_target: str = "words" # Default per user request "only examples that are words"
):
    print(f"--- Starting API Pipeline (Model: {model_name}) ---")
    print(f"--- Judge Target: {judge_target} ---")
    
    engine = HumorEngineGroq(model_name=model_name)
    miner = EnsembleMinerGroq(engine)
    
    # Load Data
    try:
        df = pd.read_csv(input_file, sep='\\t')
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    all_tasks = []
    if 'headline' in df.columns:
        for _, row in df.iterrows():
            if row['headline'] != '-': 
                all_tasks.append({"id": row['id'], "type": "headline", "prompt": f"Headline: \"{row['headline']}\""})
            elif row['word1'] != '-': 
                all_tasks.append({"id": row['id'], "type": "words", "prompt": f"Words: \"{row['word1']}, {row['word2']}\""})
    else:
        print("Warning: 'headline' column not found.")

    processed_ids = set()
    if os.path.exists(output_jsonl):
        print(f"Checking for existing progress in {output_jsonl}...")
        with open(output_jsonl, 'r') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line.strip()).get('id'))
                except: pass
    
    tasks_to_do = [t for t in all_tasks if t['id'] not in processed_ids]
    if max_items and max_items > 0:
        tasks_to_do = tasks_to_do[:max_items]

    print(f"Resuming generation. Skipped {len(processed_ids)} already done.")
    print(f"Processing {len(tasks_to_do)} new items...")

    if not tasks_to_do:
        print("✅ All requested items already completed!")
        return

    with open(output_jsonl, 'a') as f:
        for row in tqdm(tasks_to_do, desc="Generating Jokes", unit="joke"):
            candidates = miner.mine(
                row['prompt'], 
                row['type'], 
                drafts_per_persona=drafts_per_persona,
                judge_target=judge_target
            )
            
            if not candidates:
                continue

            log_entry = {
                "id": row['id'],
                "prompt": row['prompt'],
                "candidates": candidates,
                "winner_persona": candidates[0].get('persona'),
                "winner_score": candidates[0].get('score'),
                "winner_joke": candidates[0].get('joke'),
                "winner_reasoning": candidates[0].get('reasoning', '')
            }
            f.write(json.dumps(log_entry) + "\n")
            f.flush()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None, help="Input TSV; default: data/datasets/mwahaha/task-a-en.tsv in repo")
    parser.add_argument("--output_file", type=str, default="results/new_cog_generated_jokes.jsonl")
    parser.add_argument("--model", type=str, default="moonshotai/kimi-k2-instruct-0905")
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--num_drafts", type=int, default=2)
    parser.add_argument("--judge_target", type=str, default="none", 
                        choices=["all", "words", "headline", "none"],
                        help="Which tasks to run judge on (to save tokens). Default: words")

    args = parser.parse_args()
    if args.input_file is None:
        args.input_file = str(_REPO_ROOT / "data" / "datasets" / "mwahaha" / "task-a-en.tsv")

    run_generation_pipeline_api(
        input_file=args.input_file,
        output_jsonl=args.output_file,
        model_name=args.model,
        max_items=args.max_items,
        drafts_per_persona=args.num_drafts,
        judge_target=args.judge_target
    )

if __name__ == "__main__":
    main()
