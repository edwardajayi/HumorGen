"""
Cognitive Synergy Framework - Restore Generation Module (V5 Legacy)
Restoring the exact logic that produced high-quality jokes.
"""

import torch
import json
import re
import os
import random
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any

# ==============================================================================
# 1. THE ENGINE WRAPPER (Restored V5)
# ==============================================================================
class HumorEngineWrapperV5:
    def __init__(self, model, tokenizer):
        # Auto-correction for swapped arguments (common in notebook calls)
        if hasattr(model, "apply_chat_template") and not hasattr(tokenizer, "apply_chat_template"):
            print("⚠️ WARNING: Arguments swapped! Auto-correcting (model <-> tokenizer)...")
            model, tokenizer = tokenizer, model

        self.model = model
        self.tokenizer = tokenizer
        # Robust device detection
        try:
            self.device = next(model.parameters()).device
        except:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
        # Set left-padding for proper batch generation with decoder-only models
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt, system_prompt, temperature=0.7):
        # Single generation wrapper around batch
        return self.generate_batch([prompt], [system_prompt], temperature)[0]

    def generate_batch(self, prompts, system_prompts, temperature=0.7, batch_size=12):
        # Optimized Batched Generation with sub-batching for OOM protection
        all_results = []
        
        # Prepare all formatted inputs
        formatted_inputs = []
        for p, sp in zip(prompts, system_prompts):
            messages = [{"role": "system", "content": sp}, {"role": "user", "content": p}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_inputs.append(text)
        
        # Process in sub-batches to avoid OOM
        for i in range(0, len(formatted_inputs), batch_size):
            batch_texts = formatted_inputs[i:i + batch_size]
            
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            decoded = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            all_results.extend([d.strip() for d in decoded])
        
        return all_results


# ==============================================================================
# 1B. KIMI K2 ENGINE WRAPPER (API-based generation)
# ==============================================================================
class HumorEngineKimiK2:
    """
    Engine wrapper for Kimi K2 via llama-server (OpenAI-compatible API).
    
    Same interface as HumorEngineWrapperV5, but uses API calls instead of model.generate().
    This allows using Kimi K2 with the same EnsembleMinerV5 pipeline.
    
    Usage:
        from src.model_utils import get_model_kimi_k2
        model, tokenizer = get_model_kimi_k2()
        engine = HumorEngineKimiK2(model)
        miner = EnsembleMinerV5(engine)
    """
    
    def __init__(self, model, tokenizer=None):
        """
        Args:
            model: KimiK2Wrapper instance (from get_model_kimi_k2())
            tokenizer: Optional, ignored (API handles tokenization)
        """
        self.model = model
        self.device = "kimi-k2-server"  # For compatibility
        # Configurable Parallelism
        self.use_parallel = False
        self.max_workers = 4

    def set_parallel(self, enabled: bool, workers: int = 4):
        self.use_parallel = enabled
        self.max_workers = workers
        
    def generate(self, prompt, system_prompt, temperature=0.7):
        """Single generation."""
        return self.generate_batch([prompt], [system_prompt], temperature)[0]
    
    def generate_batch(self, prompts, system_prompts, temperature=0.7, batch_size=12):
        """
        Batch generation via Kimi K2 API.
        
        Note: API calls are sequential by default, but can be parallelized.
        Speed: ~1-2 tok/s per request on H100 + CPU offloading.
        """
        all_results = []
        
        if self.use_parallel:
            # Parallel Execution using ThreadPool
            from concurrent.futures import ThreadPoolExecutor
            
            def _generate_single(p, sp):
                try:
                    res = self.model.generate_text(
                        prompt=p, 
                        system_prompt=sp, 
                        max_tokens=512, 
                        temperature=temperature
                    )
                    return res.strip()
                except Exception as e:
                    print(f"⚠️ Parallel Gen Error: {e}")
                    return ""

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                all_results = list(executor.map(_generate_single, prompts, system_prompts))
        else:
            # Sequential Execution
            for prompt, system_prompt in zip(prompts, system_prompts):
                try:
                    response = self.model.generate_text(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        max_tokens=512,
                        temperature=temperature
                    )
                    all_results.append(response.strip())
                except Exception as e:
                    print(f"⚠️ Kimi K2 generation error: {e}")
                    all_results.append("")
        
        return all_results

# ==============================================================================
# 2. THE "SUPER ENSEMBLE" MINER (Fixed Judge)
# ==============================================================================
class EnsembleMinerV5:
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

        # === THE NEW 3 (SPICY & HIGH VARIANCE) ===
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

        # --- THE FIXED JUDGE (Reasoning-First to prevent "82" spam) ---
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
        # Robust regex extraction
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
        else:
            # Fallback for when models forget the closing tag or use bold
            alt_pattern = f"(?:_|\\*\\*|\\b){tag}(?:\\*\\*)?:\\s*(.*)"
            alt_match = re.search(alt_pattern, text, re.IGNORECASE | re.DOTALL)
            if alt_match:
                raw_content = alt_match.group(1).strip()
                # If extracting THOUGHT, stop at JOKE
                if tag == "THOUGHT" and "JOKE" in raw_content.upper():
                     split_match = re.split(r"(?:_|\\*\\*|\\b|S)JOKE(?:\\*\\*)?:", raw_content, flags=re.IGNORECASE)
                     content = split_match[0].strip()
                # If extracting ANALYSIS, stop at SCORE
                elif tag == "ANALYSIS" and "SCORE" in raw_content.upper():
                     split_match = re.split(r"(?:_|\\*\\*|\\b|S)SCORE(?:\\*\\*)?:", raw_content, flags=re.IGNORECASE)
                     content = split_match[0].strip()
                else:
                    content = raw_content
            else:
                 content = text
                 if f"<{tag}>" in text:
                     content = text.split(f"<{tag}>")[-1].split(f"</{tag}>")[0].strip()
        
        return re.sub(r"^\[.*?\]\s*", "", content)

    def mine(self, prompt_text, task_type, drafts_per_persona=2):
        candidates = []
        clean_input = prompt_text.replace("Headline: ", "").replace("Words: ", "").strip('"')
        
        if task_type == 'headline':
            constraint_desc = "Write a monologue joke related to this headline."
        else:
            constraint_desc = f"Include EXACTLY these words: {clean_input}."

        # 1. BATCHED GENERATION PHASE (fast - ~2 hours for 1200 prompts)
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
                joke = self.extract_tag(raw, "JOKE")
                reasoning = self.extract_tag(raw, "THOUGHT")
                if len(joke) > 900: joke = joke[:900]
                
                candidates.append({
                    "persona": p_name, 
                    "joke": joke, 
                    "reasoning": reasoning, 
                    "score": 0
                })
        except Exception as e:
            print(f"Error in batch generation: {e}")

        # 2. BATCHED JUDGING PHASE (FIXED)
        if candidates:
            judge_prompts = []
            judge_systems = []
            
            for cand in candidates:
                # Pass explicit constraint + joke
                user_msg = f"Constraint: {prompt_text}\nCandidate Joke: \"{cand['joke']}\""
                judge_prompts.append(user_msg)
                judge_systems.append(self.JUDGE_PROMPT)
                
            try:
                # Increased temperature slightly (0.6) to allow diverse reasoning
                score_strs = self.engine.generate_batch(judge_prompts, judge_systems, temperature=0.6)
                
                for cand, output in zip(candidates, score_strs):
                    # 1. Try to find the explicit tag
                    score_text = self.extract_tag(output, "SCORE")
                    
                    # 2. Cleanup non-digits
                    nums = re.findall(r'\d+', score_text)
                    
                    # 3. Fallback: if tag extraction failed, look at the whole string
                    if not nums:
                        nums = re.findall(r'\d+', output)

                    # 4. Assignment
                    if nums:
                        cand['score'] = int(nums[0])
                    else:
                        cand['score'] = 0

            except Exception as e:
                print(f"Error in batch judging: {e}")
            
        # 3. SORTING & RANKING
        sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        return sorted_candidates

# ==============================================================================
# 3. PIPELINE RUNNER
# ==============================================================================
def run_generation_pipeline_v5(
    model, 
    tokenizer, 
    input_file: str, 
    output_jsonl: str, 
    output_jsonl_no_scores: str = None,
    use_subset: bool = True, 
    subset_size: int = 50,
    drafts_per_persona: int = 1,
    use_parallel: bool = False,
    n_workers: int = 4
):
    print(f"--- Starting V5 (Legacy) Pipeline [Parallel={use_parallel}, Workers={n_workers}] ---")
    
    # Create or use the engine
    # If model is already an engine (e.g., HumorEngineKimiK2), use it directly
    if isinstance(model, HumorEngineKimiK2):
        engine = model
        engine.set_parallel(use_parallel, n_workers)
    else:
        # Wrap local model with V5 engine
        engine = HumorEngineWrapperV5(model, tokenizer)
    miner = EnsembleMinerV5(engine)
    
    # Load Data
    df = pd.read_csv(input_file, sep='\t')
    all_tasks = []
    for _, row in df.iterrows():
        if row['headline'] != '-': 
            all_tasks.append({"id": row['id'], "type": "headline", "prompt": f"Headline: \"{row['headline']}\""})
        elif row['word1'] != '-': 
            all_tasks.append({"id": row['id'], "type": "words", "prompt": f"Words: \"{row['word1']}, {row['word2']}\""})

    # Subsetting
    if use_subset:
        batch = all_tasks[:subset_size]
    else:
        batch = all_tasks

    # === RESUME CAPABILITY ===
    # Check which IDs have already been processed
    # === RESUME CAPABILITY ===
    # Check which IDs have already been processed
    processed_ids = set()
    total_written = 0
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    processed_ids.add(entry.get('id'))
                    total_written += 1
                except:
                    pass
        print(f"📁 Found {len(processed_ids)} already processed items ({total_written} lines). Resuming...")
    
    # Filter out already processed
    batch = [row for row in batch if row['id'] not in processed_ids]
    
    if len(batch) == 0:
        print(" All items already processed! Nothing to do.")
        return
    
    print(f"Processing {len(batch)} remaining items...")
    
    # Open both files in append mode
    f_no_scores = None
    if output_jsonl_no_scores:
        f_no_scores = open(output_jsonl_no_scores, 'a')

    with open(output_jsonl, 'a') as f:
        for row in tqdm(batch):
            candidates = miner.mine(row['prompt'], row['type'], drafts_per_persona=drafts_per_persona)
            
            if not candidates:
                continue

            # V5 Style Log Entry
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
            os.fsync(f.fileno()) # Force write to disk immediately

            # Write to no_scores file if available
            if f_no_scores:
                # Remove scores/reasoning for clean training data
                clean_candidates = [
                    {"persona": c["persona"], "joke": c["joke"], "reasoning": c.get("reasoning", "")}
                    for c in candidates
                ]
                clean_entry = {
                    "id": row['id'],
                    "prompt": row['prompt'],
                    "model": "kimi_k2_0905",
                    "candidates": clean_candidates,
                    "winner_joke": candidates[0].get('joke'),
                }
                f_no_scores.write(json.dumps(clean_entry) + "\n")
                f_no_scores.flush()
                os.fsync(f_no_scores.fileno())

    if f_no_scores:
        f_no_scores.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine_type", type=str, choices=["Local", "KimiK2"], default="Local")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file_scores", type=str, required=True)
    parser.add_argument("--output_file_no_scores", type=str, required=True)
    parser.add_argument("--num_drafts", type=int, default=2)
    parser.add_argument("--max_headlines", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    # New Parallel Arguments
    parser.add_argument("--parallel", action="store_true", help="Enable parallel generation")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")

    args = parser.parse_args()

    # Model Loading Logic
    if args.engine_type == "KimiK2":
        from src.model_utils import get_model_kimi_k2
        # We don't need real model object here if using API wrapper, but let's stick to pattern
        model_wrapper, _ = get_model_kimi_k2() 
        engine = HumorEngineKimiK2(model_wrapper)
    else:
        # Load Local Model (Not implemented fully here for brevity, assume passed correctly)
        pass 

    run_generation_pipeline_v5(
        engine, 
        None, 
        args.input_file, 
        args.output_file_scores, 
        output_jsonl_no_scores=args.output_file_no_scores,
        use_subset=True, 
        subset_size=args.max_headlines,
        drafts_per_persona=args.num_drafts,
        use_parallel=args.parallel,
        n_workers=args.workers
    )

if __name__ == "__main__":
    main()
