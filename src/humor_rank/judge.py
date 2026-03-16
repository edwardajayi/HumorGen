import os
import json
import time
from typing import Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import torch

# Try importing local model utils
try:
    from src.model_utils import get_model_4bit, get_model_robust, cleanup_gpu
except ImportError:
    # Handle relative import if needed
    try:
        from ...model_utils import get_model_4bit, get_model_robust
    except ImportError:
        pass

# Load environment variables
load_dotenv()

# =============================================================================
# JUDGE PROMPTS — All 3 versions preserved for paper reporting
# =============================================================================

# ── V1: Original prompt (TIE allowed freely) — produced ~62% tie rate ────────
# PAIRWISE_SYSTEM_PROMPT = """You are a professional comedy critic. Your task is to compare two jokes and decide which is funnier.
#
# You must ignore the length of the joke. Do NOT penalize long jokes. A long, narrative joke with a good payoff is just as valuable as a short, punchy one.
# You must be objective and look for:
# 1. Surprise / Incongruity
# 2. Cleverness / Wordplay
# 3. Narrative structure (if applicable)
#
# Output your analysis in JSON format."""
#
# PAIRWISE_USER_TEMPLATE = """Compare these two jokes based on the prompt: "{original_prompt}"
#
# JOKE A:
# {joke_a}
#
# JOKE B:
# {joke_b}
#
# Step-by-Step Analysis:
# 1. Analyze Joke A's technique and flaws.
# 2. Analyze Joke B's technique and flaws.
# 3. Compare them directly (ignoring length).
# 4. Decide the winner.
#
# Return JSON exactly like this:
# {{
#   "reasoning": "Joke A uses a clever pun on X, while Joke B is too literal...",
#   "decision": "A" or "B" or "TIE",
#   "winner_features": [SELECT FROM: {allowed_features}],
#   "loser_features": ["cliché", "too_long", "confusing", "weak_punchline", "offensive"]
# }}"""

# ── V2: Forced-choice + structured analysis — produced 0% tie rate ────────────
# PAIRWISE_SYSTEM_PROMPT = """You are a professional comedy critic. Your task is to compare two jokes and decide which is funnier.
#
# You must ignore the length of the joke. Do NOT penalize long jokes. A long, narrative joke with a good payoff is just as valuable as a short, punchy one.
# You must be objective and look for:
# 1. Surprise / Incongruity
# 2. Cleverness / Wordplay
# 3. Narrative structure (if applicable)
#
# Strongly prefer picking a winner (A or B). Only declare a TIE if the jokes are genuinely indistinguishable after careful analysis — ties should be rare, not a default when uncertain.
#
# Output your analysis in JSON format."""
#
# PAIRWISE_USER_TEMPLATE = """Compare these two jokes based on the prompt: "{original_prompt}"
#
# JOKE A:
# {joke_a}
#
# JOKE B:
# {joke_b}
#
# Step-by-Step Analysis:
# 1. Analyze Joke A's technique and flaws.
# 2. Analyze Joke B's technique and flaws.
# 3. Compare them directly (ignoring length).
# 4. Decide the winner.
#
# Return JSON exactly like this:
# {{
#   "reasoning": "Joke A uses a clever pun on X, while Joke B is too literal...",
#   "decision": "A" or "B" or "TIE",  # TIE only if jokes are truly indistinguishable
#   "winner_features": [SELECT FROM: {allowed_features}],
#   "loser_features": ["cliché", "too_long", "confusing", "weak_punchline", "offensive"]
# }}"""

# ── V3 (ACTIVE): Simplified, first-impression — natural tie rate (~15-20%) ────
PAIRWISE_SYSTEM_PROMPT = """You are a comedy critic judging which of two jokes is funnier.
Be direct and honest. If one joke is clearly better, pick it. If they are genuinely equal in quality, say TIE.
Do not overthink it — trust your first impression. Output JSON only."""

PAIRWISE_USER_TEMPLATE = """Prompt: "{original_prompt}"

JOKE A: {joke_a}

JOKE B: {joke_b}

Which is funnier? Return JSON:
{{
  "reasoning": "brief explanation",
  "decision": "A" or "B" or "TIE",
  "winner_features": [SELECT FROM: {allowed_features}],
  "loser_features": ["cliché", "too_long", "confusing", "weak_punchline", "offensive"]
}}"""

# Constrained Feature List from Baseline
ALLOWED_FEATURES = [
    "incongruity", "wordplay", "timing", "absurdity", "surprise", 
    "irony", "dark_humor", "observational", "sarcasm", "narrative"
]

import threading

class HumorJudge:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.1, max_concurrent_calls: int = 1):
        self.model_name = model_name
        self.temperature = temperature
        self._api_semaphore = threading.Semaphore(max_concurrent_calls)
        self.llm = self._init_llm()

    def _init_llm(self):
        # Do NOT reload .env here — it overwrites GROQ_API_KEY and breaks key rotation
        
        base_url = os.getenv("OPENAI_BASE_URL", "")
        
        # Local VLLM / OpenAI-Compatible (e.g. running Llama 70B locally)
        if "localhost" in base_url or "127.0.0.1" in base_url or "vllm" in self.model_name:
            print(f"Connecting to Local LLM at {base_url} with model {self.model_name}")
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key="EMPTY", # VLLM often uses dummy key
                base_url=base_url
            )
            
        if "gpt" in self.model_name:
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL")
            )
        else:
            return ChatGroq(
                model=self.model_name,
                temperature=self.temperature,
                api_key=os.getenv("GROQ_API_KEY")
            )

    def _parse_json(self, content: str) -> Dict[str, Any]:
        """Robust JSON parsing from baseline."""
        try:
            return json.loads(content)
        except:
            import re
            match = re.search(r"\{[\s\S]*?\}", content)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
        return {"decision": "ERROR", "reasoning": content}

    def _invoke_with_retry(self, joke_a: str, joke_b: str, headline: str, max_retries: int = 3) -> Dict[str, Any]:
        """Robust invocation logic from baseline: retries, backoff, and key rotation."""
        feature_list_str = ", ".join(ALLOWED_FEATURES)
        user_msg = PAIRWISE_USER_TEMPLATE.format(
            original_prompt=headline,
            joke_a=joke_a,
            joke_b=joke_b,
            allowed_features=feature_list_str
        )
        messages = [
            SystemMessage(content=PAIRWISE_SYSTEM_PROMPT),
            HumanMessage(content=user_msg)
        ]

        for attempt in range(max_retries):
            try:
                # Rate limiting sleep (30/min for Groq as per baseline)
                # For Local VLLM, sleep is less critical but good for stability
                if hasattr(self.llm, "base_url") and "localhost" in str(self.llm.base_url):
                    pass # minimal sleep for local
                else:
                    time.sleep(2)
                
                with self._api_semaphore:    
                    response = self.llm.invoke(messages)
                result = self._parse_json(response.content)
                
                if "decision" in result:
                    decision = str(result["decision"]).upper().strip()
                    if decision in ["A", "B", "TIE"]:
                        result["decision"] = decision
                        return result
                
                raise ValueError(f"Invalid LLM response: {response.content}")
            except Exception as e:
                error_msg = str(e).lower()
                print(f"Judge attempt {attempt+1} failed: {str(e)[:200]}")
                
                # Key rotation on quota/rate limit/auth errors
                if any(kw in error_msg for kw in ["429", "rate_limit", "quota", "limit reached", "401", "authentication", "tpd"]):
                    print("Rotating API key due to rate/quota limit...")
                    self.llm = self._init_llm()
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** min(attempt, 4))  # cap backoff at 16s
        
        return {"decision": "ERROR", "reasoning": "Max retries exceeded"}

    def compare(self, joke_a: str, joke_b: str, headline: str, id_a: str, id_b: str) -> Dict[str, Any]:
        """
        Compares two jokes using single-call mode for speed.
        
        Note: We previously used double-blind (A vs B, then B vs A) to mitigate position bias,
        but this doubled runtime. For data curation purposes, single-call with random
        position shuffling is sufficient. See experiments.md Issue #2.
        
        Returns winner_id, loser_id, and confidence metrics.
        """
        import random
        
        # Randomly shuffle positions to mitigate position bias on average
        if random.random() < 0.5:
            # Original order: A first, B second
            res = self._invoke_with_retry(joke_a, joke_b, headline)
            first_id, second_id = id_a, id_b
        else:
            # Swapped order: B first, A second
            res = self._invoke_with_retry(joke_b, joke_a, headline)
            first_id, second_id = id_b, id_a
        
        if res["decision"] == "ERROR":
            return {"winner_id": None, "loser_id": None, "is_tie": True, "confidence": "ERROR", "reasoning": "Judge failed"}
        
        decision = res["decision"]
        
        if decision == "A":
            # First joke won
            return {
                "winner_id": first_id,
                "loser_id": second_id,
                "is_tie": False, 
                "confidence": "MEDIUM",  # Single-call = medium confidence
                "reasoning": res.get("reasoning", ""),
                "features": res.get("winner_features", [])
            }
        elif decision == "B":
            # Second joke won
            return {
                "winner_id": second_id, 
                "loser_id": first_id,
                "is_tie": False, 
                "confidence": "MEDIUM",
                "reasoning": res.get("reasoning", ""),
                "features": res.get("winner_features", [])
            }
        else:  # TIE
            return {
                "winner_id": None, 
                "loser_id": None,
                "is_tie": True, 
                "confidence": "MEDIUM", 
                "reasoning": res.get("reasoning", "Tie declared"),
                "features": []
            }

class LocalHuggingFaceJudge(HumorJudge):
    """
    In-process judge using local HuggingFace/transformers model.
    Loads model directly into GPU memory (efficient for single-process, 
    threaded execution). match_history is managed by caller.
    """
    def __init__(self, model_name: str, device: str = "cuda:0", temperature: float = 0.1):
        # Local execution is GPU-bound, so max concurrent calls = 1 is crucial
        super().__init__(model_name, temperature, max_concurrent_calls=1)
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _init_llm(self):
        # Local judge doesn't use LangChain LLM object, returns dummy
        return None
        
    def _load_model(self):
        print(f"Loading local model in-process: {self.model_name}")
        
        # Try Unsloth for Llama models (optimized)
        if "llama" in self.model_name.lower():
            try:
                print("Attempting to load via Unsloth (get_model_unsloth) for Llama optimization...")
                # Try import
                try:
                    from src.model_utils import get_model_unsloth
                except ImportError:
                    from ...model_utils import get_model_unsloth
                    
                # Attempt load
                self.model, self.tokenizer = get_model_unsloth(self.model_name, max_seq_length=4096, load_in_4bit=True)
                print("Successfully loaded via Unsloth!")
                return
            except Exception as e:
                print(f"Unsloth load skipped/failed ({e}), falling back to standard 4-bit...")

        try:
            # Try 4-bit load first
            self.model, self.tokenizer = get_model_4bit(self.model_name, alias="judge", device=self.device)
        except Exception as e:
            print(f"Failed 4-bit load, trying robust: {e}")
            self.model, self.tokenizer = get_model_robust(self.model_name, alias="judge", device=self.device)
            
    def _invoke_with_retry(self, joke_a: str, joke_b: str, headline: str, max_retries: int = 1) -> Dict[str, Any]:
        """
        Direct generation call to local model. No network retries needed.
        """
        feature_list_str = ", ".join(ALLOWED_FEATURES)
        user_msg = PAIRWISE_USER_TEMPLATE.format(
            original_prompt=headline,
            joke_a=joke_a,
            joke_b=joke_b,
            allowed_features=feature_list_str
        )
        
        # Simple chat template application
        messages = [
            {"role": "system", "content": PAIRWISE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ]
        
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{PAIRWISE_SYSTEM_PROMPT}\n\nUser: {user_msg}\n\nAssistant:"
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with self._api_semaphore:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    # pad_token_id=self.tokenizer.eos_token_id
                )
            
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return self._parse_json(generated_text)
