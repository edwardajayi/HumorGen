"""
Cognitive Synergy Framework - Model Utilities

Model loading, caching, and inference utilities.
"""

import os
import gc
import time
import torch
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Define base paths relative to repo root (parent of src/)
_REPO_ROOT = Path(__file__).resolve().parent.parent
SEMEVAL_DIR = _REPO_ROOT / "data" / "semeval"

# Optional imports (only needed if available)
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except Exception as e:
    HAS_UNSLOTH = False
    print(f"WARNING: Unsloth import failed: {type(e).__name__}: {e}")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# =============================================================================
# GLOBAL CACHE
# =============================================================================

model_cache: Dict[str, Tuple[Any, Any]] = {}
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", "/local/HF_CACHE")


# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================

def get_model_robust(
    model_name: str,
    alias: str,
    use_checkpoint: bool = True,
    device: str = "cuda:0"
) -> Tuple[Any, Any]:
    """
    Load a bfloat16 model with caching and checkpointing.
    
    Args:
        model_name: HuggingFace model name or path
        alias: Short name for caching
        use_checkpoint: Whether to use local checkpoint if available
        device: Target device
        
    Returns:
        Tuple of (tokenizer, model)
    """
    global model_cache
    
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers not installed")
    
    # Check cache
    if alias in model_cache:
        model_device = next(model_cache[alias][0].parameters()).device
        if str(model_device) == device:
            print(f"--- Using cached model: {alias} (on {device}) ---")
            return model_cache[alias]
        else:
            print(f"--- Discarding cached model '{alias}' (was on {model_device}). Reloading. ---")
            del model_cache[alias]
            gc.collect()
            torch.cuda.empty_cache()
    
    checkpoint_path = os.path.join(HF_CACHE_DIR, f"{alias}_bfloat16_checkpoint")
    
    # Try checkpoint first
    if use_checkpoint and os.path.isdir(checkpoint_path):
        print(f"--- Loading from local checkpoint: {checkpoint_path} ---")
        start_time = time.time()
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map=device
            )
            print(f"--- Loaded {alias} in {time.time() - start_time:.2f}s ---")
            model_cache[alias] = (model, tokenizer)
            return model, tokenizer
        except Exception as e:
            print(f"Warning: Checkpoint load failed: {e}. Downloading fresh.")
    
    # Load from HuggingFace
    print(f"--- Loading from HuggingFace: {model_name} ---")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE_DIR
    )
    model.to(device)
    print(f"--- Model {alias} loaded to {model.device} in {time.time() - start_time:.2f}s ---")
    
    # Save checkpoint
    if use_checkpoint:
        print(f"--- Saving checkpoint to: {checkpoint_path} ---")
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
    
    model_cache[alias] = (model, tokenizer)
    return model, tokenizer


def get_model_4bit(
    model_name: str,
    alias: str,
    device: str = "cuda:0"
) -> Tuple[Any, Any]:
    """
    Load a 4-bit quantized model.
    
    Args:
        model_name: HuggingFace model name or path
        alias: Short name for caching
        device: Target device
        
    Returns:
        Tuple of (tokenizer, model)
    """
    global model_cache
    
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers not installed")
    
    # Check cache
    if alias in model_cache:
        model_device = next(model_cache[alias][0].parameters()).device
        if str(model_device) == device:
            print(f"--- Using cached model: {alias} (on {device}) ---")
            return model_cache[alias]
        else:
            print(f"--- Discarding cached model '{alias}'. Reloading. ---")
            del model_cache[alias]
            gc.collect()
            torch.cuda.empty_cache()
    
    print(f"--- Loading 4-bit model: {model_name} ---")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device,
        cache_dir=HF_CACHE_DIR
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"--- Model {alias} loaded in {time.time() - start_time:.2f}s ---")
    model_cache[alias] = (model, tokenizer)
    return model, tokenizer


def get_model_unsloth(
    model_name: str,
    max_seq_length: int = 1024,
    load_in_4bit: bool = True
) -> Tuple[Any, Any]:
    """
    Load a model using Unsloth for fast inference.
    
    Args:
        model_name: Model name or path
        max_seq_length: Maximum sequence length
        load_in_4bit: Use 4-bit quantization
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if not HAS_UNSLOTH:
        raise ImportError("unsloth not installed. Install with: pip install unsloth")
    
    print(f"--- Loading model with Unsloth: {model_name} ---")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


# =============================================================================
# GPU MANAGEMENT
# =============================================================================

def cleanup_gpu(
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    alias: Optional[str] = None
) -> None:
    """
    Clean up GPU memory.
    
    Args:
        model: Model to delete
        tokenizer: Tokenizer to delete
        alias: Alias to remove from cache
    """
    global model_cache
    
    if alias and alias in model_cache:
        del model_cache[alias]
        print(f"Removed {alias} from cache")
    
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    
    gc.collect()
    torch.cuda.empty_cache()
    
    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"GPU Memory Free: {free_mem:.2f} GB")


def nuke_gpu() -> None:
    """Complete GPU memory cleanup."""
    global model_cache
    
    model_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()
    
    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"GPU nuked. Free: {free_mem:.2f} GB")


# =============================================================================
# KIMI K2 SUPPORT (via llama-server + OpenAI client)
# =============================================================================

class KimiK2Wrapper:
    """
    Wrapper for Kimi K2 that provides a HuggingFace-like interface.
    
    Uses llama-server (OpenAI-compatible) under the hood.
    This allows using Kimi K2 with existing generation code.
    
    Usage:
        model, tokenizer = get_model_kimi_k2()
        # Then use model.generate() or model(messages) as usual
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:8001/v1", model_name: str = "kimi-k2"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        
        self.client = OpenAI(base_url=base_url, api_key="sk-no-key-required")
        self.model_name = model_name
        self.device = "kimi-k2-server"  # Fake device for compatibility
        self.base_url = base_url
        self.max_retries = 5
        self.retry_delay = 2  # seconds
        
    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 256,
        temperature: float = 0.6,
        min_p: float = 0.01
    ) -> str:
        """
        Generate text from a prompt with automatic retry on disconnection.
        
        Handles:
        - Network disconnections
        - Server restarts
        - Temporary failures
        """
        import time
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body={"min_p": min_p}  # Kimi K2 recommended setting
                )
                return response.choices[0].message.content
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a retryable error
                retryable = any(x in error_msg for x in [
                    "connection", "timeout", "refused", "reset", 
                    "network", "unavailable", "500", "502", "503", "504"
                ])
                
                if retryable and attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"WARNING: Connection error (attempt {attempt + 1}/{self.max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise  # Re-raise if not retryable or max retries exceeded
    
    def __call__(self, messages: list, max_tokens: int = 256, temperature: float = 0.6) -> str:
        """Make the wrapper callable like a model."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    
    def to(self, device):
        """Dummy method for compatibility."""
        return self


class KimiK2Tokenizer:
    """
    Minimal tokenizer wrapper for Kimi K2 compatibility.
    
    Since Kimi K2 runs via API, we don't need full tokenization.
    This is just for interface compatibility.
    """
    
    def __init__(self):
        self.pad_token = "<|im_end|>"
        self.eos_token = "<|im_end|>"
        self.bos_token = "<|im_start|>"
    
    def apply_chat_template(self, messages: list, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        """Format messages for Kimi K2."""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"<|im_system|>system<|im_middle|>{content}<|im_end|>"
            elif role == "user":
                formatted += f"<|im_user|>user<|im_middle|>{content}<|im_end|>"
            elif role == "assistant":
                formatted += f"<|im_assistant|>assistant<|im_middle|>{content}<|im_end|>"
        
        if add_generation_prompt:
            formatted += "<|im_assistant|>assistant<|im_middle|>"
        
        return formatted
    
    def __call__(self, text, **kwargs):
        """Dummy tokenizer call for compatibility."""
        return {"input_ids": [], "attention_mask": []}


def get_model_kimi_k2(
    base_url: str = "http://127.0.0.1:8001/v1",
    model_name: str = "kimi-k2",
    test_connection: bool = True
) -> Tuple[Any, Any]:
    """
    Get Kimi K2 model wrapper.
    
    IMPORTANT: Requires llama-server running separately!
    
    Start the server with:
        ./llama.cpp/llama-server \\
            --model /path/to/Kimi-K2-Instruct-0905-UD-TQ1_0-00001-of-00005.gguf \\
            --alias "kimi-k2" \\
            -fa on \\
            --n-gpu-layers 999 \\
            -ot ".ffn_.*_exps.=CPU" \\
            --temp 0.6 \\
            --min-p 0.01 \\
            --ctx-size 16384 \\
            --port 8001 \\
            --jinja
    
    Args:
        base_url: llama-server URL (default: http://127.0.0.1:8001/v1)
        model_name: Model alias in llama-server
        test_connection: Whether to test the connection
        
    Returns:
        Tuple of (KimiK2Wrapper, KimiK2Tokenizer)
    """
    print(f"--- Connecting to Kimi K2 at {base_url} ---")
    
    model = KimiK2Wrapper(base_url=base_url, model_name=model_name)
    tokenizer = KimiK2Tokenizer()
    
    if test_connection:
        try:
            response = model.generate_text("Say 'OK' if you're working.", max_tokens=10)
            print(f"Kimi K2 connected! Test response: {response[:50]}...")
        except Exception as e:
            print(f"Connection failed: {e}")
            print("Make sure llama-server is running!")
            raise
    
    return model, tokenizer


def get_kimi_k2_server_command(
    model_path: str = "path/to/kimi-k2-model.gguf",
    llama_cpp_path: str = "path/to/llama.cpp",
    port: int = 8001
) -> str:
    """
    Get the command to start llama-server for Kimi K2.
    
    Returns the shell command that should be run in a separate terminal.
    """
    cmd = f"""
{llama_cpp_path}/llama-server \\
    --model {model_path} \\
    --alias "kimi-k2" \\
    -fa on \\
    --n-gpu-layers 999 \\
    -ot ".ffn_.*_exps.=CPU" \\
    --temp 0.6 \\
    --min-p 0.01 \\
    --ctx-size 16384 \\
    --port {port} \\
    --jinja
""".strip()
    return cmd
