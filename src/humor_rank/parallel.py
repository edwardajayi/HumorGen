import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Callable

def run_parallel(items: List[Any], func: Callable, max_workers: int = 32):
    """
    Generic parallel runner using ProcessPoolExecutor.
    """
    actual_workers = min(max_workers, multiprocessing.cpu_count())
    print(f"Starting parallel execution with {actual_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        results = list(executor.map(func, items))
        
    return results
