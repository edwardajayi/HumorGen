import os
import json
from typing import Set, Optional, Dict, Any, List, List

class CheckpointManager:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def load(self) -> Dict[str, Any]:
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"processed_ids": [], "leaderboard_stats": {}}

    def save(self, processed_ids: List[str], leaderboard_stats: Dict[str, Any]):
        data = {
            "processed_ids": processed_ids,
            "leaderboard_stats": leaderboard_stats
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)
