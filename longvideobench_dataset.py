"""
LongVideoBench Dataset Integration
Adapted from longvideobench/LongVideoBench repository
For video LLM evaluation harness
"""

import json
import os
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset


class LongVideoBenchDataset(Dataset):
    """Dataset class for LongVideoBench evaluation"""
    
    def __init__(
        self,
        data_path: str,
        split: str = "val",
        max_frames: int = 32,
        frame_sample_rate: int = 1,
    ):
        self.data_path = data_path
        self.split = split
        self.max_frames = max_frames
        self.frame_sample_rate = frame_sample_rate
        
        # Load annotations
        self.annotations = self._load_annotations()
        
    def _load_annotations(self) -> List[Dict]:
        """Load dataset annotations"""
        ann_path = os.path.join(self.data_path, f"{self.split}.json")
        with open(ann_path, "r") as f:
            return json.load(f)
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single data sample"""
        item = self.annotations[idx]
        
        return {
            "video_id": item["video_id"],
            "video_path": item["video_path"],
            "question": item["question"],
            "options": item.get("options", []),
            "answer": item["answer"],
            "duration": item.get("duration", 0),
            "task_type": item.get("task_type", "qa"),
        }
    
    def get_video_metadata(self, video_id: str) -> Dict:
        """Get metadata for a specific video"""
        for item in self.annotations:
            if item["video_id"] == video_id:
                return item
        return {}


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching"""
    return {
        "video_ids": [item["video_id"] for item in batch],
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
        "video_paths": [item["video_path"] for item in batch],
    }