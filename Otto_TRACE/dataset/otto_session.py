from typing import Optional

import numpy as np

import torch
from torch.utils.data import Dataset

import json

class OttoDataSetSession(Dataset):
    """
    The first Dataset, before the TRACE Paper Cut Threshold of the Timestamps which in this case is 16 (THRESHOLD_TIMESTAMPS) 
    It Transforms the click aids in numbers 
    """
    def __init__(self, file_name : str, min_timestamps_per_sample : int = 32, max_samples : Optional[int] = None):
        self.session = []
        self.min_timestamps_per_sample = min_timestamps_per_sample

        self.event_map = {"clicks":1, "carts": 2, "orders": 3}

        for i, (session_id, eventstotal) in enumerate(self._extract_json(file_name)):
            aids, timestamps, events_type = [], [], []
            for event in eventstotal:
                aids.append(event["aid"])
                timestamps.append(event["ts"])
                events_type.append(self.event_map[event["type"]])
                
            # Jan: If the session does not have enough timestamps, drop it here
            if len(timestamps) < min_timestamps_per_sample:
                continue

            self.session.append({
                    "session_id": i,
                    "aid": np.array(aids),
                    "timestamps": np.array(timestamps),
                    'type': np.array(events_type)
                })
            
            # Jan: Use this to limit the amount of data during development
            if max_samples is not None and i >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.session)

    def _extract_json(self, filename: str):
        """
        Extracts the json that are used in the project (OttoDataset) "train.jsonl"
        - input -> filename : the name of the session
        - return all the sessions
        """
        with open(filename, "r") as f:
            for line in f:
                session = json.loads(line)
                yield session["session"], session["events"]

    def __getitem__(self, index):
        session = self.session[index]
                        
        aids = torch.tensor(session["aid"], dtype=torch.int64)
        
        timestamps = torch.tensor(session["timestamps"], dtype=torch.long)
        
        events_type = torch.tensor(session['type'], dtype=torch.int64)
        
        return {
            "session_id": torch.tensor(session["session_id"], dtype=torch.int64),
            "aid": aids,
            "timestamps": timestamps,
            "type": events_type
        }
    