import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import json


        
class OttoDataSetSession(Dataset):
    def __init__(self, session):
        self.session = session
        self.event_map = {"clicks":1, "carts": 2, "orders": 3}

    def __len__(self) -> int:
        return len(self.session)


    def __getitem__(self, index):
        session = self.session[index]
                 
        events = session["events"]
        
        aids = torch.tensor(events["aid"], dtype=torch.int64)
        
        timestamps = torch.tensor(events["timestamps"], dtype=torch.long)
        
        events_type = torch.tensor( [self.event_map[e] for e in events["events_type"]], dtype=torch.int64)
        return {
            "session_id": torch.tensor(session["session_id"], dtype=torch.int64),
            "aid": aids,
            "timestamps": timestamps,
            "type": events_type
        }
            
            
            

def extract_json(filename: str):
    with open(filename, "r") as f:
        for line in f:
            session = json.loads(line)
            yield session["session"], session["events"]
            
            
def custom_collate(batch):
    aids = [torch.tensor(d["aid"]) for d in batch]
    timestamps = [torch.tensor(d["timestamps"]) for d in batch]
    events_type = [torch.tensor(d["type"]) for d in batch]
    
    aids_padded = pad_sequence(aids, batch_first=True, padding_value=0)
    timestamps_padded = pad_sequence(timestamps, batch_first=True, padding_value=0)
    events_type_padded = pad_sequence(events_type, batch_first=True, padding_value=0)
    
    session_id = [d["session_id"] for d in batch]
    return {
        "aids" : aids_padded,
        "timestamps" : timestamps_padded,
        "events_type" : events_type_padded,
        "session_id" : torch.stack(session_id)
    }
    