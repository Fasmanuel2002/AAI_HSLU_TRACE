import json
import random
from typing import Optional, Dict, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset

class OttoDataSetSession(Dataset):
    """
    The first Dataset, before the TRACE Paper Cut Threshold of the Timestamps which in this case is 32 (THRESHOLD_TIMESTAMPS) 
    It Transforms the click aids in numbers 
    """
    def __init__(self, file_name: str, min_timestamps_per_sample : int = 16, max_samples : Optional[int] = None):
        self.session = []
        self.min_timestamps_per_sample = min_timestamps_per_sample
        self.type_event_map = {"clicks":1, "carts": 2, "orders": 3}

        
        for i, (session_id, click_events_session) in enumerate(self._extract_json(file_name)):
            aids, timestamps, events_type = [], [], []
            for single_event_click in click_events_session:
                aids.append(single_event_click["aid"])
                timestamps.append(single_event_click["ts"])
                events_type.append(self.type_event_map[single_event_click["type"]])
                
            #Filter out short sessions to increase the median sequence length and retain more informative user interactions.    
            if len(timestamps) < min_timestamps_per_sample:
                continue
            
            self.session.append({
                    "session_id": i,
                    "aid": np.array(aids),
                    "timestamps": np.array(timestamps),
                    'type': np.array(events_type)
                })
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
    
class TraceOttoDataset(OttoDataSetSession):
    """
    TRACE dataset subclass of OttoDataSession handling input/target splitting, 
    sequence padding/truncation, and task definition for model prediction. 
    """
    def __init__(self, 
                file_name : str,
                input_seq_len : int,
                min_timestamps_per_sample : int = 32,
                max_samples : Optional[int] = None,
                split_min : float = 0.70,
                split_max : float = 0.80,
                ):
        super().__init__(file_name, min_timestamps_per_sample, max_samples)
        
        self.input_seq_len = input_seq_len
        self.split_min = split_min
        self.split_max = split_max
        self._split_range = random.uniform
        
        
    
    def __getitem__(self, index) -> Tuple[Dict, Dict]:
        session = self.session[index]
        
        #Splitting -> input part and target part
        input_part, target_part = self.__split_input_target__(session)
        
        #Padding input part
        input_part_padded = self.__pad_input_sequence__([input_part])[0]

        
        inputs = {
            "session_id" : torch.tensor(input_part_padded["session_id"], dtype=torch.int64),
            "aid" : torch.tensor(input_part_padded["aid"], dtype=torch.int64),
            "timestamps" : torch.tensor(input_part_padded["timestamps"], dtype=torch.long),
            "type": torch.tensor(input_part_padded["type"], dtype=torch.int64)
        }
        
        targets = {
            "ATC" : torch.tensor(self.__ATC_task_logit__(target_part), dtype=torch.int64),
            "SAT" : torch.tensor(self.__SAT__task_logit__(target_part), dtype=torch.int64),
            "MAP" : torch.tensor(self.__MAP__task_logit__(target_part), dtype=torch.int64)
        }
        
        return inputs, targets
    
    def __pad_input_sequence__(self, input) -> List:
        """
        Truncate sessions longer than input_seq_len and pad shorter ones to ensure a fixed input sequence length.
        """
        session_padded = []
        for session in input:
            if len(session["timestamps"]) >= self.input_seq_len:
                session_padded.append({
                    "session_id": session["session_id"],
                    "aid": session["aid"][-self.input_seq_len:],
                    "timestamps": session["timestamps"][-self.input_seq_len:],
                    "type": session["type"][-self.input_seq_len:]
                })

            else:
                session_padded.append(TraceOttoDataset.__padding__(self.input_seq_len, session))
        return session_padded
    
    def __split_input_target__(self, session : Dict) -> Tuple[Dict, Dict]:
        """
        Split input and target sequences following Section 2.1 of the TRACE paper to define model inputs and prediction targets.
        """
        n_events = int(session["timestamps"].shape[0])
        cutting = self._split_range(self.split_min, self.split_max)
        input_size = int(n_events * cutting)
        
        # ensure both parts non-empty
        input_size = max(1, min(n_events - 1, input_size))
        
        input_part = {
            "session_id": session["session_id"],
            "aid": session["aid"][:input_size],
            "timestamps": session["timestamps"][:input_size],
            "type": session["type"][:input_size]
        }
        target_part = {
            "session_id": session["session_id"],
            "aid": session["aid"][input_size:],
            "timestamps": session["timestamps"][input_size:],
            "type": session["type"][input_size:]
        }
        
        return input_part, target_part
    
    @staticmethod
    def _most_frequent(a_list : List) -> Tuple:
        """
        Counts the highest number of timestamps per product, used in Logit SAT4 (Seeing the same Aid 4 times)
        """
        dict = {}
        count, itm = 0, ''
        for item in reversed(a_list):
            dict[item] = dict.get(item, 0) + 1
            if dict[item] >= count :
                count, itm = dict[item], item
        return (count, itm)
        
    @staticmethod
    def __padding__(input_seq_len : int , session : Dict) -> Dict: 
        """
        Padding the input part of the dataset, as described in Section 2.2 of the TRACE paper.
        This step is required to ensure that all input sessions have the same fixed length, since user sessions naturally vary in size.
        """  
        padd_len = input_seq_len - len(session["timestamps"])
        zeros = np.zeros(padd_len, dtype=session["aid"].dtype)
        
        aid_padded = np.concatenate((zeros, session["aid"]))
        timestamps_padded = np.concatenate((zeros,session["timestamps"]))
        type_padded = np.concatenate((zeros, session["type"]))
        return {
            "session_id":session["session_id"],
            "aid": aid_padded,
            "timestamps": timestamps_padded,
            "type": type_padded
        }
    
    
    """
    Tasks of the the model
    """
    @staticmethod
    def __ATC_task_logit__(target_part : Dict) -> int:
        """
        ATC (Add-to-Cart frequency):
            1 if the user adds products to the cart at least 2 times
            during the session, 0 otherwise.
        """
        types = np.asarray(target_part["type"])
        atc_counts = int(np.sum(types == 2))
        return 1 if atc_counts >= 2 else 0
    
    @staticmethod
    def __SAT__task_logit__(target_part : Dict) -> int:
        """
        SAT (Repeated item views):
            1 if the user views the same article identifier (AID) at least
            4 times within a session, indicating strong browsing interest.
        """
        aids = target_part["aid"]
        if len(aids) == 0:
            return 0
        count, _ = TraceOttoDataset._most_frequent(aids) 
        return 1 if count >= 4 else 0
    
    
    @staticmethod 
    def __MAP__task_logit__(target_part : Dict) -> int:
        """
        MAP (Make a purchase in the Session)
            1 if the user purchase a article at least
            1 time within a session, 0 otherwise
        """
        types = np.asarray(target_part["type"])
        map_counts = int(np.sum(types == 3))
        return 1 if map_counts >= 1 else 0
    