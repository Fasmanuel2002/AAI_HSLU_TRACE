import random
from typing import Optional, Tuple, Dict, List
import numpy as np
import torch
from dataset.otto_session import OttoDataSetSession


# Jan: Lots of things are suboptimal here
# Hard to read the code
# Using member variables to pass values between functions - just NO!
# Naming conventions are poor, hard to follow what means what

# Jan: Also the idea would be to have this dataset as a subclass of the Session dataset (as I've done below)
# The idea is not to build one torch Dataset and then get samples from it through __getitem__ into another dataset
# Some things are torch specific, so I get it is hard to immediately get right
# but some other things are quite pure software engineering / machine learning related things which were conceptually wrong
class TraceOttoDataSet(OttoDataSetSession):
    def __init__(self, file_name : str, input_seq_len : int, min_timestamps_per_sample : int = 32, max_samples : Optional[int] = None):
        super().__init__(file_name, min_timestamps_per_sample, max_samples)
        
        self.input_seq_len = input_seq_len
        
        input_part, target_part = self.__cut_input_target__()        
        self.inputs = self.__pad_input_sequence__(input_part)
        self.targets = target_part
         
        self.ATC = self.__ATC_task_logit__()
        self.SAT = self.__SAT__task_logit__()
        self.PD1 = self.__PD1_task_logit___()
        self.RA1 = self.__RA1_task_logit___()
    
    def __getitem__(self, index) -> Tuple[dict, dict]:
        inputs = {
            "session_id": torch.tensor(self.inputs[index]["session_id"], dtype=torch.int64),
            "aid": torch.tensor(self.inputs[index]["aid"], dtype=torch.int64),
            "timestamps": torch.tensor(self.inputs[index]["timestamps"], dtype=torch.long),
            "type": torch.tensor(self.inputs[index]["type"], dtype=torch.int64),
        }

        #4 classes for multi-learning task
        targets = {
            "ATC": self.ATC[index],
            "SAT": self.SAT[index],
            "PD1": self.PD1[index],
            "RA1": self.RA1[index] ,
            }

        
        return (inputs, targets)
        
     
        
    def _most_frequent(a_list):
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
    
    def __padding__(self, session) -> Dict:   
        padd_len = self.input_seq_len - len(session["timestamps"])
        zeros = np.zeros(padd_len)
        
        aid_padded = np.concatenate((session["aid"], zeros))
        timestamps_padded = np.concatenate((session["timestamps"], zeros))
        type_padded = np.concatenate((session["type"], zeros))
        return {
            "session_id":session["session_id"],
            "aid": aid_padded,
            "timestamps": timestamps_padded,
            "type": type_padded
        }
           
        
    def __cut_input_target__(self, min_value=0.80, max_value=0.90) -> Tuple[List, List]:
        inputs_part = []
        targets_part = []

        # Jan: This seems rather suboptimal, another iteration through the array of sessions

        for single_session in self.session:
            cutting_number = random.uniform(min_value, max_value)

            n_events = len(single_session["timestamps"])
            input_size = int(n_events * cutting_number)

            input_part = {
                "session_id": single_session["session_id"],
                "aid": single_session["aid"][:input_size],
                "timestamps": single_session["timestamps"][:input_size],
                "type": single_session["type"][:input_size]
            }

            target_part = {
                "session_id": single_session["session_id"],
                "aid": single_session["aid"][input_size:],
                "timestamps": single_session["timestamps"][input_size:],
                "type": single_session["type"][input_size:]
            }

            inputs_part.append(input_part)
            targets_part.append(target_part)

        return inputs_part, targets_part


            
    def __pad_input_sequence__(self, input) -> List:
        session_padded = []
        for session in input:
            if len(session["timestamps"]) >= self.input_seq_len:
                session_padded.append({
                "session_id": session["session_id"],
                "aid": session["aid"][:self.input_seq_len],
                "timestamps": session["timestamps"][:self.input_seq_len],
                "type": session["type"][:self.input_seq_len]
            })
            else: 
                session_padded.append(self.__padding__(session))
        return session_padded
    
    
    """
    Logits of the the model
    """
    def __ATC_task_logit__(self) -> List:
        """
        ATC (Add-to-Cart frequency):
            1 if the user adds products to the cart at least 3 times
            during the session, 0 otherwise.
        """
        logits_ATC = []
        for target_part in self.targets:
            atc_counts = sum(target==2 for target in target_part["type"])
            logits_ATC.append(1 if atc_counts>= 3 else 0)
        return logits_ATC
    
    def __SAT__task_logit__(self) -> List:
        """
        SAT (Repeated item views):
            1 if the user views the same article identifier (AID) at least
            4 times within a session, indicating strong browsing interest.

        """
        logits_SAT = []
        for session in self.targets:
            AidsS_repeated = []
            count = 0    
            for aids in session["aid"]:
                AidsS_repeated.append(aids)
                count, product = TraceOttoDataSet._most_frequent(AidsS_repeated) # type: ignore
            if count >= 4:
                logits_SAT.append(1)
            else:
                logits_SAT.append(0)
        return logits_SAT
    
    def __PD1_task_logit___(self) -> List:
        """
        PD1 (Purchase within 1 day):
            1 if the user completes a purchase within one day after the
            last observed event in the session, 0 otherwise.
        """
        logits_PD1 = []
        ONE_DAY = (86400 * 1000) 
        for session in self.targets:
            last_ts = session["timestamps"][-1]
            ordered_ts = session["timestamps"][session["type"] == 3]
            #Convert into int
            orders_ts = [ts for ts in ordered_ts]
            
            is_purchase = any([(order <= last_ts + ONE_DAY) for order in orders_ts] )
            if is_purchase == True:
                logits_PD1.append(1)
            else:
                logits_PD1.append(0)
            
        return logits_PD1
    
    def __RA1_task_logit___(self) -> List:
        """
        RA1 (Return to item within 1 day):
            1 if the same AID appears again in a different session within
            the next one-day window, 0 otherwise.
        """
        logits_RA1 = []
        ONE_DAY = (86400 * 1000) 
        for session in self.targets:
            first_aid = session["aid"][0]
            first_ts = session["timestamps"][0]
            indices = [index for index, aid in enumerate(session["aid"]) if aid == first_aid]
            other_ts_list = session["timestamps"][indices[1:]]

            is_aids = any((other_ts - first_ts) <= ONE_DAY for other_ts in other_ts_list)
            
            if is_aids == True:
                logits_RA1.append(1)
            else: 
                logits_RA1.append(0)
        return logits_RA1
            
           
