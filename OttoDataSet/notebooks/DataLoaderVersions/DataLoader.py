import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset
import json
import random
from torch.nn.utils.rnn import pad_sequence

THRESHOLD_TIMESTAMPS = 16
L_SEQUENCE_LENGHT = 48



def main():
    sessions_bf_threshold = []
    session_sample_lenght_after_threshold = []

    for i, (session_id, eventstotal) in enumerate(extract_json("../train.jsonl")):
        aids, timestamps, events_type = [], [], []
        for event in eventstotal:
            aids.append(event["aid"])
            timestamps.append(event["ts"])
            events_type.append(event["type"])
            
        sessions_bf_threshold.append({
                "session_id": i,
                "events": {
                "aid": aids,
                "timestamps": timestamps,
                "events_type": events_type    
                },
            })
        if i >= 200:
            break
        
    sessions_in_dataset = OttoDataSetSession(sessions_bf_threshold)
    print(f"Total len of the Sessions: {len(sessions_in_dataset)}")

    
    
    for i in range(len(sessions_in_dataset)):
        sample = sessions_in_dataset[i]
        if len(sample["timestamps"]) >= THRESHOLD_TIMESTAMPS:
            session_sample_lenght_after_threshold.append(sample)
        
    #DataSet    
    data_set_after_L = CutOttoDataSet(session_sample_lenght_after_threshold)
        
    #Training and testing batch size
    training_data_set, testing_data_set = data_set_after_L.__cut_training_testing__()
        
    #print(f"Training Batch: {training[0]})")
    #print(f"Testing Batch: {testing[0]}")
    
        
    #See how many lenghts are for the Aids in the raining batch and testing batch
    print(f"Training Batch Len: {len(training_data_set[0]["aid"])}")
    print(f"Testing Batch Len: {len(training_data_set[0]["aid"])}")
    
    print(len(data_set_after_L.__sequenceLenght__()[0]["timestamps"]))
    
    print("================================================ (Logits part) ===================================================")
    print("Logits for the ATC (Add to the Cart)")
    print(data_set_after_L.__ATC_task_logit__())
    
    print("Logits for SAT4(Seeing the same Aid 4 times)")
    print(data_set_after_L.__SAT__task_logit__())
    
    print("Logits for PD1(Make any Purchase within a day)")
    print(data_set_after_L.__PD1_task_logit___())
    
    print("Logits for RA1(Return to the same Aid in 1 days)")
    print(data_set_after_L.__RA1_task_logit___())
    
    train_loader = DataLoader(dataset=training_data_set, batch_size=64, collate_fn=custom_collate)
    
    testing_loader = DataLoader(dataset=testing_data_set, batch_size=64, collate_fn=custom_collate)
    
    
    for batch_training in train_loader:
        print(f"Shape of the Training Batch Aids:{batch_training["aids"].shape}, Shape of the Batch Ts:{batch_training["timestamps"].shape}, Shape of the batch Type:{batch_training["events_type"].shape}")
    
    print("============================================================================================================================================")
    for batch_testing in testing_loader:
        print(f"Shape of the Testing Batch Aids:{batch_testing["aids"].shape}, Shape of the Batch Ts:{batch_testing["timestamps"].shape}, Shape of the batch Type:{batch_testing["events_type"].shape}")
    
    
    
    
    
    
    
    
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
        
        
class CutOttoDataSet(OttoDataSetSession):
    def __init__(self, session):
        super().__init__(session)
     
        
        
    def __getitem__(self, index):
        session = self.session[index]

        return {
            "session_id": session["session_id"],
            "aid": session["aid"],
            "timestamps": session["timestamps"],
            "type": session["type"]
        }
     
        
    
    def __padding__(self, session):   
        padd_len = L_SEQUENCE_LENGHT - len(session["timestamps"])
        zeros = torch.zeros(padd_len, dtype=session["aid"].dtype)
        
        aid_padded = torch.concat((session["aid"], zeros))
        timestamps_padded = torch.concat((session["timestamps"], zeros))
        type_padded = torch.concat((session["type"], zeros))
        return {
            "session_id":session["session_id"],
            "aid": aid_padded,
            "timestamps": timestamps_padded,
            "type": type_padded
        }
           
        
    def __cut_training_testing__(self, min_value=0.80, max_value=0.90):
        training_batches = []
        testing_batches = []

        for single_session in self.session:
            cutting_number = random.uniform(min_value, max_value)

            
            n_events = len(single_session["timestamps"])
            train_size = int(n_events * cutting_number)

            train_part = {
                "session_id": single_session["session_id"],
                "aid": single_session["aid"][:train_size],
                "timestamps": single_session["timestamps"][:train_size],
                "type": single_session["type"][:train_size]
            }

            test_part = {
                "session_id": single_session["session_id"],
                "aid": single_session["aid"][train_size:],
                "timestamps": single_session["timestamps"][train_size:],
                "type": single_session["type"][train_size:]
            }

            training_batches.append(train_part)
            testing_batches.append(test_part)

        return training_batches, testing_batches


        
    def __sequenceLenght__(self):
        sessions_after_sequence_lenght = []
        for session in self.__cut_training_testing__()[0]:
            if len(session["timestamps"]) >= L_SEQUENCE_LENGHT:
                sessions_after_sequence_lenght.append({
                "session_id": session["session_id"],
                "aid": session["aid"][:L_SEQUENCE_LENGHT],
                "timestamps": session["timestamps"][:L_SEQUENCE_LENGHT],
                "type": session["type"][:L_SEQUENCE_LENGHT]
            })
            else: 
                sessions_after_sequence_lenght.append(self.__padding__(session))
        return sessions_after_sequence_lenght
    
    
    """
    Logis of the the model
    """
    def __ATC_task_logit__(self):
        """
        Logits for ATC(User added to the cart)
        """
        logits_ATC = []
       
        for session in self.__cut_training_testing__()[1]:
            if 3 in session["type"]:
                logits_ATC.append(1)
            else:
                logits_ATC.append(0)
        return logits_ATC
    
    def __SAT__task_logit__(self):
        """
        Logits for SAT4(Seeing the same Aid 4 times)
        """
        logits_SAT = []
        for session in self.__cut_training_testing__()[1]:
            AidsS_repeated = []
            count = 0    
            for aids in session["aid"]:
                AidsS_repeated.append(aids.item())
                count, product = most_frequent(AidsS_repeated)
            if count >= 4:
                logits_SAT.append(1)
            else:
                logits_SAT.append(0)
        return logits_SAT
    
    def __PD1_task_logit___(self):
        """
        Logits for PD1(Make any Purchase within a day)
        """
        logits_PD1 = []
        ONE_DAY = (86400 * 1000)
        for session in self.__cut_training_testing__()[1]:
            last_ts = session["timestamps"][-1].item()
            ordered_ts = session["timestamps"][session["type"] == 3]
            #Convert into int
            orders_ts = [ts.item() for ts in ordered_ts]
            
            is_purchase = any([(order <= last_ts + ONE_DAY) for order in orders_ts] )
            if is_purchase == True:
                logits_PD1.append(1)
            else:
                logits_PD1.append(0)
            
        return logits_PD1
    
    def __RA1_task_logit___(self):
        """
        Logits for RA1(Return to the same Aid in 1 days)
        """
        ONE_DAY = (86400 * 1000) 
        logits_RA1 = []
        for session in self.__cut_training_testing__()[1]:
            first_aid = session["aid"][0].item()
            first_ts = session["timestamps"][0].item()
            indices = [index for index, aid in enumerate(session["aid"]) if aid.item() == first_aid]
            other_ts_list = session["timestamps"][indices[1:]]

            is_aids = any((other_ts - first_ts) <= ONE_DAY for other_ts in other_ts_list)
            
            if is_aids == True:
                logits_RA1.append(1)
            else: 
                logits_RA1.append(0)
        return logits_RA1
            

   
    
def extract_json(filename: str):
    with open(filename, "r") as f:
        for line in f:
            session = json.loads(line)
            yield session["session"], session["events"]




def most_frequent(List):
    dict = {}
    count, itm = 0, ''
    for item in reversed(List):
        dict[item] = dict.get(item, 0) + 1
        if dict[item] >= count :
            count, itm = dict[item], item
    return(count, itm)


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
    

if __name__ == "__main__":
    main()