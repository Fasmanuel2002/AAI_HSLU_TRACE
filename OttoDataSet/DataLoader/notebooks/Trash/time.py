
scaler_time_elapsed = StandardScaler()
scaler_time_between = StandardScaler()

#Categorical Embeddings
aids_session = []
event_session = []

#Time Embeddings
log_delta_elapsed = []
log_between_time = []

for sample in data_set_after_L:
    single_session = sample["inputs"]
    aids_session.extend(single_session["aid"].tolist())
    event_session.extend(single_session["type"].tolist())
    
    
    
    ts_last = single_session["timestamps"][-1]
    ts_first = single_session["timestamps"][0]
    
    log_delta_elapsed.append(log(1 +  (ts_last.item() - ts_first.item())))
    print(ts_last)
    
    deltas_this_session = []
    
    for j in range(len(single_session["timestamps"]) - 1):
        delta_between_times = (single_session["timestamps"][j+1].item() - single_session["timestamps"][j].item())
        deltas_this_session.append(log(1 + delta_between_times))
    log_between_time.append(deltas_this_session)




#AID Embeddings (Categorical, Embedding := 32)
aid_vocab = sorted(set(aids_session))
print(aids_session)
#aid_to_idx = {aid: i for i, aid in enumerate(aid_vocab)}
num_embeddings_aid = len(aid_vocab)

#Type Event Embeddings(Categorical, Embedding := 32)
type_event_vocab = sorted(set(event_session))
num_embeddings_event_type = len(type_event_vocab)

#Time Embeddings(Timestamps, Embedding := 1 + +1)
array_time_delta_elapsed = numpy.array(log_delta_elapsed).reshape(-1, 1)
standard_time_delta_elapsed = scaler_time_elapsed.fit_transform(array_time_delta_elapsed)
print(standard_time_delta_elapsed)
flat_time_between = [d for session_list in log_between_time for d in session_list]
array_time_between = numpy.array(flat_time_between).reshape(-1, 1)
standard_time_between = scaler_time_between.fit_transform(array_time_between)

number_session = 0
session_standard_time_between = []
for session_list in log_between_time:
    L = len(session_list)
    session_standard_time_between.append(standard_time_between[number_session : number_session + L].flatten().tolist())
    number_session += L
