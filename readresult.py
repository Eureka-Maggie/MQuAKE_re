import json
import numpy as np
record_list = np.load('MQuAKE/record_list_3000_4_llama_13b.npz',allow_pickle=True)

record_forward_50 = []
for record in record_list['arr_0']:
    if record['id']>50:
        break
    else:
        record_forward_50.append(record)
print(len(record_forward_50))
result = []
id = 0
for record in record_forward_50:
    id_now = record['id']
    if id == id_now:
        continue
    else:
        result.append([record['gen_q'],record['retrieve_facts']])
        id = id_now
print(len(result))
