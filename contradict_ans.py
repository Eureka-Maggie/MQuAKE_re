import os
import json
import random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np

device = "cuda:3"
model_dir = '/home/tianxueyun/Llama-2-13b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype=torch.float16).to(device)

model.eval()

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_sent_embeddings(sents, contriever, tok, BSZ=32):    
    all_embs = []
    for i in tqdm(range(0, len(sents), BSZ)):
        sent_batch = sents[i:i+BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings)
    all_embs = torch.vstack(all_embs)
    return all_embs

def retrieve_facts(query, fact_embs, contriever, tok, k=1):
    inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs['attention_mask'])
    sim = (query_emb @ fact_embs.T)[0]
    knn = sim.topk(k, largest=True)
    return knn.indices

with open('/home/tianxueyun/MQuAKE/datasets/MQuAKE-CF-3k.json', 'r') as f:
    dataset = json.load(f)
new_facts = set()
for d in dataset:
    for r in d["requested_rewrite"]:
        new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
new_facts = list(new_facts)

contriever = AutoModel.from_pretrained("/home/tianxueyun/MQuAKE/contriever").to(device)
tokenizer_con = AutoTokenizer.from_pretrained("/home/tianxueyun/MQuAKE/contriever")
embs = get_sent_embeddings(new_facts, contriever, tokenizer_con)

with open('/home/tianxueyun/MQuAKE/prompts/contradict-prompt.txt', 'r') as f:
    contradict_prompt = f.read()

with open('/home/tianxueyun/MQuAKE/prompts/contradict-ans-prompt.txt', 'r') as f:
    contradict_ans_prompt = f.read()

with open('/home/tianxueyun/MQuAKE/prompts/ans-prompt.txt', 'r') as f:
    ans_prompt = f.read()

with open('/home/tianxueyun/MQuAKE/prompts/choose.txt', 'r') as f:
    choose_prompt = f.read()

with open('/home/tianxueyun/MQuAKE/prompts/choose_not.txt', 'r') as f:
    choose_not_prompt = f.read()

def get_ans(prompt):
    start = len(prompt)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length = input_ids.size()[1]+80,num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    #print(generated_text)
    rest = generated_text[start:]
    fa_index = rest.find('\n\nQuestion:') #找final_ans
    rf_index = rest.find('Retrieved fact:')
    if (fa_index > rf_index and rf_index!=-1 ) or fa_index == -1:
        index = rf_index
    else:
        index = fa_index

    generate_q_a = rest[:index]
    #print(generate_q_a)
    return generate_q_a

#读取contradict数据集
with open('MQuAKE/contradict.json', 'r') as file:
    j = json.load(file)
cor_c = 0 
cor_n = 0 
tot = 0
record = []
list_0 = 0 
list_1 = 0 
list_2 = 0 
list_3 = 0 
list_4 = 0 
list_5 = 0 
err = []
for data in tqdm(j):
    tot+=1
    if(tot!=2438):continue
    #print(data)
    question = data[0]['question']
    gold_retrieve_fact = data[0]['gold_retrieve_fact']
    retrieved_fact = data[0]['retrieved_fact']
    gold_generate = data[0]['gold_generate']
    answer_correct = data[0]['answer_correct']
    contradict = data[0]['contradict']
    ans_alias = data[0]['ans_alias']
    prompt = contradict_prompt +'\n\nQuestion: '+ question + '\nGenerated answer: ' + gold_generate +'.\n'+ 'Retrieved fact: ' + gold_retrieve_fact + '.'
    
    gen = get_ans(prompt)
    last_sent = gen.strip().split('\n')[-1]
    if last_sent.startswith('Retrieved fact'):
        if len(last_sent[15:-22])==10: #判断为contradict
            if contradict == 'not contradict':
                err.append({'id':tot,'type':'n2c'})
            prompt = choose_prompt +'\n\nQuestion: '+ question +'.\n'+ 'Retrieved fact: ' + gold_retrieve_fact + '.'
            gen = get_ans(prompt)
            last_sent = gen.strip().split('\n')[-1]
            length = len('The answer is: ')
            pos = last_sent.find('The answer is: ')
            ans = last_sent[pos+length:-1]
            if ans == answer_correct or ans in ans_alias:
                if(contradict=='contradict'):
                    cor_c+=1
                else:
                    record.append({'id':tot,'type':'0','answer':'correct','cor_answer':'correct'})
                    list_0+=1
                    # 0
            else:
                if(contradict=='contradict'):
                    record.append({'id':tot,'type':'2','answer':ans,'cor_answer':answer_correct})
                    list_2+=1
                    #2
                else:
                    record.append({'id':tot,'type':'4','answer':ans,'cor_answer':answer_correct})


        else: # 判断为not contradict
            if contradict == 'contradict':
                err.append({'id':tot,'type':'c2n'})
            prompt = choose_not_prompt +'\n\nQuestion: '+ question + '\nGenerated answer: ' + gold_generate +'.'
            gen = get_ans(prompt)
            last_sent = gen.strip().split('\n')[-1]
            length = len('The answer is: ')
            pos = last_sent.find('The answer is: ')
            ans = last_sent[pos+length:-1]
            if ans == answer_correct or ans in ans_alias:
                if(contradict=='not contradict'):
                    cor_n+=1
                else:
                    #print('not_contradict+',contradict,answer_correct,"+",last_sent)
                    record.append({'id':tot,'type':'1','answer':'correct','cor_answer':'correct'})
                    list_1+=1
                    # 1
            else:
                if(contradict=='not contradict'):
                    record.append({'id':tot,'type':'3','answer':ans,'cor_answer':answer_correct})
                    list_3+=1
                    # 3
                else:
                    record.append({'id':tot,'type':'5','answer':ans,'cor_answer':answer_correct})
                    list_5+=1

        if(tot%100==0):
            print(tot,", ",cor_c,' ',cor_n)
            
print('total:',(cor_c+cor_n)/tot)
print('cor_c:',cor_c)
print('cor_n:',cor_n)
print('0:',list_0,' 1:',list_1,' 2:',list_2,' 3:',list_3,' 4:',list_4,' 5:',list_5)
print('err_contradict:',len(err))
json_data = json.dumps(record)
with open('err_contradict_ans_13b.json', 'w') as file:
    file.write(json_data)

json_data = json.dumps(err)
with open('err_contradict_13b.json', 'w') as file:
    file.write(json_data)

# 0:答案对了，判断的是contradict，实际是not contradict
# 1:答案对了，判断的是not contradict，实际是contradict
# 2:答案错了，判断是contradict是对的
# 3:答案错了，判断是not contradict是对的
# 4:答案错了，判断contradict，实际是not contradict
# 5:答案错了，判断not contradict，实际是contradict