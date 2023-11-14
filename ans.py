import os
import json
import random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np

device = "cuda:4"
model_dir = '/home/tianxueyun/gptj'
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
err = []
for data in tqdm(j):
    tot+=1
    #print(data)
    question = data[0]['question']
    gold_retrieve_fact = data[0]['gold_retrieve_fact']
    retrieved_fact = data[0]['retrieved_fact']
    gold_generate = data[0]['gold_generate']
    answer_correct = data[0]['answer_correct']
    contradict = data[0]['contradict']
    ans_alias = data[0]['ans_alias']
    prompt = ans_prompt +'\n\nQuestion: '+ question + '\nGenerated answer: ' + gold_generate +'.\n'+ 'Retrieved fact: ' + retrieved_fact + '.'
    
    gen = get_ans(prompt)
    last_sent = gen.strip().split('\n')[-1]
    if last_sent.startswith('The answer is: '):
        pos = last_sent.find('The answer is: ')
        length = len('The answer is: ')
        ans = last_sent[pos+length:-1]
        if ans == answer_correct or ans in ans_alias:
            if(contradict=='contradict'):
                cor_c+=1
            else:
                cor_n+=1
        else:
            err.append(data)
        
print('total:',(cor_c+cor_n)/tot)
print('cor_c:',cor_c)
print('cor_n:',cor_n)

json_data = json.dumps(err)
with open('MQuAKE/err_ans_retrieve_gptj.json', 'w') as file:
    file.write(json_data)