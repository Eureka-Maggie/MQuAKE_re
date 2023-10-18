#使用llama-2-13B运行的循环为4的实验
import os
import json
import random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np

device = "cuda:7"
model_dir = 'Llama-2-13b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype=torch.float16).to(device)

model.eval()

with open('/home/tianxueyun/MQuAKE/prompts/MeLLo-prompt.txt', 'r') as f:
    task_prompt = f.read()

def call_gpt(cur_prompt, start):
    # 将输入文本编码为模型输入
    input_ids = tokenizer.encode(cur_prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=input_ids.size()[1]+100,num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    rest = generated_text[start:]
    fa_index = rest.find('\n\nQuestion:')#找final_ans
    rf_index = rest.find('Retrieved fact:')
    
    if (fa_index > rf_index and rf_index!=-1 ) or fa_index == -1:
        index = rf_index
    else:
        index = fa_index

    generate_q_a = rest[:index]
    #print(generate_q_a)
    return generate_q_a
#==============================for contriever====================================
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

contriever = AutoModel.from_pretrained("/home/tianxueyun/MQuAKE/contriever").to(device)
tokenizer_con = AutoTokenizer.from_pretrained("/home/tianxueyun/MQuAKE/contriever")

with open('/home/tianxueyun/MQuAKE/datasets/MQuAKE-CF-3k.json', 'r') as f:
    dataset = json.load(f)
new_facts = set()
for d in dataset:
    for r in d["requested_rewrite"]:
        new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
new_facts = list(new_facts)

embs = get_sent_embeddings(new_facts, contriever, tokenizer_con)

T = 10

cor = 0
tot = 0
start = len(task_prompt)

model.config.pad_token_id = model.config.eos_token_id

record_list = []
cor_list = []
for d in tqdm(dataset):
    #print(d)
    real_edit = []
    tot += 1
    hop = len(d["new_single_hops"])
    real_hop = []
    #print(hop)
    #用于记录该问题应该retrieve哪些edit fact
    for r in d["requested_rewrite"]:
        real_edit.append(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
    for h in d['new_single_hops']:
        real_hop.append(h['question'])
    cnt = 0
    for q in d["questions"]:
        cnt+=1
        retrieved_facts = []
        found_ans = False
        prompt = task_prompt + "\n\nQustion: " + q
        flag = 0
        gen_q = []
        i = 0
        for i in range(4):
            # prompt the model to generate a subquestion and a tentative answer
            start = len(prompt)
            gen = call_gpt(prompt, start)
            gen_q.append(gen)
            last_sent = gen.strip().split('\n')[-1]
            
            # if final answer is there, get the answer and exit
            if last_sent.startswith('Final answer: '):
                found_ans = True
                ans = last_sent[len("Final answer: "):]
                break
            
            # otherwise, extract the generated subquestion
            if len(gen.strip().split('\n')) < 2:
                record = {'id':tot,'hop':hop,'question':q,'real_edit':real_edit,'retrieve_facts':retrieved_facts,'real_hop:':real_hop,'gen_q':gen_q,'answer':"failed_1"}
                record_list.append(record)
                flag = 1
                break # failed case
            subquestion = gen.strip().split('\n')[-2]
            if not subquestion.startswith('Subquestion: '):#生成有问题
                record = {'id':tot,'hop':hop,'question':q,'real_edit':real_edit,'retrieve_facts':retrieved_facts,'real_hop:':real_hop,'gen_q':gen_q,'answer':"failed_2"}
                record_list.append(record)
                flag = 1
                break # failed case
            subquestion = subquestion[len("Subquestion: "):]
            
            # retrieve an edited fact using the generated subquestion
            fact_ids = retrieve_facts(subquestion, embs, contriever, tokenizer_con)
            fact_sent = new_facts[fact_ids[0]]
            retrieved_facts.append(fact_sent)
            
            # put the retrieved fact at the end of the prompt, the model self-checks if it contradicts
            prompt = prompt + gen + 'Retrieved fact: ' + fact_sent + '.'
            
        prompt = prompt + gen
        
        if not found_ans:
            if flag == 0:
                record = {'id':tot,'hop':hop,'question':q,'real_edit':real_edit,'retrieve_facts':retrieved_facts,'real_hop:':real_hop,'gen_q':gen_q,'answer':"no_final_ans"}
                record_list.append(record)
            continue
        # if the answer is correct
        if ans == d["new_answer"] or ans in d["new_answer_alias"]:
            cor += 1
            cor_record = {'id':tot,'hop':hop,'used_hop':i,'question':q,'real_edit':real_edit,'retrieve_facts':retrieved_facts,'real_hop:':real_hop,'gen_q':gen_q}
            cor_list.append(cor_record)
            break
        else:
            record = {'id':tot,'hop':hop,'question':q,'real_edit':real_edit,'retrieve_facts':retrieved_facts,'real_hop:':real_hop,'gen_q':gen_q,'answer':"not_correct_ans"}
            record_list.append(record)
    if tot %1000 == 0:
        np.savez('cor_list_'+str(tot)+'_4_llama_13b',cor_list)
        np.savez('record_list_'+str(tot)+'_4_llama_13b',record_list)

print(f'Multi-hop acc = {cor / tot} ({cor} / {tot})')

np.savez('cor_list_4_llama_13b',cor_list)
np.savez('record_list_4_llama_13b',record_list)