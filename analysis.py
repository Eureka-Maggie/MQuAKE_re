import json
cor = 0

with open ('err_contradict_ans_13b.json','r',encoding='utf-8')as f:
    content = json.load(f)
for record in content:
    type = record['type']
    if(type==str(2) or type==str(3)):
        answer = record['answer']
        cor_ans = record['cor_answer']
        if(answer.lower()==cor_ans.lower()):
            cor+=1
        else :
            print(record)
print(cor)
