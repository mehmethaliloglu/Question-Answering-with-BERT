from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from urllib.request import urlopen
from bs4 import BeautifulSoup
import numpy as np
from nltk.tokenize import sent_tokenize,word_tokenize
import re
import ctypes  
from flask import Flask, redirect, url_for, request,render_template
app = Flask(__name__,template_folder='C:/Users/MONSTER/Desktop')

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    
    question = str(request.form['question'])
    url = str(request.form['subject'])
    
    source = urlopen(url).read()
    soup = BeautifulSoup(source,'lxml')
    
    text = ""
    
    for paragraph in soup.find_all('p'):
    
        text+=" "+str(paragraph.text)
        text=re.sub(r'\[.*?\]+', '', text)
        text=text.replace('\n', '')
        
    answer={}
    
    text=sent_tokenize(text)
    
    for i in text:
    
        input_id = tokenizer.encode(question, i)
    
        token=tokenizer.tokenize(tokenizer.decode(input_id))      
        
        
        segment_ids = list(tokenizer(question,i)["token_type_ids"])
        
            
        start_score, end_score = model(torch.tensor([input_id]),token_type_ids=torch.tensor([segment_ids]))
        
        answer_start = torch.argmax(start_score)        
        answer_end = torch.argmax(end_score)
        
        result=""
        for J in range(answer_start, answer_end + 1):
            if(token[J][0:2]=="##"):
               result+=token[J][2:]
        
            else:
                result +=" "+token[J]
        answer[result]=(float(start_score[0][answer_start]),float(end_score[0][answer_end]))
        
    lst_answers=[]
    for val in answer.values():
        lst_answers.append(np.mean(val))
        
    print(list(answer.values())[np.argmax(lst_answers)])
    
    if(np.mean(list(answer.values())[np.argmax(lst_answers)])>4.5):
    
        ret=list(answer.keys())[np.argmax(lst_answers)]
        
    else:
        ret="Not Found"
    return ret

if __name__ == '__main__':
   app.run(debug = True)

























