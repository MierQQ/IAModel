from fastapi import FastAPI
from fastapi.responses import FileResponse
from typing import List
from fastapi import Query
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from gtts import gTTS
from transformers import pipeline
import eclf
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

labels_name_to_ids = {'no_emotion': 0, 'joy': 1, 'sadness': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
labels_ids_to_name = ['no_emotion', 'joy', 'sadness', 'anger', 'fear', 'surprise']

device = torch.device('cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu')

clf = pipeline(
    task='sentiment-analysis',
    model='cointegrated/rubert-tiny2-cedr-emotion-detection',
    device = 0 if torch.cuda.is_available() else -1)

model_name_or_path = "ai-forever/rugpt3small_based_on_gpt2"
tokenizerGPT = GPT2Tokenizer.from_pretrained(model_name_or_path)
tokenizerGPT.add_special_tokens({"pad_token": "<pad>",
                                 "bos_token": "<startofstring>",
                                 "eos_token": "<endofstring>"})
tokenizerGPT.add_tokens(["<first>"])
tokenizerGPT.add_tokens(["<second>"])
tokenizerGPT.add_tokens(["<bot>:"])
modelGPT = GPT2LMHeadModel.from_pretrained(model_name_or_path)
modelGPT.to(device)
modelGPT.resize_token_embeddings(len(tokenizerGPT))
modelGPT.load_state_dict(torch.load('model_state_gpt__3.pt', map_location=device))
modelGPT.eval()

modelEClf = eclf.EmotionClf()
modelEClf.load_state_dict(torch.load("model_state_ce_19.pt", map_location=device))
modelEClf.to(device)
modelEClf.eval()


def classify(text: str):
    return clf(text, top_k=1)[0]


def generateTTS(text: str):
    tts = gTTS(text=text, lang='ru', slow=False)
    tts.save('out.mpeg')


def generateTextAndInfo(context: list):
    generated_text = infer(context)
    em1 = classify(context[-1])['label']
    em2 = classifyEClf(context[-1])
    em3 = classify(generated_text)['label']
    weights = [0.3, 0.3, 0.4]
    emotions = [0, 0, 0, 0, 0, 0]
    emotions[labels_name_to_ids[em1]] += weights[0]
    emotions[labels_name_to_ids[em2]] += weights[1]
    emotions[labels_name_to_ids[em3]] += weights[2]
    result = {}
    for x in labels_ids_to_name:
        result[x] = emotions[labels_name_to_ids[x]]
    return {"text": generated_text, "emotions": result}

@torch.no_grad()
def classifyEClf(text: str):
    tokens = modelEClf.tokenizer(
        text,
        max_length=50,
        padding='max_length',
        truncation=True,
        return_tensors="pt")
    preds = modelEClf(tokens['input_ids'].to(device), tokens['attention_mask'].to(device))
    s = nn.Softmax(dim=1)
    probs = s(preds * 10).detach().cpu().numpy()[0]
    probs_top_p = []
    sum = 0
    for x in probs:
        if x < 0.01:
            probs_top_p.append(0)
            continue
        sum += x
        probs_top_p.append(x)

    probs_norm = [x / sum for x in probs_top_p]
    return np.random.choice(labels_ids_to_name, 1, p=probs_norm)[0]


@torch.no_grad()
def infer(context: list):
    inp = "<startofstring> "
    for i, x in enumerate(context):
        if (i % 2) == 0:
            inp += "<first> "
        else:
            inp += "<second> "
        inp += x
    inp += " <bot>:"
    #print(inp)
    inp = tokenizerGPT(inp, return_tensors="pt")

    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    output = modelGPT.generate(X, attention_mask=a,
                               pad_token_id=tokenizerGPT.eos_token_id,
                               do_sample=True,
                               temperature=1.5,
                               num_beams=5,
                               top_k=20,
                               top_p=0.80,
                               repetition_penalty=1.5,
                               max_length=50,
                               early_stopping=True)

    output = tokenizerGPT.decode(output[0])
    return output.split('<bot>: ')[1].split(' <endofstring>')[0]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/getTextAndInfo")
async def getTextAndInfo(text: List[str] = Query(None)):
    return generateTextAndInfo(text)


@app.get("/getTTS/")
async def getTSS(text: str = ""):
    generateTTS(text)
    return FileResponse('out.mpeg')
