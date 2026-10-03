import sys, json, numpy as np, pandas as pd, torch, torch.nn as nn, time
from transformers import AutoTokenizer, AutoModel, AutoConfig, PreTrainedModel, AutoModelForSequenceClassification
from transformers.utils import logging; logging.set_verbosity_error()
name, repo = sys.argv[1], sys.argv[2]
ai_idx = int(sys.argv[3]) if len(sys.argv)>3 else None
df=pd.read_json("bench.jsonl",lines=True)
tok=AutoTokenizer.from_pretrained(repo)
if name=="desklib":
    class M(PreTrainedModel):
        config_class=AutoConfig
        def __init__(s,c):
            super().__init__(c); s.model=AutoModel.from_config(c); s.classifier=nn.Linear(c.hidden_size,1); s.post_init()
        def forward(s,input_ids,attention_mask):
            h=s.model(input_ids,attention_mask=attention_mask)[0]
            mk=attention_mask.unsqueeze(-1).expand(h.size()).float()
            return s.classifier((h*mk).sum(1)/mk.sum(1).clamp(min=1e-9))
    model=M.from_pretrained(repo)
else:
    model=AutoModelForSequenceClassification.from_pretrained(repo)
    print("id2label",model.config.id2label)
model=model.eval().cuda()
order=np.argsort(df.text.str.len().values); out=np.zeros(len(df)); t0=time.time()
with torch.no_grad():
    for i in range(0,len(df),16):
        idx=order[i:i+16]
        b=tok(df.text.iloc[idx].tolist(),return_tensors="pt",padding=True,truncation=True,max_length=512).to("cuda")
        if name=="desklib": p=torch.sigmoid(model(b["input_ids"],b["attention_mask"])).reshape(-1)
        else: p=model(**b).logits.softmax(-1)[:,ai_idx]
        out[idx]=p.float().cpu().numpy()
np.save(f"scores/{name}.npy",out); print(name,"done in",round(time.time()-t0),"s")
