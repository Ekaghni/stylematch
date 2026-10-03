import json, random, re, requests, pandas as pd
random.seed(7)
rows=[]
def add(text, label, group):
    text=re.sub(r"\s+"," ",str(text)).strip()
    w=text.split()
    if len(w)<40: return
    rows.append(dict(text=" ".join(w[:350]), label=label, group=group))

# HC3 (2023 ChatGPT vs humans on Q&A)
hc=[json.loads(l) for l in open("Hello-SimpleAI_HC3__all.jsonl",encoding="utf8")]
random.shuffle(hc)
nh=na=0
for d in hc:
    if nh<300 and d["human_answers"]: add(d["human_answers"][0],0,"hc3_human"); nh+=1
    if na<300 and d["chatgpt_answers"]: add(d["chatgpt_answers"][0],1,"hc3_chatgpt"); na+=1

# MAGE main: many generators and domains. label 1=human, 0=machine in MAGE.
m=pd.read_csv("yaful_MAGE__test.csv")
for lab,n,g in [(1,450,"mage_human"),(0,450,"mage_machine")]:
    for t in m[m.label==lab].sample(n,random_state=1).text: add(t,0 if lab==1 else 1,g)
# MAGE OOD: GPT-4 and GPT-4 paraphrased (evasion attack)
o=pd.read_csv("yaful_MAGE__test_ood_set_gpt.csv")
for t in o[o.src.str.endswith("_human")].sample(350,random_state=2).text: add(t,0,"gpt4set_human")
for t in o[o.src.str.endswith("_gpt4")].sample(350,random_state=2).text: add(t,1,"gpt4_raw")
p=pd.read_csv("yaful_MAGE__test_ood_set_gpt_para.csv")
for t in p[p.src.str.endswith("gpt4_para")].sample(350,random_state=2).text: add(t,1,"gpt4_paraphrased")

# Formal human text: Wikipedia intros
H={"User-Agent":"stylematch-bench/0.1 (research)"}
import time
def get(*a,**k):
    for i in range(6):
        try: return requests.get(*a,**k)
        except Exception as e: time.sleep(3)
    return None
n=0;fails=0
while n<150 and fails<12:
    resp=get("https://en.wikipedia.org/w/api.php",params=dict(action="query",generator="random",grnnamespace=0,grnlimit=20,prop="extracts",exintro=1,explaintext=1,exlimit=20,format="json"),headers=H,timeout=20)
    if resp is None: fails+=1; continue
    r=resp.json()
    for pg in r["query"]["pages"].values():
        before=len(rows); add(pg.get("extract",""),0,"wikipedia_human"); n+=len(rows)-before
# Formal human text: Gutenberg passages (older prose, a known false-positive trap)
for bid in [1342,84,11,2701,98,1661,174,345]:
    t=get(f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt",timeout=30)
    if t is None: continue
    t.encoding="utf-8"; t=t.text.replace("\r","")
    t=t[len(t)//10:-len(t)//10]
    paras=[re.sub(r"\s+"," ",x) for x in t.split("\n\n")]
    paras=[x for x in paras if len(x.split())>=60 and "�" not in x]
    for x in random.sample(paras,min(20,len(paras))): add(x,0,"gutenberg_human")
df=pd.DataFrame(rows)
df["id"]=range(len(df))
# 50/50 dev/test split within each group
df["split"]=df.groupby("group")["id"].transform(lambda s: ["dev" if i%2==0 else "test" for i in range(len(s))])
df.to_json("bench.jsonl",orient="records",lines=True,force_ascii=False)
print(df.groupby(["group","label"]).size()); print(df.text.str.split().str.len().describe())
