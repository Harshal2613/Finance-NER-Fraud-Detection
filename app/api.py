from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from google import genai
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name_summariser = "AventIQ-AI/t5-text-summarizer-for-financial-reports"
tokenizer_summariser = T5Tokenizer.from_pretrained(model_name_summariser)
model_summariser = T5ForConditionalGeneration.from_pretrained(model_name_summariser).to(device)

key = 'AIzaSyB838BgImTVD7L4VwgV_t-SgFpztrKexcs'
client = genai.Client(api_key=key)





current_dir = Path.cwd().as_posix()
print(current_dir)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend's origin for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the data model
class Item(BaseModel):
    text: str
    ner_res: dict

model_path = f"{current_dir}/app/models/finer"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
import json
# Set up NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=0)

@app.get('/')
async def get_status():
    return {'msg': "API is up and running"}

@app.get('/entities')
async def ent():
    label_map = model.config.id2label

    return json.dumps({label_id:label for label_id, label in label_map.items()}, indent=4)
# Display all supported entity labels
# print("Supported entity labels:")
# for label_id, label in label_map.items():
#     print(f"{label_id}: {label}")




# POST endpoint
@app.post("/ner")
async def create_item(item: Item):
    result= ner_pipeline(item.text)
    res = {
        i: [entity['word'], entity['entity_group'], float(entity['score']), entity['start'], entity['end']] for i, entity in enumerate(result)
        }
    return {
        "status": "success",
        "message": res
        # "item_data": item
    }

@app.post("/summary")
async def create_summary(item: Item):
    
    prompt= f"""Understand below text and NER entity groups and to which tokens they are classified and generate overall financially valuable summary.
Make sure to add information that was captured by NER and what entity groups were given, add them in summary report as well.
Text: {item.text}
NER model response: {item.ner_res}
"""
    response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt,
    )

    # print()
    return {
        "status":"success",
        "summary":response.text
    }

@app.post("/fraud_analysis")
async def create_analysis(item: Item):
    
    prompt = f"""Analyze the following financial text for potential fraud indicators.  
Text: `{item.text}`  
  
Instructions:  
1. Examine both narrative and tabular content for anomalies, red flags, or suspicious patterns (e.g., unusually large or round-number transactions, inconsistent or mismatched party names, off-balance-sheet items, vague or manipulative language, timeline conflicts, and regulatory non-compliance indicators).  
2. Consider contextual cues such as the use of ambiguous financial terminology, sudden changes in reporting, or shell entities.  
3. Highlight any signs of potential fraud clearly and concisely.
4. Limit the output to **3 to 4 sentences** summarizing key concerns.  
Output Format:
> Concise fraud analysis report stating key suspicious elements and a final conclusion on whether the text may contain potential fraud indicators.
"""
    response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt,
    )

    # print()
    return {
        "status":"success",
        "analysis":response.text
    }



if __name__=="__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
    # uvicorn.run(app, "0.0.0.0", port=8000)
