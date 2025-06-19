from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()
templates = Jinja2Templates(directory="templates")

tokenizer = AutoTokenizer.from_pretrained("Tab1bit0/gpt2-oracle-1")
model = AutoModelForCausalLM.from_pretrained("Tab1bit0/gpt2-oracle-1")
model.eval()

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, question: str = Form(...)):
    prompt = f"Вопрос: {question}\nОтвет:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "question": question,
        "result": answer.split("Ответ:")[-1].strip()
    })
