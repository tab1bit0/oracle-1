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
    
    print("== Вопрос:", question)
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        print("== Начинаем генерацию")
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=40,  # уменьшено с 60
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        print("== Генерация завершена")

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    final_answer = answer.split("Ответ:")[-1].strip()
    
    print("== Ответ:", final_answer)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "question": question,
        "result": final_answer
    })
