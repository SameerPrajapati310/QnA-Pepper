from fastapi import FastAPI, Form, Request, Response, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import uvicorn
import os
import aiofiles
import json
import csv
from src.helper import llm_pipeline

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_file(request: Request, pdf_file: UploadFile = File(...), filename: str = Form(...)):
    base_folder = "static/docs/"

    # Ensure base folder exists
    if not os.path.isdir(base_folder):
        os.makedirs(base_folder)

    # Save uploaded file
    pdf_filename = os.path.join(base_folder, filename)
    async with aiofiles.open(pdf_filename, "wb") as f:
        content = await pdf_file.read()
        await f.write(content)

    response_data = {"msg": "success", "pdf_filename": pdf_filename}
    return Response(content=json.dumps(response_data), media_type="application/json")


def get_csv(file_path):

    # Generate questions and answers (dummy implementation)
    answer_generation_chain, ques_list = llm_pipeline(file_path)
    base_folder = "static/output/"
    if not os.path.isdir(base_folder):
        os.makedirs(base_folder)

    output_file = os.path.join(base_folder, "QA.csv")
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])  # Header row

        # Writing data
        for question in ques_list:
            print("Question: ", question)
            answer = answer_generation_chain.run(question) if answer_generation_chain else "No Answer"
            print("Answer:", answer)
            print("-------------------\n\n")
            csv_writer.writerow([question, answer])

    return output_file


@app.post("/analyze")
async def analyze_file(request: Request, pdf_filename: str = Form(...)):
    output_file = get_csv(pdf_filename)
    response_data = {"output_file": output_file}
    return Response(content=json.dumps(response_data), media_type="application/json")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
