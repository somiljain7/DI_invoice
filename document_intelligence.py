import os
from langchain.document_loaders import UnstructuredImageLoader
import pandas as pd
import re
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import os
import numpy as np
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil, os, json, re
from pdf2image import convert_from_path
from starlette.templating import Jinja2Templates

def extract_data_from_image(image_path: str):
    """Extract tables and key fields from an image using Unstructured."""
    loader = UnstructuredImageLoader(image_path)
    docs = loader.load()

    extracted_texts = []
    tables = []

    for doc in docs:
        category = doc.metadata.get("category", "").lower()
        if "table" in category:
            tables.append(doc.page_content)
        else:
            extracted_texts.append(doc.page_content)

    return extracted_texts, tables

def process_invoice_texts(texts):
    """Extract key fields from invoice text."""
    invoice_data = {}
    for text in texts:
        lines = text.split("\n")
        for line in lines:
            if ":" in line:
                key, value = map(str.strip, line.split(":", 1))
                invoice_data[key] = value
    return invoice_data

def process_document(image, processor, model, device):
    """Process a document image using the Donut model."""
    # prepare encoder inputs
    pixel_values = processor(image, return_tensors="pt").pixel_values
    
    # prepare decoder inputs
    task_prompt = "<s_rvlcdip>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
          
    # generate answer
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    
    # postprocess
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    
    return processor.token2json(sequence)


def model_load():
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    os.environ["OCR_AGENT"] = "unstructured.partition.utils.ocr_models.paddle_ocr.OCRAgentPaddle"
    return processor,model,device

def main(image_path):
    processor,model,device = model_load()
    image = Image.open(image_path).convert("RGB")
    result = process_document(image,processor,model,device)
    # return result
    # image_path = "test.jpg"  # Replace with your image path
    print(f"Processing image: {image_path}" , result)
    if result['class'] == 'invoice':
        # Extract data from imageS
        texts, tables = extract_data_from_image(image_path)
        # Process extracted texts for invoice fields
        invoice_fields = process_invoice_texts(texts)
        # Output results
        print("\nExtracted Invoice Fields:")
        os.makedirs('static', exist_ok=True)
        # Determine the base filename without extension
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        # Save invoice fields as JSON using the image file name
        os.makedirs('static', exist_ok=True)
        with open(f'static/{base_filename}_invoice_fields.json', 'w') as f:
            json.dump(invoice_fields, f, indent=2)

        # Process and save tables using the image file name as prefix
        if tables:
            print("\nExtracted Tables:")
            for i, table in enumerate(tables, 1):
            # Convert table to DataFrame (if structured)
                df = pd.DataFrame([row.split() for row in table.strip().split("\n") if row])
                df.to_csv(f"static/{base_filename}_table_{i}.csv", index=False)
        else:
            print("No tables found in the image.")
    return invoice_fields,tables


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
os.makedirs("static", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return HTMLResponse(content="""
    <!doctype html>
    <html lang="en">
        <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <title>Invoice Extraction</title>
        </head>
        <body>
        <div class="container mt-5">
            <h1>Upload Invoice PDF/Image</h1>
            <form action="/extract" method="post" enctype="multipart/form-data">
            <div class="mb-3">
                <label for="file" class="form-label">Choose File</label>
                <input class="form-control" type="file" id="file" name="file" accept=".pdf,image/*" required>
            </div>
            <button type="submit" class="btn btn-primary">Submit</button>
            </form>
        </div>
        </body>
    </html>
    """)

@app.post("/extract", response_class=HTMLResponse)
async def extract_invoice(file: UploadFile = File(...)):
    file_location = f"static/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # Process PDF files by converting to an image (using the first page)
    if file.filename.lower().endswith(".pdf"):
        try:
            images = convert_from_path(file_location)
            if not images:
                raise HTTPException(status_code=400, detail="Failed to convert PDF to image.")
            image = images[0]
        except Exception as e:
            raise HTTPException(status_code=400, detail="PDF conversion error.")
    else:
        try:
            image = Image.open(file_location).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image file.")
    
    processor, model, device = model_load()
    result = process_document(image, processor, model, device)
    
    if result.get("class") == "invoice":
        texts, tables = extract_data_from_image(file_location)
        invoice_fields = process_invoice_texts(texts)
        base_filename = os.path.splitext(os.path.basename(file.filename))[0]
        json_path = f"static/{base_filename}_invoice_fields.json"
        with open(json_path, "w") as f:
            json.dump(invoice_fields, f, indent=2)
        return HTMLResponse(content=f"""
        <!doctype html>
        <html lang="en">
            <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <title>Invoice Extraction Result</title>
            </head>
            <body>
            <div class="container mt-5">
                <h1>Invoice Extraction Result</h1>
                <pre>{json.dumps(invoice_fields, indent=2)}</pre>
                <a href="/" class="btn btn-secondary">Upload Another</a>
            </div>
            </body>
        </html>
        """)
    else:
        raise HTTPException(status_code=400, detail="Invoice not found in the uploaded document.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
