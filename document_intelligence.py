# pip install langchain unstructured[pdf, image] pandas pi_heif unstructured_inference , pdf2image,unstructured_paddleocr
#pip install "pdfminer.six<20221105"

import os
from langchain.document_loaders import UnstructuredImageLoader
import pandas as pd
import re
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


os.environ["OCR_AGENT"] = "unstructured.partition.utils.ocr_models.paddle_ocr.OCRAgentPaddle"
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

# Example usage
image_path = "test.jpg"  # Replace with your image path
texts, tables = extract_data_from_image(image_path)

# Process extracted texts for invoice fields
invoice_fields = process_invoice_texts(texts)

# Output results
print("\nExtracted Invoice Fields:")
for key, value in invoice_fields.items():
    print(f"{key}: {value}")

if tables:
    print("\nExtracted Tables:")
    for i, table in enumerate(tables, 1):
        print(f"Table {i}:\n{table}\n")
        # Convert table to DataFrame (if structured)
        df = pd.DataFrame([row.split() for row in table.split("\n")])
        print(df)
else:
    print("No tables found in the image.")



def process_document(image):
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

