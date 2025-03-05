
## Dependencies

The project requires the following dependencies:

- `langchain`
- `unstructured[pdf, image]`
- `pandas`
- `pi_heif`
- `unstructured_inference`
- `pdf2image`
- `unstructured_paddleocr`
- `pdfminer.six<20221105`
- `transformers[sentencepiece]`
- `uvicorn`
- `fastapi`

## Routes Documentation

### Home Route

**URL:** `/`

**Method:** `GET`

**Response Class:** `HTMLResponse`

**Description:** 
This route serves the home page of the application. It provides an HTML form for users to upload an invoice PDF or image file.

**Flow:**
1. The user accesses the home page.
2. The server responds with an HTML form for file upload.

**Status Codes:**
- `200 OK`: The home page is successfully loaded.

### Extract Invoice Route

**URL:** `/extract`

**Method:** `POST`

**Response Class:** `HTMLResponse`

**Description:** 
This route handles the file upload and processes the uploaded invoice PDF or image file to extract key fields and tables.

**Flow:**
1. The user uploads a file using the form on the home page.
2. The server saves the uploaded file to the `static` directory.
3. If the file is a PDF, it is converted to an image using the first page.
4. The image is processed using the Donut model to extract document information.
5. If the document is identified as an invoice, key fields and tables are extracted.
6. The extracted data is saved as JSON and CSV files in the `static` directory.
7. The server responds with an HTML page displaying the extracted invoice fields.

**Status Codes:**
- `200 OK`: The invoice is successfully processed and extracted fields are displayed.
- `400 Bad Request`: 
  - The uploaded file is invalid or cannot be processed.
  - The PDF conversion to image fails.
  - The document is not identified as an invoice.

### Static Files

**URL:** `/static/{path:path}`

**Description:** 
This route serves static files from the `static` directory, such as the uploaded files and the extracted JSON and CSV files.

**Status Codes:**
- `200 OK`: The requested static file is successfully served.
- `404 Not Found`: The requested static file does not exist.

## Functions

### `extract_data_from_image(image_path: str)`

Extracts tables and key fields from an image using Unstructured.

### `process_invoice_texts(texts)`

Extracts key fields from invoice text.

### `process_document(image, processor, model, device)`

Processes a document image using the Donut model.

### `model_load()`

Loads the Donut model and processor.

### `main(image_path)`

Main function to process the image and extract invoice fields and tables.
