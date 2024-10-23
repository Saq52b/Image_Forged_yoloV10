import fitz 
import pdfid
import os
import json
#---- 1 for real
#---- 0 for fake
def meta_data(file_path):
    pdf_document = fitz.open(file_path)
    links = []
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)  
        links.extend(page.get_links())  
    print(f'Links found: {links}')
    metadata = pdf_document.metadata
    print(f'Metadata: {metadata}')
    creation_date=metadata['creationDate']
    modification_date=metadata['modDate']
    pdf_document.close()
    if creation_date==modification_date:
        return 1
    else:
        return 0
    
def analyze_fields(file_path):
    doc = fitz.open(file_path)
    spans = []  # Initialize spans list once
    page = doc[0]
    blocks = page.get_text("dict", flags=0)["blocks"]
    
    for b in blocks:
        for l in b["lines"]:
            for s in l["spans"]:
                spans.append((s["bbox"], s["text"]))

    for span in spans:
        print(span)  
 

if __name__ == "__main__":
    file_path = 'D:\Saqib\IdentifyModifiedImage\Mycode\MLSource\yolov10-main\pdf\otempred.pdf'
    meta_data_check=meta_data(file_path)
    analyze_fields_check=analyze_fields(file_path)
    if meta_data_check==1:
        print('Real')
    else:
        print('Tempred')
