import os
import json
import re
import fitz
import numpy as np
import cv2
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch
from tqdm import tqdm
import gc

# Store the original torch.load before overriding
_original_torch_load = torch.load  

def safe_torch_load(*args, **kwargs):
    """
    Safe wrapper for torch.load to avoid loading full checkpoints.
    
    Args:
    *args: Variable length argument list
    **kwargs: Arbitrary keyword arguments
    
    Returns:
    The loaded model
    """
    kwargs["weights_only"] = True  # Ensuring only weights are loaded
    return _original_torch_load(*args, **kwargs)  # Call the original function

# Assign the safe function only once
if torch.load!= safe_torch_load:  
    torch.load = safe_torch_load  

class PDFLoader:
    """
    Handles loading and validation of PDF documents.
    
    Attributes:
    pdf_path (str): Path to the PDF file
    doc (DocumentFile): The loaded PDF document
    """
    
    def __init__(self, pdf_path, max_pages=None):
        """
        Initializes the PDFLoader.
        
        Args:
        pdf_path (str): Path to the PDF file
        max_pages (int, optional): Maximum number of pages to load. Defaults to None.
        """
        self.pdf_path = pdf_path
        self.doc = None
        self._validate_pdf()
        self._load_pdf(max_pages)
    
    def _validate_pdf(self):
        """
        Validates if the PDF file exists.
        
        Raises:
        FileNotFoundError: If the PDF file does not exist
        """
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"File not found: {self.pdf_path}")
    
    def _load_pdf(self, max_pages=None):
        """
        Loads the PDF document with optional page limit.
        
        Args:
        max_pages (int, optional): Maximum number of pages to load. Defaults to None.
        """
        doc = DocumentFile.from_pdf(self.pdf_path)
        if max_pages is not None:
            doc = doc[:max_pages]
        self.doc = doc

class ChapterExtractor:
    """
    Extracts chapter title and last line from the PDF.
    
    Attributes:
    doc (DocumentFile): The loaded PDF document
    chapter_title (str): The extracted chapter title
    chapter (str): The extracted chapter
    """
    
    def __init__(self, doc):
        """
        Initializes the ChapterExtractor.
        
        Args:
        doc (DocumentFile): The loaded PDF document
        """
        self.doc = doc
        self.chapter_title = "Unknown"
        self.chapter = "Unknown"
        self._extract_chapter_title()
        tqdm.write(f"✅ Chapter title identified: {self.chapter_title}")
    
    def _extract_chapter_title(self):
        """
        Extracts chapter title and last line using OCR.
        """
        predictor = ocr_predictor(pretrained=True)
        ocr_data = predictor(self.doc[:1]).export()
        
        lines = [
            " ".join(word["value"] for word in line["words"]).strip()
            for block in ocr_data["pages"][0]["blocks"]
            for line in block["lines"]
        ]

        if len(lines) > 1:
            self.chapter_title = lines[1]
        self.chapter = re.sub(r'[0-9:]', '', lines[-1]).strip() if lines else "Unknown"

class TextExtractor:
    """
    Extracts text from PDF using OCR while preserving bullet points, lists, and formatting.
    
    Attributes:
    doc (DocumentFile): The loaded PDF document
    output_dir (str): The output directory
    chapter_title (str): The chapter title
    chapter (str): The chapter
    ocr_captions (dict): The OCR captions
    progress (tqdm): The progress bar
    """
    
    def __init__(self, doc, output_dir, chapter_title, chapter):
        """
        Initializes the TextExtractor.
        
        Args:
        doc (DocumentFile): The loaded PDF document
        output_dir (str): The output directory
        chapter_title (str): The chapter title
        chapter (str): The chapter
        """
        self.doc = doc
        self.output_dir = os.path.join(output_dir, "Text_Extraction", chapter)
        self.chapter_title = chapter_title
        self.chapter = chapter
        self.ocr_captions = {}
        self.progress = None  # For progress tracking
        self._process_text()
    
    def _process_text(self):
        """
        Processes OCR data while preserving lists, bullet points, and formatting.
        """
        predictor = ocr_predictor(pretrained=True)
        ocr_data = predictor(self.doc).export()
        chapter_pattern = re.compile(r'Chest:\s*(\d+)', re.IGNORECASE)
        current_chapter = self.chapter

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        with tqdm(total=len(ocr_data["pages"]), desc=f" Processing text for {self.chapter_title}", leave=False) as pbar:
            for i, page in enumerate(ocr_data["pages"], 1):
                page_folder = os.path.join(self.output_dir, f"Page_{i}")
                os.makedirs(page_folder, exist_ok=True)
                
                sorted_blocks = sorted(page["blocks"], key=lambda b: b["geometry"][0][1])
                self.ocr_captions[i] = []
                page_text = []

                for block in sorted_blocks:
                    block_text = []
                    for line in block["lines"]:
                        words = " ".join(word["value"] for word in line["words"]).strip()

                        # Detect and preserve bullet points and numbered lists
                        if re.match(r'^[•\-*]\s+', words):  
                            words = f"\n{words}"
                        elif re.match(r'^\d+\.\s+', words):  
                            words = f"\n{words}"
                        
                        block_text.append(words)
                    
                    block_text = "\n".join(block_text).strip()
                    if block_text:
                        page_text.append(block_text)
                        if len(block_text) > 20:
                            self.ocr_captions[i].append(block_text)

                        if match := chapter_pattern.match(block_text):
                            current_chapter = f"{self.chapter_title}_{match.group(1)}"

                formatted_text = self._clean_text("\n".join(page_text))

                # Save JSON and TXT files
                file_prefix = current_chapter
                with open(os.path.join(page_folder, f"{file_prefix}.json"), "w", encoding="utf-8") as f:
                    json.dump({"chapter": current_chapter, "content": formatted_text}, f, indent=4)
                
                with open(os.path.join(page_folder, f"{file_prefix}.txt"), "w", encoding="utf-8") as f:
                    f.write(formatted_text)
                
                pbar.update(1)
        tqdm.write(f"✅ Text extraction complete for {self.chapter_title}")
        del ocr_data
        gc.collect()
    
    def _clean_text(self, text):
        """
        Cleans text for better readability while fixing list formatting.
        
        Args:
        text (str): The text to clean
        
        Returns:
        str: The cleaned text
        """
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s{2,}','', text)
        text = re.sub(r'([•\-*]\s+)(\S)', r'\1 \2', text)
        text = re.sub(r'(\d+\.\s+)(\S)', r'\1 \2', text)
        return text.strip()

class ImageExtractor:
    """
    Extracts images from the PDF and generates metadata.
    
    Attributes:
    pdf_path (str): The path to the PDF file
    output_dir (str): The output directory
    ocr_captions (dict): The OCR captions
    chapter_title (str): The chapter title
    chapter (str): The chapter
    """
    
    def __init__(self, pdf_path, output_dir, ocr_captions, chapter_title, chapter):
        """
        Initializes the ImageExtractor.
        
        Args:
        pdf_path (str): The path to the PDF file
        output_dir (str): The output directory
        ocr_captions (dict): The OCR captions
        chapter_title (str): The chapter title
        chapter (str): The chapter
        """
        self.pdf_path = pdf_path
        self.output_dir = os.path.join(output_dir, "Images", chapter)
        self.ocr_captions = ocr_captions
        self.chapter_title = chapter_title
        self.chapter = chapter
        self._extract_images()
    
    def _extract_images(self):
        """
        Extracts images and generates metadata.
        """
        doc_img = fitz.open(self.pdf_path)
        max_pages = len(doc_img)
        
        # Create main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        with tqdm(total=max_pages, desc=f" Processing images for {self.chapter_title}", leave=False) as pbar:
            for page_num in range(max_pages):
                page = doc_img[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_cv = np.array(img)

                # Convert to grayscale and detect edges
                gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)

                # Detect contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                page_folder = os.path.join(self.output_dir, f"Page_{page_num + 1}")
                metadata_images = []
                image_counter = 1

                # Create page folder even if no images are found
                os.makedirs(page_folder, exist_ok=True)

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 250 and h > 180 and 0.54 < (w / h) < 4:
                        cropped_img = img_cv[y:y + h, x:x + w]
                        file_name = f"{self.chapter}_FIG_{image_counter}.jpg"
                        image_path = os.path.join(page_folder, file_name)
                        
                        # Ensure directory exists
                        os.makedirs(os.path.dirname(image_path), exist_ok=True)
                        
                        cv2.imwrite(image_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
                        
                        ocr_caption = " ".join(self.ocr_captions.get(page_num + 1, [])[:2])
                        metadata_images.append({
                            "file_name": file_name,
                            "section": f"{self.chapter_title} {self.chapter} Page: {page_num + 1}",
                            "ocr_caption": ocr_caption
                        })
                        
                        image_counter += 1

                if metadata_images:
                    with open(os.path.join(page_folder, "metadata.json"), "w", encoding="utf-8") as f:
                        json.dump(metadata_images, f, indent=4)
                
                pbar.update(1)
        doc_img.close()
        del doc_img
        gc.collect()
        tqdm.write(f"✅ Image extraction complete for {self.chapter_title}")

class MetadataHandler:
    """
    Handles metadata and master index creation, appending new chapters.
    
    Attributes:
    output_dir (str): The output directory
    chapter_info (dict): The chapter information
    progress (tqdm): The progress bar
    """
    
    def __init__(self, output_dir, chapter_info, progress=None):
        """
        Initializes the MetadataHandler.
        
        Args:
        output_dir (str): The output directory
        chapter_info (dict): The chapter information
        progress (tqdm, optional): The progress bar. Defaults to None.
        """
        self.output_dir = output_dir
        self.chapter_info = chapter_info
        self.progress = progress
        self._create_master_index()
    
    def _create_master_index(self):
        """
        Appends chapter info to Master_Index.json or creates a new file.
        """
        master_index_path = os.path.join(self.output_dir, "Master_Index.json")
        metadata = {
            "DOCUMENT_TITLE": "Core Radiology, Second Edition",
            "AUTHORS": ["Ellen X. Sun", "Junzi Shi", "Jacob C. Mandell"],
            "EDITION": "Second, Volume 1 and 2",
            "PUBLISHER": "Cambridge University Press",
            "PUBLICATION_DATE": ["2021", "2013"],
            "ISBN": {
                "Set": "9781108965910",
                "Volume 1": "9781108984447",
                "Volume 2": "9781108984454"
            }
        }

        if os.path.exists(master_index_path):
            with open(master_index_path, "r", encoding="utf-8") as f:
                master_index = json.load(f)
        else:
            master_index = {
                "Metadata": metadata,
                "Chapters": [],
                "Index_References_And_Contributors": "Index_References_And_Contributors/",
                "Text_Extraction": "Text_Extraction/",
                "Images": "Images/",
                "Tables": {
                    "CSV": "Tables/Extracted_Tables.csv",
                    "JSON": "Tables/Extracted_Tables.json"
                }
            }

        # Append new chapter
        master_index["Chapters"].append({
            "number": self.chapter_info["number"],
            "name": self.chapter_info["name"],
            "title": self.chapter_info["title"],
            "chapter": self.chapter_info["chapter"]
        })

        with open(master_index_path, "w", encoding="utf-8") as f:
            json.dump(master_index, f, indent=4)
        
        tqdm.write(f"✅ Metadata updated for {self.chapter_info['title']}")

class ContributorsReferencesExtractor:
    """
    Extracts contributors and references from specific PDF files.
    
    Attributes:
    output_dir (str): The output directory
    contributors_pdf (str): The path to the contributors PDF
    references_pdf (str): The path to the references PDF
    index_pdf (str): The path to the index PDF
    """
    
    def __init__(self, output_dir, contributors_pdf=None, references_pdf=None, index_pdf=None):
        """
        Initializes the ContributorsReferencesExtractor.
        
        Args:
        output_dir (str): The output directory
        contributors_pdf (str, optional): The path to the contributors PDF. Defaults to None.
        references_pdf (str, optional): The path to the references PDF. Defaults to None.
        index_pdf (str, optional): The path to the index PDF. Defaults to None.
        """
        self.output_dir = os.path.join(output_dir, "Index_References_And_Contributors")
        self.contributors_pdf = contributors_pdf
        self.references_pdf = references_pdf
        self.index_pdf = index_pdf
        
        if contributors_pdf or references_pdf or index_pdf:
            os.makedirs(self.output_dir, exist_ok=True)
            self._extract_content()

    def _extract_content(self):
        """
        Extracts and saves content if PDFs are provided.
        """
        with tqdm(total=3, desc="Processing auxiliary files", leave=False) as pbar:
            if self.contributors_pdf and os.path.exists(self.contributors_pdf):
                text = self._extract_text_from_pdf(self.contributors_pdf)
                with open(os.path.join(self.output_dir, "Contributors.txt"), "w", encoding="utf-8") as f:
                    f.write(text)
                pbar.update(1)
                tqdm.write(f"✅ Contributors extracted")
            else:
                pbar.update(1)

            if self.references_pdf and os.path.exists(self.references_pdf):
                text = self._extract_text_from_pdf(self.references_pdf)
                with open(os.path.join(self.output_dir, "References.txt"), "w", encoding="utf-8") as f:
                    f.write(text)
                pbar.update(1)
                tqdm.write(f"✅ References extracted")
            else:
                pbar.update(1)

            if self.index_pdf and os.path.exists(self.index_pdf):
                text = self._extract_text_from_pdf(self.index_pdf)
                with open(os.path.join(self.output_dir, "Index.txt"), "w", encoding="utf-8") as f:
                    f.write(text)
                pbar.update(1)
                tqdm.write(f"✅ Index extracted")
            else:
                pbar.update(1)
    
    def _extract_text_from_pdf(self, pdf_path):
        """
        Extracts text from a PDF file using PyMuPDF.
        
        Args:
        pdf_path (str): The path to the PDF file
        
        Returns:
        str: The extracted text
        """
        text = ""
        doc = fitz.open(pdf_path)
        try:
            for page in doc:
                text += page.get_text() + "\n"
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
        finally:
            doc.close()
        return text.strip()

class PDFProcessor:
    """
    Orchestrates the entire PDF processing pipeline.
    
    Attributes:
    pdf_path (str): The path to the PDF file
    output_dir (str): The output directory
    contributors_pdf (str): The path to the contributors PDF
    references_pdf (str): The path to the references PDF
    """
    
    def __init__(self, pdf_path, output_dir, contributors_pdf=None, references_pdf=None):
        """
        Initializes the PDFProcessor.
        
        Args:
        pdf_path (str): The path to the PDF file
        output_dir (str): The output directory
        contributors_pdf (str, optional): The path to the contributors PDF. Defaults to None.
        references_pdf (str, optional): The path to the references PDF. Defaults to None.
        """
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.contributors_pdf = contributors_pdf
        self.references_pdf = references_pdf
        self._process_pdf()
    
    def _process_pdf(self):
        try:
            # Load PDF
            with tqdm(desc=f"Loading {self.pdf_path}", leave=False) as pbar:
                pdf_loader = PDFLoader(self.pdf_path)
                pbar.update(1)
                tqdm.write(f"✅ PDF loaded: {self.pdf_path}")
            
            # Extract chapter title
            chapter_extractor = ChapterExtractor(pdf_loader.doc)
            
            # Extract chapter number and name from filename
            filename = os.path.basename(self.pdf_path)
            match = re.match(r"chapter_(\d+)_(.+?)\.pdf", filename)
            if match:
                chapter_num = match.group(1)
                chapter_name = match.group(2).replace('_','')
            else:
                chapter_num = "unknown"
                chapter_name = "unknown"
            
            # Extract text
            text_extractor = TextExtractor(
                pdf_loader.doc, 
                self.output_dir, 
                chapter_extractor.chapter_title, 
                chapter_extractor.chapter
            )
            
            # Extract images
            image_extractor = ImageExtractor(
                self.pdf_path, 
                self.output_dir, 
                text_extractor.ocr_captions, 
                chapter_extractor.chapter_title, 
                chapter_extractor.chapter
            )
            
            # Update metadata
            chapter_info = {
                "number": chapter_num,
                "name": chapter_name,
                "title": chapter_extractor.chapter_title,
                "chapter": chapter_extractor.chapter
            }
            MetadataHandler(self.output_dir, chapter_info)
            
            print(f"✅ Extraction Complete for Chapter {chapter_num} ({chapter_name})!")
            
            del pdf_loader
            del chapter_extractor
            del text_extractor
            del image_extractor
            gc.collect()
        except Exception as e:
            print(f"❌ Error during PDF processing: {str(e)}")
            raise

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    core_radiology_dir = os.path.join(script_dir, "core_radiology")
    output_dir = os.path.join(script_dir, "Extracted_Data")
    
    # Initialize progress display
    print("Starting PDF processing pipeline...")
    
    # Process index, contributors and references once
    contributors_path = os.path.join(core_radiology_dir, "contributors.pdf")
    references_path = os.path.join(core_radiology_dir, "references.pdf")
    index_path = os.path.join(core_radiology_dir, "index.pdf")
    
    if os.path.exists(contributors_path) or os.path.exists(references_path) or os.path.exists(index_path):
        with tqdm(total=1, desc="Processing auxiliary files", leave=False) as pbar:
            ContributorsReferencesExtractor(
                output_dir,
                contributors_pdf=contributors_path if os.path.exists(contributors_path) else None,
                references_pdf=references_path if os.path.exists(references_path) else None,
                index_pdf=index_path if os.path.exists(index_path) else None
            )
            pbar.update(1)
    
    # Process all chapter PDFs
    chapter_files = sorted([pdf_name for pdf_name in os.listdir(core_radiology_dir) 
                          if pdf_name.startswith("chapter_") and pdf_name.endswith(".pdf")],
                          key=lambda x: int(re.search(r'chapter_(\d+)_', x).group(1)))
    
    with tqdm(total=len(chapter_files), desc="Processing chapters", leave=True) as main_pbar:
        for pdf_name in chapter_files:
            pdf_path = os.path.join(core_radiology_dir, pdf_name)
            try:
                PDFProcessor(pdf_path, output_dir, None, None)
                main_pbar.update(1)
            except Exception as e:
                main_pbar.write(f"❌ Error processing {pdf_name}: {str(e)}")
                main_pbar.update(1)