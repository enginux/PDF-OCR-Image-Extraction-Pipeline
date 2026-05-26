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
import itertools

# ====================== SAFE TORCH LOAD ======================
# Description: This section creates a safe wrapper for torch.load to prevent arbitrary code execution.
def safe_torch_load(*args, **kwargs):
    """
    Safe wrapper for torch.load to prevent arbitrary code execution.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Loaded PyTorch model.
    """
    kwargs["weights_only"] = True
    return _original_torch_load(*args, **kwargs)

_original_torch_load = torch.load

if torch.load!= safe_torch_load:
    torch.load = safe_torch_load

# ====================== PDF LOADER ======================
class PDFLoader:
    """
    Handles PDF loading and validation with error handling.

    Attributes:
        pdf_path (str): Path to the PDF file.
        doc (DocumentFile): Loaded PDF document.
    """
    def __init__(self, pdf_path: str, max_pages: int = None):
        """
        Initializes the PDF loader.

        Args:
            pdf_path (str): Path to the PDF file.
            max_pages (int, optional): Maximum number of pages to load. Defaults to None.
        """
        self.pdf_path = pdf_path
        self.doc = None
        self._validate_pdf()
        self._load_pdf(max_pages)

    def _validate_pdf(self):
        """
        Validates the PDF file existence and readability.
        """
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        if not os.access(self.pdf_path, os.R_OK):
            raise PermissionError(f"Cannot read PDF: {self.pdf_path}")

    def _load_pdf(self, max_pages: int = None):
        """
        Loads the PDF document.

        Args:
            max_pages (int, optional): Maximum number of pages to load. Defaults to None.
        """
        try:
            doc = DocumentFile.from_pdf(self.pdf_path)
            self.doc = doc[:max_pages] if max_pages else doc
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF: {str(e)}")

# ====================== CHAPTER HANDLER ======================
class ChapterHandler:
    """
    Manages chapter metadata and naming conventions.

    Attributes:
        pdf_path (str): Path to the PDF file.
        number (str): Chapter number.
        name (str): Chapter name.
        title (str): Chapter title.
        clean_name (str): Cleaned chapter name.
    """
    def __init__(self, pdf_path: str):
        """
        Initializes the chapter handler.

        Args:
            pdf_path (str): Path to the PDF file.
        """
        self.pdf_path = pdf_path
        self.number = self._extract_chapter_number()
        self.name = self._extract_chapter_name()
        self.title = self._get_title_from_file()
        self.clean_name = self._clean_chapter_name()

    def _extract_chapter_number(self) -> str:
        """
        Extracts the chapter number from the PDF file name.

        Returns:
            str: Chapter number.
        """
        match = re.search(r"chapter_(\d+)", os.path.basename(self.pdf_path))
        return match.group(1) if match else "unknown"

    def _extract_chapter_name(self) -> str:
        """
        Extracts the chapter name from the PDF file name.

        Returns:
            str: Chapter name.
        """
        match = re.search(r"chapter_\d+_(.+?)\.pdf", os.path.basename(self.pdf_path))
        return match.group(1).replace('_','').lower() if match else "unknown"

    def _get_title_from_file(self) -> str:
        """
        Retrieves the chapter title from a text file.

        Returns:
            str: Chapter title.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        txt_path = os.path.join(
            script_dir, "titles_sections", 
            f"chapter_{self.number}_{self.name.replace(' ', '_')}.txt"
        )
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.readline().strip().rstrip('.')
        return "Unknown"

    def _clean_chapter_name(self) -> str:
        """
        Cleans the chapter name by removing special characters.

        Returns:
            str: Cleaned chapter name.
        """
        clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', self.name).strip()
        return clean_name if clean_name else "unknown"

# ====================== OCR PROCESSOR ======================
class OCRProcessor:
    """
    Handles OCR processing and text extraction.

    Attributes:
        predictor (ocr_predictor): OCR predictor model.
    """
    def __init__(self):
        """
        Initializes the OCR processor.
        """
        self.predictor = ocr_predictor(pretrained=True)

    def extract_text(self, doc_page) -> list:
        """
        Extracts text from a document page using OCR.

        Args:
            doc_page: Document page.

        Returns:
            list: Extracted text blocks.
        """
        ocr_result = self.predictor([doc_page]).export()
        return self._structure_ocr_output(ocr_result)

    def _structure_ocr_output(self, ocr_data: dict) -> list:
        """
        Structures the OCR output into text blocks.

        Args:
            ocr_data (dict): OCR output data.

        Returns:
            list: Structured text blocks.
        """
        structured_data = []
        for block in ocr_data["pages"][0]["blocks"]:
            block_text = []
            for line in block["lines"]:
                words = " ".join(word["value"] for word in line["words"]).strip()
                if re.match(r'^[•\-*]\s+', words) or re.match(r'^\d+\.\s+', words):
                    words = f"\n{words}"
                block_text.append(words)
            block_content = "\n".join(block_text).strip()
            if block_content:
                structured_data.append(block_content)
        return structured_data

# ====================== TEXT EXTRACTOR ======================
class TextExtractor:
    """
    Extracts and saves text content with OCR captions.

    Attributes:
        doc (DocumentFile): Loaded PDF document.
        output_dir (str): Output directory.
        chapter_info (ChapterHandler): Chapter information.
        ocr_processor (OCRProcessor): OCR processor.
        ocr_captions (dict): OCR captions for each page.
    """
    def __init__(self, doc: DocumentFile, output_dir: str, chapter_info: ChapterHandler):
        """
        Initializes the text extractor.

        Args:
            doc (DocumentFile): Loaded PDF document.
            output_dir (str): Output directory.
            chapter_info (ChapterHandler): Chapter information.
        """
        self.doc = doc
        self.chapter = chapter_info
        self.output_dir = os.path.join(output_dir, "Text_Extraction", 
                                     f"Chapter_{self.chapter.number}_{self.chapter.clean_name}")
        self.ocr_processor = OCRProcessor()
        self.ocr_captions = {}
        self._process_document()

    def _process_document(self):
        """
        Processes the document and extracts text.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        with tqdm(total=len(self.doc), desc=f"Extracting text: {self.chapter.title}", leave=False) as pbar:
            for i, page in enumerate(self.doc, 1):
                page_dir = os.path.join(self.output_dir, f"Page_{i}")
                os.makedirs(page_dir, exist_ok=True)
                
                structured_text = self.ocr_processor.extract_text(page)
                # Capture OCR captions for images
                self.ocr_captions[i] = [block for block in structured_text if len(block.strip()) > 20]
                
                cleaned_text = self._clean_text("\n\n".join(structured_text))
                self._save_page_content(page_dir, i, cleaned_text)
                pbar.update(1)
        tqdm.write(f"✔️  Text extraction complete for {self.chapter.title}")

    def _save_page_content(self, page_dir: str, page_num: int, content: str):
        """
        Saves the page content to a file.

        Args:
            page_dir (str): Page directory.
            page_num (int): Page number.
            content (str): Page content.
        """
        base_name = f"Chapter_{self.chapter.number}_{self.chapter.clean_name}_Page_{page_num}"
        with open(os.path.join(page_dir, f"{base_name}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "chapter_number": self.chapter.number,
                "chapter_name": self.chapter.name,
                "chapter_title": self.chapter.title,
                "page_number": page_num,
                "content": content
            }, f, indent=4)
        with open(os.path.join(page_dir, f"{base_name}.txt"), "w", encoding="utf-8") as f:
            f.write(content)

    def _clean_text(self, text: str) -> str:
        """
        Cleans the extracted text.

        Args:
            text (str): Extracted text.

        Returns:
            str: Cleaned text.
        """
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s{2,}','', text) 
        text = re.sub(r'([•\-*]\s+)(\S)', r'\1\2', text)
        return text.strip()

# ====================== IMAGE EXTRACTOR ======================
class ImageExtractor:
    """
    Extracts images using contour detection and generates metadata.

    Attributes:
        pdf_path (str): Path to the PDF file.
        output_dir (str): Output directory.
        ocr_captions (dict): OCR captions for each page.
        chapter_title (str): Chapter title.
        chapter (str): Chapter name.
    """
    def __init__(self, pdf_path: str, output_dir: str, ocr_captions: dict, chapter_title: str, chapter: str):
        """
        Initializes the image extractor.

        Args:
            pdf_path (str): Path to the PDF file.
            output_dir (str): Output directory.
            ocr_captions (dict): OCR captions for each page.
            chapter_title (str): Chapter title.
            chapter (str): Chapter name.
        """
        self.pdf_path = pdf_path
        self.output_dir = os.path.join(output_dir, "Images", chapter)
        self.ocr_captions = ocr_captions
        self.chapter_title = chapter_title
        self.chapter = chapter
        self._extract_images()

    def _extract_images(self):
        """
        Extracts images from the PDF document.
        """
        doc_img = fitz.open(self.pdf_path)
        max_pages = len(doc_img)
        os.makedirs(self.output_dir, exist_ok=True)
        
        with tqdm(total=max_pages, desc=f" Processing images for {self.chapter_title}", leave=False) as pbar:
            for page_num in range(max_pages):
                page = doc_img[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_cv = np.array(img)

                gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                page_folder = os.path.join(self.output_dir, f"Page_{page_num + 1}")
                os.makedirs(page_folder, exist_ok=True)
                metadata_images = []
                image_counter = 1

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 250 and h > 180 and 0.54 < (w / h) < 4:
                        cropped_img = img_cv[y:y + h, x:x + w]
                        file_name = f"{self.chapter}_FIG_{image_counter}.jpg"
                        image_path = os.path.join(page_folder, file_name)
                        
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
        gc.collect()
        tqdm.write(f"✔️  Image extraction complete for {self.chapter_title}")

# ====================== METADATA HANDLER ======================
class MetadataHandler:
    """
    Manages master metadata index with chapter information.

    Attributes:
        output_dir (str): Output directory.
        chapter (ChapterHandler): Chapter information.
    """
    def __init__(self, output_dir: str, chapter_info: ChapterHandler):
        """
        Initializes the metadata handler.

        Args:
            output_dir (str): Output directory.
            chapter_info (ChapterHandler): Chapter information.
        """
        self.output_dir = output_dir
        self.chapter = chapter_info
        self.master_index_path = os.path.join(output_dir, "Master_Index.json")
        self._update_master_index()

    def _update_master_index(self):
        """
        Updates the master metadata index.
        """
        master_index = self._load_or_create_index()
        chapter_data = self._create_chapter_entry()
        
        existing_idx = next((i for i, ch in enumerate(master_index["Chapters"]) 
                           if ch["number"] == self.chapter.number), -1)
        if existing_idx >= 0:
            master_index["Chapters"][existing_idx] = chapter_data
        else: 
            master_index["Chapters"].append(chapter_data)
        
        master_index["Chapters"].sort(key=lambda x: int(x["number"]))
        with open(self.master_index_path, "w", encoding="utf-8") as f:
            json.dump(master_index, f, indent=4)
        tqdm.write(f"✔️  Metadata updated for Chapter {self.chapter.number} ({self.chapter.title})")

    def _load_or_create_index(self) -> dict:
        """
        Loads or creates the master metadata index.

        Returns:
            dict: Master metadata index.
        """
        if os.path.exists(self.master_index_path):
            with open(self.master_index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "Metadata": {
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
            },
            "Chapters": [],
            "Index_References_And_Contributors": "Index_References_And_Contributors/",
            "Text_Extraction": "Text_Extraction/",
            "Images": "Images/"
        }

    def _create_chapter_entry(self) -> dict:
        """
        Creates a chapter entry for the master metadata index.

        Returns:
            dict: Chapter entry.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        txt_path = os.path.join(
            script_dir, "titles_sections",
            f"chapter_{self.chapter.number}_{self.chapter.clean_name.replace(' ', '_')}.txt"
        )
        sections = []
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in itertools.islice(f, 1, None):
                    line = line.strip()
                    if not line:
                        continue
                    match = (re.match(r'^(.+?)\s*\.{4,}\s*(\d+)\s*$', line) or 
                            re.match(r'^(.+?)\s*\.{2,}\s*(\d+)\s*$', line) or 
                            re.match(r'^(.+?)\s+(\d+)\s*$', line))
                    if match:
                        sections.append({
                            "section": match.group(1).strip().rstrip('.'),
                            "section_page_number": match.group(2).strip()
                        })
        return {
            "number": self.chapter.number,
            "title": self.chapter.title,
            "chapter": self.chapter.clean_name.title(),
            "sections": sections
        }

# ====================== MAIN PROCESSOR ======================
class PDFProcessor:
    """
    Orchestrates the entire PDF processing pipeline.

    Attributes:
        pdf_path (str): Path to the PDF file.
        output_dir (str): Output directory.
    """
    def __init__(self, pdf_path: str, output_dir: str):
        """
        Initializes the PDF processor.

        Args:
            pdf_path (str): Path to the PDF file.
            output_dir (str): Output directory.
        """
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self._process_pdf()

    def _process_pdf(self):
        """
        Processes the PDF document.
        """
        try:
            chapter_info = ChapterHandler(self.pdf_path)
            tqdm.write(f"✔️  Processing Chapter {chapter_info.number}: {chapter_info.title}")
            
            pdf_loader = PDFLoader(self.pdf_path)
            tqdm.write(f"✔️  PDF loaded: {os.path.basename(self.pdf_path)}")
            
            # Process text extraction and collect OCR captions
            text_extractor = TextExtractor(pdf_loader.doc, self.output_dir, chapter_info)
            
            # Process image extraction with OCR data
            chapter_str = f"Chapter_{chapter_info.number}_{chapter_info.clean_name}"
            ImageExtractor(
                self.pdf_path,
                self.output_dir,
                text_extractor.ocr_captions,
                chapter_info.title,
                chapter_str
            )
            
            # Update metadata
            MetadataHandler(self.output_dir, chapter_info)
            
            print(f"✔️  Extraction complete for Chapter {chapter_info.number} ({chapter_info.title})")
            del pdf_loader
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(self.pdf_path)}: {str(e)}")
            raise

# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    core_radiology_dir = os.path.join(script_dir, "core_radiology")
    output_dir = os.path.join(script_dir, "Extracted_Data")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Text_Extraction"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Images"), exist_ok=True)
    
    print("Starting PDF processing pipeline...")
    
    chapter_files = sorted(
        [f for f in os.listdir(core_radiology_dir) if f.startswith("chapter_") and f.endswith(".pdf")],
        key=lambda x: int(re.search(r'chapter_(\d+)', x).group(1)))
    
    with tqdm(total=len(chapter_files), desc="Processing chapters") as pbar:
        for pdf_name in chapter_files:
            pdf_path = os.path.join(core_radiology_dir, pdf_name)
            try:
                PDFProcessor(pdf_path, output_dir)
            except Exception as e:
                tqdm.write(f"❌ Failed to process {pdf_name}: {str(e)}")
            finally:
                pbar.update(1)
    
    print("✔️  All chapters processed successfully!")