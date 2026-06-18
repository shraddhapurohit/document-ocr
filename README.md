# Document Intelligence System

A Python-based document scanning system that automates text extraction from images using OpenCV and Pytesseract OCR.

## Features
- Multi-step image preprocessing pipeline
- Otsu thresholding for binarization
- Median blur for noise removal
- Batch processing of multiple documents
- Supports JPG, PNG, BMP, TIFF formats

## Technologies
Python, OpenCV, Pytesseract, Pillow

## Setup
pip install opencv-python pytesseract pillow

## Usage
1. Add images to `samples/` folder
2. Run: `python main.py`
3. Results saved in `output/` folder