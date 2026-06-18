import cv2
import pytesseract
from PIL import Image
import os

# --- IMPORTANT: Set your Tesseract path ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    """
    Load image and apply preprocessing to improve OCR accuracy.
    Each step has a reason — understand these for the interview.
    """
    # Step 1: Load image using OpenCV
    img = cv2.imread(image_path)

    # Step 2: Convert to grayscale
    # WHY: OCR works on text contrast, color adds noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply thresholding (binarization)
    # WHY: Makes text pure black, background pure white
    # OTSU method automatically finds best threshold value
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Noise removal using median blur
    # WHY: Removes small dots/specks that confuse OCR
    denoised = cv2.medianBlur(thresh, 3)

    return denoised

def extract_text(image_path):
    """
    Full pipeline: preprocess → extract text → return result
    """
    print(f"Processing: {image_path}")

    # Preprocess the image
    processed = preprocess_image(image_path)

    # Run OCR on processed image
    # config: psm 6 = assume uniform block of text
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed, config=custom_config)

    return text.strip()

def process_all_documents(input_folder, output_folder):
    """
    Batch process all images in a folder
    """
    os.makedirs(output_folder, exist_ok=True)

    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    files = [f for f in os.listdir(input_folder)
             if f.lower().endswith(supported_formats)]

    if not files:
        print("No images found in samples folder.")
        return

    print(f"\nFound {len(files)} document(s) to process\n")

    for filename in files:
        image_path = os.path.join(input_folder, filename)

        # Extract text
        extracted_text = extract_text(image_path)

        # Save output to text file
        output_filename = filename.rsplit('.', 1)[0] + '_extracted.txt'
        output_path = os.path.join(output_folder, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Source: {filename}\n")
            f.write("=" * 40 + "\n")
            f.write(extracted_text)

        print(f"✓ {filename} → {output_filename}")
        print(f"  Preview: {extracted_text[:80]}...\n")

    print(f"\nDone. Results saved in '{output_folder}' folder.")

# --- Run the program ---
if __name__ == "__main__":
    process_all_documents("samples", "output")