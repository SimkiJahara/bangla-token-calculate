import os
import easyocr
import cv2
import statistics
try:
    from bnlp import BasicTokenizer
    tokenizer = BasicTokenizer()
    use_bnlp = True
except ImportError:
    use_bnlp = False
    print("Warning: bnlp_toolkit not available. Using whitespace split for tokenization.")

# Initialize EasyOCR reader for Bangla
try:
    reader = easyocr.Reader(['bn'], gpu='mps')  # MPS for M1
except Exception:
    print("MPS failed, falling back to CPU")
    reader = easyocr.Reader(['bn'], gpu=False)

# Define versions and number of texts for testing
versions = ['typed', 'best_written']
num_texts = 1
base_dir = '/Users/simnim/text_project/texts'

# Lists to hold data
perf_data = {v: [] for v in versions[1:]}  # Skip 'typed' (ground truth)
token_counts = []

for i in range(1, num_texts + 1):
    text_dir = os.path.join(base_dir, f'text{i}')
    gt_count = None
    gt_text = None
    current_counts = {'text': i}
    
    if not os.path.exists(text_dir):
        print(f"Directory not found: {text_dir}")
        continue
    
    for v in versions:
        # Try multiple extensions
        possible_extensions = ['.jpg', '.jpeg', '.png']
        file_path = None
        for ext in possible_extensions:
            temp_path = os.path.join(text_dir, f'{v}{ext}')
            if os.path.exists(temp_path):
                file_path = temp_path
                break
        if not file_path:
            print(f"File not found for {v} in {text_dir} with extensions {possible_extensions}")
            continue
        
        # Preprocess image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to load image: {file_path}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        enhanced = cv2.convertScaleAbs(thresh, alpha=1.5, beta=50)
        temp_path = f'temp_text{i}_{v}.jpg'
        cv2.imwrite(temp_path, enhanced)
        
        # OCR
        ocr_result = reader.readtext(temp_path, detail=1)
        extracted_text = ' '.join([item[1] for item in ocr_result])
        print(f"\nText {i}, {v} OCR Extracted Text: {extracted_text}")
        print(f"OCR Confidence Scores: {[item[2] for item in ocr_result]}")
        
        # Tokenize
        if use_bnlp:
            tokens = tokenizer.tokenize(extracted_text)
        else:
            tokens = extracted_text.split()
        count = len(tokens)
        print(f"Tokens for {v}: {tokens}")
        print(f"Token Count for {v}: {count}")
        current_counts[v] = count
        
        if v == 'typed':
            gt_count = count
            gt_text = extracted_text
    
    if gt_count is None:
        print(f"Ground truth not found for text{i}")
        continue
    
    # Calculate accuracy for best_written relative to typed
    for v in versions[1:]:
        if v in current_counts:
            other_count = current_counts[v]
            accuracy = 1 - abs(other_count - gt_count) / gt_count if gt_count > 0 else 0
            perf_data[v].append(accuracy)
    
    token_counts.append(current_counts)

# Calculate mean performance
mean_perfs = {}
for v in versions[1:]:
    mean_perfs[v] = statistics.mean(perf_data[v]) if perf_data[v] else None

# Output to console
print("\nToken Counts:")
for counts in token_counts:
    print(counts)
print("\nPerformance (Accuracy) for best_written:")
for v, mean in mean_perfs.items():
    print(f"{v}: {mean:.4f}" if mean is not None else f"{v}: No data")

# Save to file
with open('test_results.txt', 'w') as f:
    f.write("Token Counts:\n")
    for counts in token_counts:
        f.write(f"{counts}\n")
    f.write("\nPerformance (Accuracy) for best_written:\n")
    for v, mean in mean_perfs.items():
        f.write(f"{v}: {mean:.4f}\n" if mean is not None else f"{v}: No data\n")