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

try:
    import pytesseract
    use_tesseract = True
except ImportError:
    use_tesseract = False
    print("Warning: pytesseract not available. Using EasyOCR only.")

try:
    from Levenshtein import distance
    use_levenshtein = True
except ImportError:
    use_levenshtein = False
    print("Warning: python-Levenshtein not available. Using token-based accuracy only.")

# Initialize EasyOCR reader
try:
    reader = easyocr.Reader(['bn'], gpu='mps')
except Exception:
    print("MPS failed, falling back to CPU")
    reader = easyocr.Reader(['bn'], gpu=False)

# Define versions and number of texts
versions = ['typed', 'best_written']
num_texts = 1
base_dir = '/Users/simnim/text_project/texts'

# Lists to hold data
perf_data = {v: [] for v in versions[1:]}
token_counts = []
extracted_texts = {}
distinct_chars = {}

for i in range(1, num_texts + 1):
    text_dir = os.path.join(base_dir, f'text{i}')
    gt_count = None
    gt_text = None
    current_counts = {'text': i}
    
    if not os.path.exists(text_dir):
        print(f"Directory not found: {text_dir}")
        continue
    
    for v in versions:
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
        if image.shape[0] > 1000:
            image = cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if v == 'typed':
            enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=50)
        else:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            enhanced = cv2.dilate(thresh, kernel, iterations=1)
        temp_path = f'temp_text{i}_{v}.jpg'
        cv2.imwrite(temp_path, enhanced)
        
        # OCR
        ocr_result = reader.readtext(temp_path, detail=1, contrast_ths=0.2, adjust_contrast=0.7)
        extracted_text = ' '.join([item[1] for item in ocr_result])
        confidence_scores = [item[2] for item in ocr_result]
        if use_tesseract and max(confidence_scores, default=0) < 0.3:
            print(f"Low confidence for {v}, trying Tesseract...")
            extracted_text = pytesseract.image_to_string(temp_path, lang='ben')
            confidence_scores = [0.5]  # Dummy confidence
        print(f"\nText {i}, {v} OCR Extracted Text: {extracted_text}")
        print(f"OCR Confidence Scores: {confidence_scores}")
        
        # Tokenize
        if use_bnlp:
            tokens = tokenizer.tokenize(extracted_text)
        else:
            tokens = extracted_text.split()
        count = len(tokens)
        print(f"Tokens for {v}: {tokens}")
        print(f"Token Count for {v}: {count}")
        current_counts[v] = count
        extracted_texts[v] = extracted_text
        
        # Count distinct characters
        unique_chars = set(extracted_text.replace(' ', ''))  # Remove spaces
        distinct_chars[v] = {'count': len(unique_chars), 'chars': sorted(unique_chars)}
        print(f"Distinct Characters for {v}: {len(unique_chars)}")
        print(f"Character Set for {v}: {sorted(unique_chars)}")
        
        if v == 'typed':
            gt_count = count
            gt_text = extracted_text
    
    if gt_count is None:
        print(f"Ground truth not found for text{i}")
        continue
    
    # Calculate accuracy
    for v in versions[1:]:
        if v in current_counts:
            other_count = current_counts[v]
            other_text = extracted_texts.get(v, '')
            token_accuracy = 1 - abs(other_count - gt_count) / gt_count if gt_count > 0 else 0
            text_accuracy = 0
            if use_levenshtein:
                edit_dist = distance(gt_text, other_text)
                text_accuracy = 1 - edit_dist / max(len(gt_text), len(other_text)) if max(len(gt_text), len(other_text)) > 0 else 0
            perf_data[v].append({'token_accuracy': token_accuracy, 'text_accuracy': text_accuracy})
    
    token_counts.append(current_counts)

# Calculate mean performance
mean_perfs = {}
for v in versions[1:]:
    if perf_data[v]:
        mean_token_accuracy = statistics.mean([d['token_accuracy'] for d in perf_data[v]])
        mean_text_accuracy = statistics.mean([d['text_accuracy'] for d in perf_data[v]]) if use_levenshtein else None
        mean_perfs[v] = {'token_accuracy': mean_token_accuracy, 'text_accuracy': mean_text_accuracy}
    else:
        mean_perfs[v] = None

# Output to console
print("\nToken Counts:")
for counts in token_counts:
    print(counts)
print("\nDistinct Character Counts:")
for v in versions:
    if v in distinct_chars:
        print(f"{v}: {distinct_chars[v]['count']} unique characters")
print("\nPerformance (Accuracy) for best_written:")
for v, mean in mean_perfs.items():
    if mean is not None:
        if use_levenshtein:
            print(f"{v}: Token Accuracy = {mean['token_accuracy']:.4f}, Text Accuracy = {mean['text_accuracy']:.4f}")
        else:
            print(f"{v}: Token Accuracy = {mean['token_accuracy']:.4f}")
    else:
        print(f"{v}: No data")

# Save to file
with open('test_results.txt', 'w') as f:
    f.write("Token Counts:\n")
    for counts in token_counts:
        f.write(f"{counts}\n")
    f.write("\nDistinct Character Counts:\n")
    for v in versions:
        if v in distinct_chars:
            f.write(f"{v}: {distinct_chars[v]['count']} unique characters\n")
    f.write("\nPerformance (Accuracy) for best_written:\n")
    for v, mean in mean_perfs.items():
        if mean is not None:
            if use_levenshtein:
                f.write(f"{v}: Token Accuracy = {mean['token_accuracy']:.4f}, Text Accuracy = {mean['text_accuracy']:.4f}\n")
            else:
                f.write(f"{v}: Token Accuracy = {mean['token_accuracy']:.4f}\n")
        else:
            f.write(f"{v}: No data\n")