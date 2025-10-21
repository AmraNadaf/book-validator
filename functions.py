import io
import cv2
import easyocr
import difflib
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import logging

# --- LAZY LOADING: Initialize as None ---
_reader = None

def get_ocr_reader():
    global _reader
    if _reader is None:
        logging.info("First request: Loading EasyOCR model (this may take 20–60 seconds)...")
        _reader = easyocr.Reader(['en'], gpu=False)
        logging.info("EasyOCR model loaded and cached.")
    return _reader

app = Flask(__name__, static_url_path='', static_folder='.')

# ==================== FUNCTION 1: Badge Overlap Detection ====================

def find_overlap_text_dual(text_detections, badge_coords, expected_text1, expected_text2, similarity_threshold=0.8, tolerance=3):
    badge_x1, badge_y1, badge_x2, badge_y2 = badge_coords
    safe_margin_flag_foverlap = False
    overlap_flag = False
    overlap_text = ""

    for bbox, text, score in text_detections:
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]

        if any((badge_x1 - tolerance) <= x <= (badge_x2 + tolerance) and
               (badge_y1 - tolerance) <= y <= (badge_y2 + tolerance) for x, y in zip(xs, ys)):

            similarity1 = difflib.SequenceMatcher(None, text.lower(), expected_text1.lower()).ratio()
            similarity2 = difflib.SequenceMatcher(None, text.lower(), expected_text2.lower()).ratio()

            if similarity1 < similarity_threshold and similarity2 < similarity_threshold:
                logging.warning(f"⚠️ Overlapping text found: '{text}'")
                safe_margin_flag_foverlap = True
                overlap_flag = True
                overlap_text = text

    if not overlap_flag:
        logging.info("✅ No unwanted overlaps detected.")

    return safe_margin_flag_foverlap, overlap_flag, overlap_text


# ==================== FUNCTION 2: Safe Margin Detection ====================

def check_safe_margins(text_detections_right, safe_margin_px, right_half_width):
    safe_margin_flagged = False
    unsafe_texts = []

    for bbox, text, score in text_detections_right:
        xs = [pt[0] for pt in bbox]

        if any(x < safe_margin_px for x in xs):
            logging.warning(f"⚠️ Text '{text}' violates LEFT margin")
            safe_margin_flagged = True
            unsafe_texts.append(text)
        elif any(x > (right_half_width - safe_margin_px) for x in xs):
            logging.warning(f"⚠️ Text '{text}' violates RIGHT margin")
            safe_margin_flagged = True
            unsafe_texts.append(text)

    if not safe_margin_flagged:
        logging.info("✅ All text within safe margins")

    return safe_margin_flagged, unsafe_texts


# ==================== FUNCTION 3: Image Quality Assessment ====================

def comprehensive_image_quality(image_region, expected_width_inches=5, expected_height_inches=8):
    height_px, width_px = image_region.shape[:2]
    dpi_width = width_px / expected_width_inches
    dpi_height = height_px / expected_height_inches
    avg_dpi = (dpi_width + dpi_height) / 2

    if avg_dpi >= 300:
        dpi_status = "✅ EXCELLENT - Print Ready"
        dpi_score = 100
    elif avg_dpi >= 200:
        dpi_status = "⚠️ ACCEPTABLE - May show quality loss"
        dpi_score = 60
    elif avg_dpi >= 150:
        dpi_status = "⚠️ POOR - Visible pixelation expected"
        dpi_score = 30
    else:
        dpi_status = "❌ REJECTED - Not suitable for printing"
        dpi_score = 0

    gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY).astype(np.float32)
    block_size = 8
    vertical_diffs = [np.mean(np.abs(gray[:, i] - gray[:, i - 1])) for i in range(block_size, width_px, block_size)]
    horizontal_diffs = [np.mean(np.abs(gray[j, :] - gray[j - 1, :])) for j in range(block_size, height_px, block_size)]
    blockiness = (np.mean(vertical_diffs) + np.mean(horizontal_diffs)) / 2

    if blockiness > 15:
        pixel_status = "Highly Pixelated"
        pixel_score = 0
    elif blockiness > 4:
        pixel_status = "Moderately Pixelated"
        pixel_score = 50
    else:
        pixel_status = "Not Pixelated"
        pixel_score = 100

    if dpi_score >= 60 and pixel_score >= 50:
        overall_assessment = "✅ PASS - Suitable for printing"
    elif dpi_score >= 30 and pixel_score >= 50:
        overall_assessment = "⚠️ REVIEW - May have issues"
    else:
        overall_assessment = "❌ FAIL - Not suitable for printing"

    return {
        'dpi': round(avg_dpi, 1),
        'dpi_status': dpi_status,
        'dpi_score': dpi_score,
        'blockiness_score': round(blockiness, 2),
        'pixelation_status': pixel_status,
        'pixel_score': pixel_score,
        'overall_assessment': overall_assessment
    }


# ==================== FUNCTION 4: Final Decision ====================

def make_final_assessment(overlap_flag, safe_margin_flagged, quality_results):
    dpi_score = quality_results.get('dpi_score', 0)
    if overlap_flag or dpi_score < 60 or safe_margin_flagged:
        return "Review Needed"
    return "Pass"


# ==================== FLASK ROUTES ====================

@app.route('/')
def serve_html():
    return render_template('upload.html')

@app.route('/validate', methods=['POST'])
def validate():
    file = request.files.get('file')
    if not file:
        return jsonify({"message": "No file received"}), 400
    try:
        # Read image
        image_bytes = file.read()
        img = np.array(Image.open(io.BytesIO(image_bytes)))
        height, width = img.shape[:2]
        right_half = img[:, width // 2:]

        # Lazy-load OCR reader
        reader = get_ocr_reader()

        expected_text1 = "Winner of the 21st Century Emily Dickinson Award"
        expected_text2 = "Award"

        text_detections = reader.readtext(img)
        badge_coords = (width // 2, height - 106, width, height)
        safe_margin_flag_foverlap, overlap_flag, overlap_text = find_overlap_text_dual(
            text_detections, badge_coords, expected_text1, expected_text2
        )

        quality_results = comprehensive_image_quality(right_half)
        text_detections_right = reader.readtext(right_half)

        actual_dpi = 100
        safe_margin_px = (3 / 25.4) * actual_dpi
        right_half_width = right_half.shape[1]
        safe_margin_flagged, unsafe_texts = check_safe_margins(
            text_detections_right, safe_margin_px, right_half_width
        )

        if safe_margin_flag_foverlap:
            safe_margin_flagged = True

        overall_assessment = make_final_assessment(overlap_flag, safe_margin_flagged, quality_results)

        return jsonify({
            'overlap_flag': overlap_flag,
            'overlap_text': overlap_text,
            'safe_margin_flagged': safe_margin_flagged,
            'unsafe_texts': unsafe_texts,
            **quality_results,
            'overall_assessment': overall_assessment
        })

    except Exception as e:
        logging.error(f"Error during classification: {str(e)}", exc_info=True)
        return jsonify(result="Error during cover validation"), 500


# --- Required for Render ---
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
