from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)

upload_folder = "uploads"
allowed_extension = {"jpg", "png", "jpeg", "gif", "bmp"}
max_file_size = 10 * 1024 * 1024

os.makedirs(upload_folder, exist_ok=True)
os.makedirs("static", exist_ok=True)

app.config["UPLOAD_FOLDER"] = upload_folder
app.config["MAX_CONTENT_LENGTH"] = max_file_size

model = YOLO("yolo11m-seg.pt") 

np.random.seed(42)
colors = np.random.randint(0, 255, (80, 3), dtype="uint8")

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in allowed_extension

def extract_data(img, model):
    h, w, ch = img.shape
    results = model.predict(source=img.copy(), save=False, conf=0.5)
    result = results[0]

    if result.masks is None:
        return [], [], [], [], result.names

    class_names = result.names
    seg_contour_idx = []
    for seg in result.masks.xyn:
        seg_rescaled = seg.copy()
        seg_rescaled[:, 0] = seg_rescaled[:, 0] * w
        seg_rescaled[:, 1] = seg_rescaled[:, 1] * h
        segment = np.array(seg_rescaled, dtype=np.int32)
        seg_contour_idx.append(segment)
        
    boxes = np.array(result.boxes.xyxy.cpu(), dtype="int")  
    class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
    scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
    
    return boxes, class_ids, seg_contour_idx, scores, class_names

def process_image(img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    boxes, class_ids, seg_contour_ids, scores, class_names = extract_data(img, model)

    if len(boxes) == 0:
        return img, []
    
    overlay = img.copy()
    detections = []

    for seg_contour, class_id in zip(seg_contour_ids, class_ids):
        color = [int(c) for c in colors[class_id]]
        cv2.fillPoly(overlay, [seg_contour], color)
        cv2.polylines(img, [seg_contour], True, color, 1)

    alpha = 0.35
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    for box, class_id, score in zip(boxes, class_ids, scores):
        (xmin, ymin, xmax, ymax) = box
        color = [int(c) for c in colors[class_id]]
        class_name = class_names[class_id]
        label = f"{class_name}: {score}"

        font_scale = 0.35
        thickness = 1
        
        (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_ymin = max(ymin, label_height + 5)

        cv2.rectangle(img, (xmin, label_ymin - label_height - 5),
                      (xmin + label_width + 2, label_ymin + baseline - 2), (0, 0, 0), -1)

        cv2.putText(img, label, (xmin + 1, label_ymin - 2), font, font_scale, (255, 255, 255), thickness)

        detections.append({
            "class": class_name,
            "confidence": float(score),
            "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)]
        })

    return img, detections

def image_to_base64(img):
    _, buffer = cv2.imencode(".png", img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"
    
@app.route('/')
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Dosya bulunamadı'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Dosya seçilmedi'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Geçersiz dosya formatı'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        if img is None:
            os.remove(filepath)
            return jsonify({'error': 'Görüntü okunamadı'}), 400

        original_base64 = image_to_base64(img)
        
        processed_img, detect_objects = process_image(img)
        processed_base64 = image_to_base64(processed_img)

        if os.path.exists(filepath):
            os.remove(filepath) 

        return jsonify({
            "success": True,
            "original_image": original_base64,
            "processed_image": processed_base64,
            "detected_objects": detect_objects,
            "total_objects": len(detect_objects)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)