from flask import Flask, render_template, request, jsonify, send_from_directory
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers as L
import numpy as np
import cv2
import os
import base64
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

# Custom layers
class Patches(L.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images, sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1], padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(L.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = L.Dense(units=projection_dim)
        self.position_embedding = L.Embedding(input_dim=num_patches, output_dim=projection_dim)
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return config

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = None
CLASS_NAMES = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

def load_model():
    global model
    try:
        print("Loading model...")
        model = tf.keras.models.load_model(
            'best_model.h5',
            custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder}
        )
        print("✓ Model loaded!")
        print(f"  Input: {model.input_shape}")
        print(f"  Output: {model.output_shape}")
    except Exception as e:
        print(f"✗ Error: {e}")

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

def get_gradcam_heatmap(model, image, class_index, layer_name='top_conv'):
    """
    Grad-CAM using top_conv - EXACT match to your notebook
    """
    try:
        print(f"  Generating Grad-CAM using layer: {layer_name}")
        
        # Get the target layer
        target_layer = model.get_layer(layer_name)
        
        # Create gradient model
        grad_model = Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )
        
        # Convert image to tensor
        if isinstance(image, np.ndarray):
            image_tensor = tf.constant(image, dtype=tf.float32)
        else:
            image_tensor = image
        
        # Compute gradients
        with tf.GradientTape() as tape:
            outputs = grad_model(image_tensor)
            if isinstance(outputs, (list, tuple)):
                conv_outputs = outputs[0]
                predictions = outputs[1]
            else:
                conv_outputs, predictions = outputs, outputs
            
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            
            class_channel = predictions[:, class_index]
        
        # Get gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Convert to numpy
        conv_outputs_array = np.array(conv_outputs)
        grads_array = np.array(grads)
        
        # Global average pooling - EXACTLY as in your notebook
        pooled_grads = np.mean(grads_array, axis=(0, 1, 2))
        
        # Get first image
        conv_outputs_array = conv_outputs_array[0]
        
        # Weight channels - EXACTLY as in your notebook
        for i in range(pooled_grads.shape[0]):
            conv_outputs_array[:, :, i] *= pooled_grads[i]
        
        # Average across channels
        heatmap = np.mean(conv_outputs_array, axis=-1)
        
        # ReLU and normalize - EXACTLY as in your notebook  
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap = heatmap / np.max(heatmap)
        
        print(f"  ✓ Grad-CAM generated! Shape: {heatmap.shape}")
        return heatmap
        
    except Exception as e:
        print(f"  ✗ Grad-CAM error: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_gradcam_overlay(original_img, heatmap, alpha=0.4):
    """
    Create Grad-CAM overlay EXACTLY matching your notebook's display_gradcam_pair
    """
    # Resize heatmap to match image dimensions
    img_height, img_width = original_img.size[1], original_img.size[0]
    heatmap_resized = cv2.resize(heatmap, (img_width, img_height))
    
    # Convert to uint8 (0-255 range)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply JET colormap - EXACTLY as in notebook
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Convert original image to numpy and ensure RGB
    img_array = np.array(original_img)
    
    # Convert to uint8 if needed
    if img_array.dtype != np.uint8:
        img_array = np.uint8(255 * img_array)
    
    # Convert to BGR for OpenCV
    if len(img_array.shape) == 2:  # Grayscale
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4:  # RGBA
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    elif img_array.shape[2] == 3:  # RGB
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    
    # Create overlay - EXACTLY as in notebook
    # addWeighted(img1, alpha1, img2, alpha2, gamma)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Convert back to RGB
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(overlay_rgb)

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

def create_confidence_chart(predictions):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(CLASS_NAMES)))
    bars = plt.barh(CLASS_NAMES, predictions * 100, color=colors)
    plt.xlabel('Confidence (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Classes', fontsize=12, fontweight='bold')
    plt.title('Prediction Confidence', fontsize=14, fontweight='bold')
    plt.xlim(0, 100)
    
    for bar, pred in zip(bars, predictions):
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{pred*100:.2f}%', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()
    return f"data:image/png;base64,{img_base64}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files (images, etc.)"""
    return send_from_directory('static', filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess and predict
        img_array, original_img = preprocess_image(filepath)
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        print(f"\n✓ Prediction: {predicted_class} ({confidence*100:.2f}%)")
        
        # Generate Grad-CAM
        heatmap = get_gradcam_heatmap(model, img_array, predicted_idx)
        
        # Create visualizations
        original_base64 = image_to_base64(original_img)
        
        gradcam_overlay = None
        if heatmap is not None:
            try:
                gradcam_img = create_gradcam_overlay(original_img, heatmap)
                gradcam_overlay = image_to_base64(gradcam_img)
                print("  ✓ Grad-CAM overlay created")
            except Exception as e:
                print(f"  ✗ Overlay creation failed: {e}")
        else:
            print("  ⊘ Grad-CAM not available (but predictions work fine!)")
        
        confidence_chart = create_confidence_chart(predictions)
        
        all_predictions = [
            {
                'class': CLASS_NAMES[i],
                'confidence': float(predictions[i]),
                'percentage': f"{predictions[i] * 100:.2f}%"
            }
            for i in range(len(CLASS_NAMES))
        ]
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'confidence_percentage': f"{confidence * 100:.2f}%",
            'all_predictions': all_predictions,
            'original_image': original_base64,
            'gradcam_overlay': gradcam_overlay,
            'confidence_chart': confidence_chart
        })
        
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files'}), 400
    
    files = request.files.getlist('files[]')
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    results = []
    for file in files:
        if file.filename == '':
            continue
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img_array, original_img = preprocess_image(filepath)
            predictions = model.predict(img_array, verbose=0)[0]
            predicted_idx = np.argmax(predictions)
            
            results.append({
                'filename': filename,
                'predicted_class': CLASS_NAMES[predicted_idx],
                'confidence': float(predictions[predicted_idx]),
                'image': image_to_base64(original_img)
            })
            os.remove(filepath)
        except Exception as e:
            results.append({'filename': file.filename, 'error': str(e)})
    
    return jsonify({'results': results})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔬 Diabetic Retinopathy Detection System")
    print("="*60)
    load_model()
    print("\n🚀 Starting server...")
    print("📍 http://localhost:5000")
    print("   OR http://10.243.27.217:5000")
    print("="*60 + "\n")
    print("💡 TIP: Open in incognito mode (Ctrl+Shift+N) to avoid cache issues")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
