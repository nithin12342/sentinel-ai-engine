"""
Azure ML Inference Scoring Script
Biometric Recognition Model Scoring Service
"""
import json
import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import mlflow
from insightface.app import FaceAnalysis


def init():
    """
    Initialize the model and face analysis app
    Called once when the endpoint starts
    """
    global face_app, model, device
    
    # Get model path from environment
    model_path = os.getenv("AZUREML_MODEL_DIR", "models")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize InsightFace
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Load model if exists
    model_path = os.path.join(model_path, "biometric_model.pt")
    if os.path.exists(model_path):
        model = torch.jit.load(model_path)
    else:
        model = None
    
    print(f"Model loaded on device: {device}")
    print("Biometric inference service initialized")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess input image for face detection"""
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(image)
    
    return img_array


def detect_and_embed(img_array: np.ndarray) -> dict:
    """
    Detect faces and generate embeddings
    Returns list of detected faces with embeddings
    """
    # Detect faces
    faces = face_app.get(img_array)
    
    results = []
    for face in faces:
        # Get embedding
        embedding = face.embedding
        
        results.append({
            "bbox": face.bbox.tolist(),
            "score": float(face.det_score),
            "embedding": embedding.tolist(),
            "embedding_norm": float(np.linalg.norm(embedding))
        })
    
    return results


def match_faces(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Match two face embeddings using cosine similarity"""
    # Normalize embeddings
    e1 = embedding1 / np.linalg.norm(embedding1)
    e2 = embedding2 / np.linalg.norm(embedding2)
    
    # Cosine similarity
    similarity = np.dot(e1, e2)
    
    return float(similarity)


def run(raw_data):
    """
    Process incoming requests
    Called for each inference request
    """
    try:
        # Parse input
        input_data = json.loads(raw_data.decode('utf-8'))
        
        if "image" in input_data:
            # Single image processing
            import io
            image_bytes = input_data["image"].encode('utf-8')
            import base64
            image_bytes = base64.b64decode(image_bytes)
            
            # Preprocess and detect
            img_array = preprocess_image(image_bytes)
            faces = detect_and_embed(img_array)
            
            return json.dumps({
                "status": "success",
                "faces_detected": len(faces),
                "faces": faces
            })
        
        elif "images" in input_data:
            # Face verification (compare two images)
            import base64
            import io
            
            img1_bytes = base64.b64decode(input_data["images"][0]["data"])
            img2_bytes = base64.b64decode(input_data["images"][1]["data"])
            
            img1 = preprocess_image(img1_bytes)
            img2 = preprocess_image(img2_bytes)
            
            faces1 = detect_and_embed(img1)
            faces2 = detect_and_embed(img2)
            
            if not faces1 or not faces2:
                return json.dumps({
                    "status": "error",
                    "message": "No faces detected in one or both images"
                })
            
            # Compare first face from each image
            similarity = match_faces(
                np.array(faces1[0]["embedding"]),
                np.array(faces2[0]["embedding"])
            )
            
            # Decision threshold
            threshold = 0.5
            match = similarity >= threshold
            
            return json.dumps({
                "status": "success",
                "match": match,
                "similarity": similarity,
                "threshold": threshold,
                "faces_in_image1": len(faces1),
                "faces_in_image2": len(faces2)
            })
        
        else:
            return json.dumps({
                "status": "error",
                "message": "Invalid input format"
            })
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


def main():
    """Local testing entry point"""
    # Test init
    init()
    
    print("Biometric scoring service ready")


if __name__ == "__main__":
    main()
