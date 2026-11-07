import numpy as np
from PIL import Image, ImageFilter
 
INVERT = False

def load_model():
    from tensorflow.keras.applications import MobileNetV2
    model = MobileNetV2(weights="imagenet", include_top=False,
                        pooling="avg", input_shape=(224, 224, 3))
    return model

def to_model_input(images_28):
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
 
    x = np.array(images_28, dtype="float32").reshape(-1, 28, 28)
 
    if INVERT:
        x = 255 - x                      
    x = x[..., np.newaxis]             
    x = tf.image.resize(x, (224, 224)).numpy()  
    x = np.repeat(x, 3, axis=-1)         
    x = preprocess_input(x)              
    return x

def embed(model, images_28):
    all_vectors = []
    for start in range(0, len(images_28), 64):           
        batch = images_28[start:start + 64]
        vectors = model.predict(to_model_input(batch), verbose=0)
        all_vectors.append(vectors)
 
    vectors = np.vstack(all_vectors)
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / lengths 
 
def preprocess_query(image):
    gray = np.array(flatten_to_white(image).convert("L"), dtype="float32")
 
    if gray.max() - gray.min() < 40:
        return None
 
    strokes = gray < (gray.mean() - 40)
    if strokes.sum() < 20:
        return None
 
    ys, xs = np.where(strokes)
    strokes = strokes[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
 
    h, w = strokes.shape
    side = max(h, w)
    margin = max(2, int(side * 0.12))
    square = np.zeros((side + 2 * margin, side + 2 * margin), dtype="uint8")
    top = margin + (side - h) // 2
    left = margin + (side - w) // 2
    square[top:top + h, left:left + w] = strokes * 255  
    img = Image.fromarray(square).resize((56, 56))
    img = img.filter(ImageFilter.MaxFilter(3))     
    img = img.resize((28, 28))
 
    arr = np.array(img, dtype="float32")
    arr = arr / arr.max() * 255                    
    return arr.astype("uint8")

def search(query_vector, embeddings, k=25):
    scores = embeddings @ query_vector     
    best = np.argsort(-scores)[:k]          
    return best, scores[best]
