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

def flatten_to_white(image):
     gray_img = flatten_to_white(image).convert("L")
    gray_img.thumbnail((400, 400))
    gray = np.array(gray_img, dtype="float32")
 
    # ب) ما فيه رسم؟ (الصورة كلها لون واحد تقريباً)
    if gray.max() - gray.min() < 40:
        return None
 
    # ج) الخطوط = البكسلات اللي أغمق من اللي حولها بشكل واضح
    #    (هذا يتجاهل الخلفيات الغامقة الكبيرة مثل الخشب والظلال)
    blur = max(2, max(gray.shape) // 15)
    around = np.array(gray_img.filter(ImageFilter.GaussianBlur(blur)), dtype="float32")
    strokes = gray < (around - 30)
 
    # د1) ننظف: نشيل النقاط الصغيرة، ونشيل أي شي ملاصق لحافة الصورة
    #     (يد، قلم، حافة طاولة...)
    labels, count = ndimage.label(strokes, structure=np.ones((3, 3)))
    sizes = ndimage.sum(strokes, labels, range(1, count + 1))
    on_edge = np.unique(np.concatenate(
        [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    keep = np.zeros(count + 1, dtype=bool)
    for number in range(1, count + 1):
        if sizes[number - 1] >= 10 and number not in on_edge:
            keep[number] = True
    cleaned = keep[labels]
    if cleaned.sum() >= 20:          # إذا بقي شي نستخدمه، وإلا نرجع للأصلي
        strokes = cleaned
 
    if strokes.sum() < 20:
        return None
 
    # د2) نقص حول الرسمة
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
