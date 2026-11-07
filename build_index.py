import os
import numpy as np
from search_core import load_model, embed
 
os.makedirs("index", exist_ok=True)
 
data = np.load("data/subset.npz")

gallery_images = data["gallery_images"]
gallery_labels = data["gallery_labels"]
 
print("Loading model")
model = load_model()
# transform the imge into numbers
print("Embedding gallery")
embeddings = embed(model, gallery_images)

np.save("index/embeddings.npy", embeddings)
np.save("index/gallery_images.npy", gallery_images)
np.save("index/gallery_labels.npy", gallery_labels)   
# show the shape of enbeddings
print("Shape:", embeddings.shape)
# to see the vector norm for first img 
print("Length of first row:", np.linalg.norm(embeddings[0]))
 
