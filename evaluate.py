import json
import numpy as np
import matplotlib.pyplot as plt
from search_core import load_model, embed, search
 
#read data 
data = np.load("data/subset.npz")
gallery_images = data["gallery_images"]
gallery_labels = data["gallery_labels"]
query_images = data["query_images"]
query_labels = data["query_labels"]
heldout_images = data["heldout_images"]
 
embeddings = np.load("index/embeddings.npy")    
model = load_model()
# trans test img into emedding 
query_vectors = embed(model, query_images)
heldout_vectors = embed(model, heldout_images)  
 
 
# chack the correct result and see most similarity ones 
def precision(q_vectors, g_vectors, k):
    scores = q_vectors @ g_vectors.T
    total = 0
    for i in range(len(scores)):
        top = np.argsort(-scores[i])[:k]
        total = total + (gallery_labels[top] == query_labels[i]).mean()
    return total / len(scores)
 
 
# trans to numbres then normalizaition row pix for dirict combaring
raw_gallery = gallery_images.reshape(len(gallery_images), -1).astype("float32")
raw_gallery = raw_gallery / np.linalg.norm(raw_gallery, axis=1, keepdims=True)
raw_query = query_images.reshape(len(query_images), -1).astype("float32")
raw_query = raw_query / np.linalg.norm(raw_query, axis=1, keepdims=True)
 
print("Raw pixels : p@5 =", round(precision(raw_query, raw_gallery, 5), 3),
      "| p@25 =", round(precision(raw_query, raw_gallery, 25), 3))
print("Your model : p@5 =", round(precision(query_vectors, embeddings, 5), 3),
      "| p@25 =", round(precision(query_vectors, embeddings, 25), 3))
 
 
# highst similarity score 
query_best = (query_vectors @ embeddings.T).max(axis=1)
heldout_best = (heldout_vectors @ embeddings.T).max(axis=1)
 
# try from 0 to 1 , 201 values 
best_t = 0
fewest_errors = 999999
for t in np.linspace(0, 1, 201):
    errors = (
    (query_best < t).mean() +
    (heldout_best >= t).mean())
    if errors < fewest_errors:
        fewest_errors = errors
        best_t = t
 
print("Threshold =", round(float(best_t), 3))
print("Wrongly rejected:", int((query_best < best_t).sum()), "out of", len(query_best))
print("Wrongly accepted:", int((heldout_best >= best_t).sum()), "out of", len(heldout_best))
 
json.dump({"threshold": round(float(best_t), 3), "top_k": 25},
          open("index/config.json", "w"))
 

plt.hist(query_best, bins=40, alpha=0.6, label="in-index queries")
plt.hist(heldout_best, bins=40, alpha=0.6, label="held-out")
plt.axvline(best_t, color="black", linestyle="--", label="threshold")
plt.legend()
plt.savefig("threshold_hist.png")
plt.close()
 
 

good = []
bad = []
for i in range(len(query_vectors)):
    best, _ = search(query_vectors[i], embeddings, 5)
    correct = (gallery_labels[best] == query_labels[i]).sum()
    if correct == 5 and len(good) < 3:
        good.append(i)
    if correct == 0 and len(bad) < 3:
        bad.append(i)
 
plt.figure(figsize=(9, 10))
row = 0
for i in good + bad:
    best, _ = search(query_vectors[i], embeddings, 5)
    plt.subplot(6, 6, row * 6 + 1)
    plt.imshow(query_images[i], cmap="gray")
    plt.title("GOOD" if row < len(good) else "BAD", fontsize=8)
    plt.axis("off")
    for col in range(5):
        plt.subplot(6, 6, row * 6 + col + 2)
        plt.imshow(gallery_images[best[col]], cmap="gray")
        plt.title(str(gallery_labels[best[col]]), fontsize=8)
        plt.axis("off")
    row = row + 1
plt.savefig("visual_check.png")
plt.show()
