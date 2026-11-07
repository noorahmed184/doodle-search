import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
 
INDEX_CATEGORIES = ["cat", "dog", "sun", "smiley face", "chair",
                    "apple", "tree", "house", "car", "flower"]
 
HELD_OUT = ["cell phone", "bicycle"]
# creat file to save orginal img
os.makedirs("data/raw", exist_ok=True)
rng = np.random.default_rng(42)          
 
 
def load_category(name, how_many):
    path = "data/raw/" + name + ".npy"
 
    if not os.path.exists(path):
        url = ("https://storage.googleapis.com/quickdraw_dataset/full/"
               "numpy_bitmap/" + name.replace(" ", "%20") + ".npy")
        print("Downloading", name, "...")
        urllib.request.urlretrieve(url, path)
 
    data = np.load(path, mmap_mode="r")                    
    chosen = rng.choice(len(data), size=how_many, replace=False) # dont chose same img 
    images = np.array(data[np.sort(chosen)])
    return images.reshape(-1, 28, 28)                   

gallery_images, gallery_labels = [], []
query_images, query_labels = [], []
heldout_images, heldout_labels = [], []
 
# take 500 img \ 450 for gallery \ 50 for queries 
for name in INDEX_CATEGORIES:
    images = load_category(name, 500)
    gallery_images.append(images[:450])
    gallery_labels += [name] * 450
    query_images.append(images[450:])
    query_labels += [name] * 50
 
# for the held out take 50 img 
for name in HELD_OUT:
    images = load_category(name, 50)
    heldout_images.append(images)
    heldout_labels += [name] * 50
 # to combain 
gallery_images = np.concatenate(gallery_images)
query_images = np.concatenate(query_images)
heldout_images = np.concatenate(heldout_images)
 
np.savez_compressed("data/subset.npz",
                    gallery_images=gallery_images, gallery_labels=np.array(gallery_labels),
                    query_images=query_images, query_labels=np.array(query_labels),
                    heldout_images=heldout_images, heldout_labels=np.array(heldout_labels))
 
print("gallery:", gallery_images.shape)
print("queries:", query_images.shape)
print("held-out:", heldout_images.shape)
 
picked = rng.choice(len(gallery_images), 25, replace=False)
plt.figure(figsize=(6, 6))
for i, number in enumerate(picked):
    plt.subplot(5, 5, i + 1)
    plt.imshow(gallery_images[number], cmap="gray")
    plt.axis("off")
plt.savefig("data/grid_check.png")
plt.show()
