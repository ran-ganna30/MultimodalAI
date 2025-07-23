# Multimodal AI with CLIP - Python Code

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(images=image, return_tensors="pt")
    return inputs, processor

# Generate Image Embeddings
def generate_image_embeddings(inputs):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features, model

#Match Image to Captions
def match_captions(image_features, captions, clip_model, processor):
    text_inputs = processor(text=captions, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)

    image_features = image_features.detach().cpu().numpy()
    text_features = text_features.detach().cpu().numpy()

    similarities = cosine_similarity(image_features, text_features)
    best_indices = similarities.argsort(axis=1)[0][::-1]
    best_captions = [captions[i] for i in best_indices]
    return best_captions, similarities[0][best_indices].tolist()

def image_captioning(image_path, candidate_captions):
    inputs, processor = load_and_preprocess_image(image_path)
    image_features, clip_model = generate_image_embeddings(inputs)
    best_captions, similarities = match_captions(image_features, candidate_captions, clip_model, processor)
    return best_captions, similarities

# Example Captions List
candidate_captions = [
    "A refreshing beverage.",
    "Unwind and enjoy.",
    "Taste the world, one sip at a time.",
    "Your moment of peace, your daily indulgence.",
    "Savor the moment of peace."
]

if __name__ == "__main__":
    best_captions, similarities = image_captioning("/content/aman.png", candidate_captions)
    for i, (caption, sim) in enumerate(zip(best_captions[:5], similarities[:5])):
        print(f"{i+1}. {caption} (Similarity: {sim:.4f})")
