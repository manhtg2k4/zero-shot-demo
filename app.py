import streamlit as st
from PIL import Image
import torch
import clip
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# ⚙️ 1. Khởi tạo app
# ============================================================
st.set_page_config(page_title="Zero-Shot Animal Classifier", page_icon="🐾", layout="wide")
st.title("🐾 Zero-Shot Animal Classification with CLIP (OpenAI)")
st.write("Upload an animal image — model will predict its species and, if cat/dog, its breed (zero-shot).")

# Buộc dùng CPU vì Streamlit Cloud không có GPU
device = "cpu"

# ============================================================
# 📚 2. Cache model và text embeddings (load lazily để tiết kiệm memory)
# ============================================================
@st.cache_resource
def load_clip_and_features():
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Species classes
    species_classes = ["cat", "dog", "cattle", "chicken", "elephant"]
    species_tokens = clip.tokenize(species_classes).to(device)
    with torch.no_grad():
        species_text_features = model.encode_text(species_tokens)
        species_text_features /= species_text_features.norm(dim=-1, keepdim=True)
    
    # Sửa breed lists (swap để đúng: cat_breeds là mèo, dog_breeds là chó)
    cat_breeds = ['Abyssinian', 'Bengal', 'Birman', 'Maine_Coon', 'Siamese']
    dog_breeds = ['french_bulldog', 'german_shepherd', 'golden_retriever', 'poodle', 'yorkshire_terrier']
    all_breeds = cat_breeds + dog_breeds  # Giữ all để predict chung
    
    breed_tokens = clip.tokenize(all_breeds).to(device)
    with torch.no_grad():
        breed_text_features = model.encode_text(breed_tokens)
        breed_text_features /= breed_text_features.norm(dim=-1, keepdim=True)
    
    return model, preprocess, species_classes, species_text_features, all_breeds, breed_text_features

# ============================================================
# 🧩 3. Hàm dự đoán 1 ảnh
# ============================================================
def predict_single_image(img_path_or_obj, show=False):
    model, preprocess, species_classes, species_text_features, all_breeds, breed_text_features = load_clip_and_features()
    
    image = preprocess(Image.open(img_path_or_obj)).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = model.encode_image(image)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        sim_species = (img_feat @ species_text_features.T).squeeze(0)
        probs = sim_species.softmax(dim=0).cpu().numpy()
        top_species_idx = sim_species.topk(1).indices.item()
        predicted_species = species_classes[top_species_idx]

        predicted_breed = None
        breed_probs = None
        if predicted_species in ["dog", "cat"]:
            sim_breed = (img_feat @ breed_text_features.T).squeeze(0)
            breed_probs = sim_breed.softmax(dim=0).cpu().numpy()
            top_breed_idx = sim_breed.topk(1).indices.item()
            predicted_breed = all_breeds[top_breed_idx]

    return predicted_species, predicted_breed, probs, breed_probs

# ============================================================
# 🖼️ 4. Upload ảnh & dự đoán
# ============================================================
uploaded_file = st.file_uploader("Upload your animal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Hiển thị ảnh
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Dự đoán
        species, breed, species_probs, breed_probs = predict_single_image(uploaded_file)

        # Hiển thị kết quả
        st.subheader(f"Predicted Species: **{species.capitalize()}** 🧠")

        # Biểu đồ xác suất 5 loài
        st.write("### Species Probability:")
        fig, ax = plt.subplots()
        ax.bar(load_clip_and_features()[2], species_probs, color="skyblue")  # species_classes từ cache
        ax.set_ylabel("Confidence")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        if breed:
            st.markdown(f"### 🐶 Predicted Breed (Zero-Shot): **{breed}**")

            # Hiển thị top-3 breed
            topk = np.argsort(breed_probs)[::-1][:3]
            st.write("Top-3 Breed Predictions:")
            for i in topk:
                st.write(f"{load_clip_and_features()[4][i]}: {breed_probs[i]*100:.2f}%")  # all_breeds từ cache

            # Biểu đồ xác suất breed
            fig2, ax2 = plt.subplots()
            ax2.barh([load_clip_and_features()[4][i] for i in topk][::-1],
                     [breed_probs[i] for i in topk][::-1],
                     color="salmon")
            ax2.set_xlabel("Confidence")
            st.pyplot(fig2)
    
    except Exception as e:
        st.error(f"Lỗi xử lý: {str(e)}. Có thể do memory hoặc hình ảnh không hợp lệ. Thử reboot app hoặc dùng hình khác.")

else:
    st.info("👆 Upload an image to start prediction.")