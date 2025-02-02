from django.shortcuts import render
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import os

# Model ve tokenizer'ı yükle
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "..", "bert_model")

print("Model ve tokenizer başarıyla yüklendi.")

model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tahmin edilen türler listesi (Eğitim sırasında kullandığınız türler)
classes = [
    "Bilim-Kurgu", "Animasyon", "Macera", "Komedi", "Aile",
    "Suç", "Gerilim", "Korku", "Aksiyon", "Dram", "Romantik", "Gizem"
]

# Tahmin fonksiyonu
def predict_genre(overview):
    inputs = tokenizer.encode_plus(
        overview,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    threshold = 0.095
    predicted_labels = [classes[i] for i, p in enumerate(probabilities) if p > threshold]
    return predicted_labels


# Ana sayfa görünümü
def home(request):
    if request.method == 'POST':
        film_ozeti = request.POST.get('film_ozeti', '')
        prediction = predict_genre(film_ozeti)
        return render(request, 'home.html', {'prediction': prediction, 'film_ozeti': film_ozeti})
    return render(request, 'home.html')
