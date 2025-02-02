import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- 1. Veri Yükleme ve İşleme ---
# Script'in bulunduğu dizini al
base_dir = os.path.dirname(os.path.abspath(__file__))

# Veri dosyasının tam yolunu oluştur
file_path = os.path.join(base_dir, "..", "data", "MovieData.csv")

# Veri dosyasını oku
data = pd.read_csv(file_path)

# Boş değerleri temizle
data.fillna('', inplace=True)

# Etiketleri birleştir
data['tags'] = data[['tags1', 'tags2', 'tags3', 'tags4']].values.tolist()
data['tags'] = data['tags'].apply(lambda x: list(set([tag for tag in x if tag])))  # Benzersiz ve boş olmayan etiketler

# Sadece belirli türlerle sınırlama
allowed_tags = ['Bilim-Kurgu', 'Animasyon', 'Macera', 'Komedi', 'Aile',
                'Suç', 'Gerilim', 'Korku', 'Aksiyon', 'Dram', 'Romantik', 'Gizem']
data['tags'] = data['tags'].apply(lambda x: [tag for tag in x if tag in allowed_tags])

# Çoklu etiketleri binary formatta dönüştür
mlb = MultiLabelBinarizer(classes=allowed_tags)
y = mlb.fit_transform(data['tags'])

# Eğitim ve test verilerini ayır
X_train, X_test, y_train, y_test = train_test_split(data['overview'], y, test_size=0.2, random_state=42)

# --- 2. Tokenizer ve Veri Seti ---
# BERT tokenizer'ı yükle
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Veri seti sınıfı
class MovieDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts.iloc[index]
        labels = self.labels[index]

        # Tokenize et
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(labels, dtype=torch.float),
        }

# Veri setlerini oluştur
train_dataset = MovieDataset(X_train, y_train, tokenizer)
test_dataset = MovieDataset(X_test, y_test, tokenizer)

# Veri yükleyiciler
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# --- 3. Model Tanımlama ---
# BERT modelini yükle
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=y.shape[1],  # Etiket sayısı
    problem_type="multi_label_classification",
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizasyon
optimizer = AdamW(model.parameters(), lr=1e-5)

# --- 4. Eğitim ---
def train_model(model, data_loader, optimizer, device, num_epochs=15):
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        loop = tqdm(data_loader, leave=True)
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Model tahmini ve loss hesaplama
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Optimizasyon
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # İlerleme durumu
            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item())


# --- Modeli Kaydetme ---
def save_model(model, tokenizer, save_path="bert_model"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"Model {save_path} konumuna kaydediliyor...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Model başarıyla kaydedildi.")

# --- Modeli Yükleme ---
def load_model(save_path="bert_model"):
    print(f"{save_path} konumundan model yükleniyor...")
    model = BertForSequenceClassification.from_pretrained(save_path)
    tokenizer = BertTokenizer.from_pretrained(save_path)
    print("Model başarıyla yüklendi.")
    return model, tokenizer

# Modeli eğit
train_model(model, train_loader, optimizer, device)

# Modeli kaydet
save_path = "../bert_model"  # Kaydetme klasörü
save_model(model, tokenizer, save_path)

# --- 5. Test ---
def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.sigmoid(logits).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    return np.array(predictions), np.array(targets)

# Modeli değerlendir
y_pred, y_true = evaluate_model(model, test_loader, device)

# Sınır belirleme (threshold)
threshold = 0.3
y_pred_binary = (y_pred > threshold).astype(int)

# Accuracy (doğruluk) hesaplama
accuracy = accuracy_score(y_true.flatten(), y_pred_binary.flatten())
print(f"Genel Doğruluk (Accuracy): {accuracy:.2f}")

print("Sınıflandırma Raporu:")
print(classification_report(y_true, y_pred_binary, target_names=mlb.classes_, zero_division=1))
