# Leaf Disease Classifier – Potato & Tomato (Deep Learning + FastAPI)

A machine-learning powered API that identifies **plant leaf diseases** (Potato & Tomato) from a single image.  
This project supports automation in agriculture, early-disease intervention, and decision-support systems for farmers & smart-farming platforms.

---

## 🚀 Tech Stack

- TensorFlow / Keras – Deep learning & transfer learning  
- MobileNetV2 – Fine-tuned for high accuracy  
- FastAPI – High-performance web API  
- Docker + Google Cloud Run – Deployment  
- Python – Backend + ML inference pipeline  

---

## 📌 Features

| Capability | Description |
|-----------|-------------|
| 🎯 Disease prediction | Upload a plant leaf image → API returns disease name + confidence |
| 🌱 Supported plants | Potato, Tomato |
| 🧠 Transfer learning | MobileNetV2 fine-tuned (first 100 layers unfrozen) |
| ☁️ Cloud deployment | Fully serverless on GCP, auto-scaling |
| 📱 Integratable | Use with mobile apps, IoT devices & dashboards |

---

## 🧠 Model Summary

### 🥔 Potato Model

| Model Type | Train Acc. | Val. Acc. | Test Acc. |
|------------|-------------|-----------|-----------|
| Custom CNN | 0.8913 | 0.8418 | **0.8193** |
| **MobileNetV2 (fine-tuned)** | **0.9981** | **0.9473** | **0.9067** |

📌 Best model used for inference → potato_model.keras

---

### 🍅 Tomato Model

| Model Type | Model Used | Notes |
|------------|------------|-------|
| Custom CNN | – | Under-performing |
| **MobileNetV2 (fine-tuned)** | ✔ | Achieved high validation accuracy, used in final API |

Final deployed model → tomato_model.keras

---

## 🏗 Project Structure

Leaf-Disease-Classifier
│── Models/
│   ├── Potato/
│   │    └── potato_model.keras
│   └── Tomato/
│        └── tomato_model.keras
│
│── main.py              # FastAPI backend
│── requirements.txt     # Dependencies
│── README.md

---

## 🌍 Public API URL

https://leaf-api-888866409066.asia-south1.run.app/

---

## ⚙️ API Usage

### 🧪 Health Check – Root
GET /

Response:
{ "message": "Welcome to the Leaf Disease Classifier API!" }

---

### 🥔 Predict Potato Disease
POST /potato/
Content-Type: multipart/form-data
file = <image>

Example Response:
{ "disease": "Phytopthora", "confidence": 0.91 }

---

### 🍅 Predict Tomato Disease
POST /tomato/
Content-Type: multipart/form-data
file = <image>

Example Response:
{ "disease": "Tomato_Early_blight", "confidence": 0.87 }

---

## 🧪 Run Locally (Development)

1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Start the Server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

3️⃣ Test With CURL
curl -X POST "http://localhost:8000/potato/" \
 -H "Content-Type: multipart/form-data" \
 -F "file=@leaf.jpg"

---

## 🐳 Optional: Docker Run

docker build -t leaf-api .
docker run -p 8000:8000 leaf-api

---

## ☁️ Deployment Notes

This app is deployed using:
- Docker container build
- Hosted on Google Cloud Run
- Public endpoint for browser / mobile apps / IoT integrations

---

## 🔮 Future Enhancements

- Add support for rice, tea, and maize leaf disease detection
- Add bounding-box lesion segmentation
- Add Flutter-based farmer-mobile app
- Create web dashboard analytics

---

## 👨‍💻 Author

Anushka Sudeera Bandara
AI | Deep Learning | Full Stack Development

---

## ⭐ Support

If you like this project — please ⭐ star the repo!  
Pull requests & ideas to improve the classifier are welcome 😊

---

## 📜 License
MIT License – free to use & modify.
"""
