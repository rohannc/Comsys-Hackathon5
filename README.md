# COMSYS Hackathon-5: Robust Face Recognition & Gender Classification

A machine learning solution for **Face Recognition** and **Gender Classification** under adverse visual conditions, developed for the COMSYS Hackathon-5 competition themed "Robust Face Recognition and Gender Classification under Adverse Visual Conditions."

## 🎯 Project Overview

This project addresses real-world challenges in computer vision where face recognition systems must perform reliably despite environmental degradations such as:
- Motion blur and weather distortions
- Poor lighting conditions (low-light, overexposed scenes)
- Atmospheric effects (fog, rain simulation)
- Image noise and quality degradation

## 📊 Dataset: FACECOM

**FACECOM (Face Attributes in Challenging Environments)** - A specialized dataset with 5,000+ face images captured/synthesized under challenging visual conditions.

### Visual Conditions Included:
- Motion blur
- Overexposed/sunny scenes  
- Foggy conditions
- Rainy weather simulation
- Low light visibility
- Uneven lighting and glare

### ⚠️ Dataset Limitations
**Critical Note**: This implementation works with a **very limited subset** of the original FACECOM dataset. The small sample size (significantly reduced from 5,000+ images) presents major challenges:
- High risk of overfitting
- Limited model generalization capability
- Reduced robustness across diverse conditions
- Performance metrics may not reflect real-world deployment scenarios

## 🚀 Task Implementation

### Task A : Gender Classification (Binary Classification)
**Objective**: Predict gender (Male/Female) from degraded face images

**Architecture Used** :
- **Base Model** : MobileNetV2 with Transfer Learning
- **Input Size** : 224×224×3
- **Output** : Binary classification (sigmoid activation)
- **Preprocessing** : Normalization, data augmentation

**Key Features** :
- Transfer learning from ImageNet pre-trained MobileNetV2
- Data augmentation to compensate for limited samples
- Class weight balancing for imbalanced datasets
- Custom CNN fallback architecture available

### Task B : Face Recognition (Multi-class Classification)  
**Objective** : Identify specific individuals from facial features

**Architecture Used** :
- **Feature Extractor** : InceptionResNetV2 (pre-trained)
- **Classifier** : K-Nearest Neighbors (k = 1, cosine similarity)
- **Input Size** : 299×299×3
- **Feature Dimension** : 1536D embeddings

**Key Features** :
- Deep feature extraction using InceptionResNetV2
- Cosine similarity for face matching
- KNN classifier for person identification
- Robust embedding-based approach

## 🛠️ Technologies & Libraries

```python
# Core ML/DL Frameworks
tensorflow==2.x
keras
scikit-learn
numpy
opencv-python

# Data Processing & Augmentation  
albumentations
imageio
matplotlib
seaborn

# Model Persistence
joblib

# Environment
google-colab (development)
```

## 📁 Project Structure

```
├── Comsys_Hackathon5/
│   ├── Task_A/              # Gender classification data
│   │   ├── train/
│   │   │   ├── male/
│   │   │   └── female/
│   │   └── val/
│   ├── Task_B/              # Face recognition data  
│   │   └── all/             # Individual person folders
│   └── ...
├── files/                   # Processed data splits
│   ├── train/
│   ├── test/
│   └── validation/
├── models/
│   ├── my_model.keras           # Gender classification model
│   ├── gender_classification_model.h5
│   ├── knn_face_recognition_model.joblib
│   └── label_encoder.joblib
└── README.md
```

## 🔧 Implementation Workflow

### Data Preprocessing Pipeline
1. **Data Extraction** : Unzip and organize FACECOM dataset
2. **Folder Restructuring** : Move files from nested distortion folders  
3. **Train-Val-Test Split** : 70%-20%-10% split with stratification
4. **Class Reduction** : Trim to manageable subset (900 classes max)
5. **Data Augmentation** : Albumentations for geometric and photometric transforms

### Gender Classification Pipeline
```python
# Key preprocessing steps
- Image resizing: 224×224
- Normalization: [0,1] range
- Data augmentation: rotation, flip, shear, blur
- Class balancing: Compute sample weights for imbalanced classes

# Model architecture  
MobileNetV2 (frozen) → GlobalAveragePooling2D → Dense(128) → Dropout(0.5) → Dense(1, sigmoid)
```

### Face Recognition Pipeline
```python
# Feature extraction
InceptionResNetV2 (frozen) → Average Pooling → 1536D embeddings

# Classification
KNN classifier with cosine similarity distance metric
```

## 📈 Data Handling Strategies

Given the **severely limited dataset**, multiple strategies were implemented:

### Data Augmentation Techniques :
- **Geometric** : Rotation (±30°), horizontal flip, affine transformations
- **Photometric** : Gaussian blur, brightness adjustment
- **Noise Addition** : For regularization
- **Class Balancing** : Augment minority class to match majority

### Transfer Learning :
- **Pre-trained Models** : ImageNet weights for feature extraction
- **Frozen Base** : Prevent overfitting on small dataset
- **Fine-tuning** : Only train classification heads

### Validation Strategy :
- **Stratified Splits** : Maintain class distribution
- **Early Stopping** : Prevent overfitting
- **Cross-validation** : Where computationally feasible

## 🎯 Performance Metrics

### Gender Classification Results :
- **Primary Metric** : Binary accuracy
- **Additional** : Precision, Recall, F1-Score
- **Monitoring** : Training/validation accuracy curves
- **Visualization** : Confusion matrix heatmap

### Face Recognition Results :
- **Primary Metric** : Top-1 accuracy
- **Secondary** : Macro-averaged F1-Score  
- **Distance Metric** : Cosine similarity scores
- **Evaluation** : Classification report with per-class metrics

*⚠️ Performance Note : Results significantly impacted by limited training data. Larger datasets would substantially improve model robustness and generalization.*

## 🚧 Key Challenges & Limitations

### Dataset Constraints :
- **Sample Scarcity** : Insufficient data for robust model training
- **Class Imbalance** : Uneven distribution across gender/identity classes
- **Limited Diversity** : Reduced representation of challenging conditions
- **Overfitting Risk** : High variance due to small training sets

### Technical Challenges :
- **Memory Management** : Efficient processing of limited GPU resources
- **Data Pipeline** : Complex preprocessing for multiple degradation types
- **Model Selection** : Balancing complexity vs. available training data
- **Evaluation** : Reliable metrics with small validation sets

## 🔮 Future Improvements

### Data Enhancement :
- **Dataset Expansion** : Collect more diverse samples across all conditions
- **Synthetic Augmentation** : GAN-based data generation for rare conditions  
- **Active Learning** : Iterative data collection for challenging cases

### Model Architecture :
- **Ensemble Methods** : Combine multiple model predictions
- **Multi-task Learning** : Joint optimization of both tasks
- **Attention Mechanisms** : Focus on relevant facial regions
- **Domain Adaptation** : Transfer across different degradation types

### Deployment Optimization :
- **Model Compression** : Quantization and pruning for efficiency
- **Real-time Processing** : Optimize inference speed
- **Mobile Deployment** : Edge device compatibility
- **Robustness Testing** : Comprehensive evaluation across conditions

## 🚀 Usage Instructions

### Setup & Installation :
```bash
# Clone repository
git clone https://github.com/yourusername/comsys-hackathon5-solution.git
cd comsys-hackathon5-solution

# Install dependencies  
pip install tensorflow opencv-python scikit-learn albumentations joblib matplotlib seaborn
```

### Running the Models :
```python
# Gender Classification
from tensorflow.keras.models import load_model
model = load_model('my_model.keras')
prediction = predict_image('path/to/image.jpg')

# Face Recognition  
import joblib
knn = joblib.load('knn_face_recognition_model.joblib')
le = joblib.load('label_encoder.joblib')
identity = predict_face('path/to/face.jpg')
```

## 📊 Evaluation Metrics

### Task A - Gender Classification :
- **Accuracy** : Overall classification accuracy
- **Precision** : True positive rate per class
- **Recall** : Sensitivity per class  
- **F1-Score** : Harmonic mean of precision and recall

### Task B - Face Recognition:
- **Top-1 Accuracy** : Exact match accuracy
- **Macro F1-Score** : Average F1 across all identities
- **Cosine Similarity** : Feature space distance metric

## 🏆 Competition Context

**Event** : COMSYS Hackathon-5, 2025  
**Theme** : Robust Face Recognition and Gender Classification under Adverse Visual Conditions  
**Organizer** : COMSYS Educational Trust, Kolkata  
**Conference** : COMSYS-2025 (September 25-27, 2025, Warsaw, Poland)

## 🤝 Contributing

Contributions welcome! Areas for improvement :
- Enhanced data augmentation strategies
- Alternative model architectures
- Robust evaluation frameworks
- Real-world deployment optimizations

## 📞 Contact

- **Author** : **Bijit Mondal, Rohan Chakraborty, Soumita Nag**
- **Email** : **chakrabortyrohan.abc01@gmail.com**
- **Batch** : **Master of Computer Applications '26**
- **Institution** : **Jadavpur University**
- **Competition**: **COMSYS Hackathon-5 Participants**

## 🙏 Acknowledgments

- **COMSYS Educational Trust** for organizing the hackathon
- **FACECOM Dataset** creators for the challenging benchmark
- **Open Source Community** for excellent ML/DL frameworks
- **Warsaw University of Technology** for hosting COMSYS-2025

---

**⚠️ Important Disclaimer** : This solution was developed under significant data constraints with a very limited subset of the original FACECOM dataset. Performance metrics and model robustness should be interpreted considering these limitations. For production deployment, substantially larger and more diverse datasets are strongly recommended.
