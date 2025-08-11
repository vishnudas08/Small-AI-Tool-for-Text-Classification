# AI-Tool-for-Text-Classification
#📌 Project Overview

This project focuses on developing an NLP-based text classification tool using the BERT Transformer architecture and PyTorch. The model classifies news headlines into four categories — World, Sports, Business, and Sci/Tech — using the AG News dataset.

This was implemented as part of my effort to explore transformer models for real-world NLP tasks, apply data preprocessing, and build an inference-ready pipeline. The goal is to showcase fine-tuning BERT for domain-specific classification and to implement robust, reproducible training workflows.

🔗 Live Demo / Repository
GitHub Repository: [Your Repo Link Here]


⚙️ Tech Stacks
Python

PyTorch

Hugging Face Transformers

Pandas, NumPy

Scikit-learn

🧰 NLP & Deep Learning Techniques Used
Fine-tuning BERT (bert-base-uncased)

Tokenization & Encoding using Hugging Face Tokenizer

Custom PyTorch Classifier Head with Dropout & ReLU

Stratified Sampling for class balance

CrossEntropyLoss & Adam Optimizer for training

GPU Acceleration with PyTorch CUDA support

📊 Dataset Understanding
Dataset: AG News Dataset
Classes:

🌍 World

🏅 Sports

💼 Business

🔬 Sci/Tech

Training Data Preparation Steps:

Checked class distribution

Applied stratified sampling to ensure equal samples per class (807 samples/class)

Removed unnecessary columns (Title)

Split into 80% train / 20% validation

🧠 Data Preprocessing
Tokenized text with BertTokenizer using:

Padding: max_length

Truncation: max_length=512

Converted tokenized data into PyTorch Tensors

Created TensorDatasets for training and validation

Used DataLoaders with batch size 128

🏗 Model Architecture
BERT Base Model (bert-base-uncased)

Encoder layers frozen for faster training

Custom Classification Head:

Linear Layer → ReLU → Dropout → Linear Layer → Output Classes

Loss Function: CrossEntropyLoss

Optimizer: Adam (lr=0.001)

🖥 Training Process
Epochs: 2 (configurable)

Training Loop:

Forward pass

Loss computation

Backpropagation

Optimizer step

Validation Loop:

No gradient updates

Accuracy calculation

📌 Example Prediction
python
Copy
Edit
print(predict("Business Standard All India offers extensive coverage of financial news."))
# Output: Business
📈 Model Performance
Metric	Score
Validation Acc	~85%
Loss Function	CrossEntropyLoss
Optimizer	Adam

🏆 Project Outcome
This BERT-based NLP model:

Demonstrates transfer learning for text classification

Provides real-time classification for unseen text

Offers a reproducible training pipeline for similar NLP tasks

Can be extended for domain-specific datasets beyond AG News
Predict as i expected

