# IMDb Sentiment Analysis Using Deep Learning

This project implements sentiment classification on IMDb movie reviews using deep learning models built with PyTorch. The notebook walks through data preparation, tokenization, model development, and performance evaluation for binary text classification.

## 🎬 Dataset

- **Source:** [IMDb Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Classes:** Positive / Negative
- **Format:** Raw `.tar.gz` file containing text data

## 🔧 Preprocessing

- Extract dataset from archive
- Tokenize using `torchtext` and `CountVectorizer`
- Encode labels and pad sequences
- Prepare PyTorch `TensorDataset` for training and evaluation

## 🧠 Models Implemented

- **Fully Connected Feedforward Neural Network**
- **Recurrent Neural Network (RNN)**

Both models are implemented in PyTorch and trained on padded tokenized sequences for binary classification.

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## 📈 Visualization

- Training loss curves
- Confusion matrix heatmap
- Prediction performance metrics

## 🧰 Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch torchtext torchinfo nltk
