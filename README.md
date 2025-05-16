# Disaster Tweet Classification

This project is a machine learning solution to classify tweets as **real disaster-related** or **not**. It was developed for a competition setting, using a dataset of over 7,600 tweets that have been hand-labeled. The model is built using deep learning techniques with TensorFlow and Keras.

---

## 🧾 Introduction

Twitter is increasingly being used in times of emergency to share critical updates. Automatically identifying real disaster tweets can assist emergency responders and journalists. This project aims to build an NLP classifier that distinguishes between real and metaphorical references to disasters.

---

## 🚀 Features

- Preprocessing of tweet text using tokenization and padding.
- Deep learning model based on Bidirectional GRU architecture.
- Performance metrics include Accuracy, Precision, and F1 Score.
- Includes early stopping and learning rate reduction to prevent overfitting.

---

## ⚙️ Installation

Before running the project, set up a Python environment and install the dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/iamSabib/Tweet-Classification.git
cd Tweet-Classification
````

### 2. Install Required Packages

Use `pip` to install the necessary packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:

```bash
pip install pandas numpy scikit-learn tensorflow
```

---

## 💻 Usage

To train and evaluate the model:

```bash
python Team_29.py
```

Ensure `train.csv` and `test.csv` are in the same directory as the script or update the script paths accordingly.

---

## 🧠 Model Architecture

* **Embedding Layer**: Converts words to dense vectors.
* **Bidirectional GRU**: Captures context from both directions.
* **Dense Layers**: For classification with Batch Normalization and Dropout.
* **Optimizer**: Adam
* **Loss Function**: Binary Crossentropy
* **Regularization**: L2

---

## 📂 Dataset

The project uses a CSV file `train.csv` containing:

| Column   | Description                         |
| -------- | ----------------------------------- |
| id       | Unique identifier for the tweet     |
| keyword  | Disaster-related keyword (optional) |
| location | Tweet location (optional)           |
| text     | Tweet content                       |
| target   | 1 = real disaster, 0 = not disaster |

---

## 📈 Evaluation Metrics

The model is evaluated using:

* **Accuracy**
* **Precision**
* **F1 Score**

---

## 📌 Examples

Example of a real disaster tweet:

> "Forest fire near La Ronge Sask. Canada"

Example of a non-disaster metaphorical tweet:

> "I'm on fire today in the gym 🔥🔥"

---

## 🛠️ Troubleshooting

* **Memory Errors**: Try reducing batch size or sequence length.
* **Low Accuracy**: Ensure preprocessing steps are correctly applied and check dataset quality.
* **Package Issues**: Make sure TensorFlow is properly installed with a compatible Python version.

---

## 📄 License

This project is provided for educational purposes. Please review licensing terms if you intend to use it for production or commercial use.

---

