# Resume-Screening-Using-Machine-Learning

# Resume Screening Project

## 📌 Project Overview
This project focuses on automating the **resume screening** process using **Natural Language Processing (NLP) and Machine Learning**. The goal is to classify resumes into different job categories based on their content.

## 📂 Dataset
**Dataset Name:** [UpdatedResumeDataSet.csv](https://www.kaggle.com/datasets)

- The dataset contains resumes along with their associated job categories.
- It is used to train a classification model that predicts the most relevant category for a given resume.

## 🛠️ Project Steps

### 1️⃣ Data Preprocessing
- Load and clean the dataset.
- Remove stopwords, special characters, and unnecessary symbols.
- Convert text to lowercase and perform **lemmatization**.

### 2️⃣ Feature Extraction
- **TF-IDF Vectorization** to convert textual data into numerical form.

### 3️⃣ Model Training
- Train different machine learning models (Logistic Regression, SVM, Random Forest, etc.)
- Evaluate the models using accuracy, precision, recall, and F1-score.

### 4️⃣ Visualization
- **Category Distribution:** Understanding the distribution of resumes across categories.
- **Word Cloud:** Identifying frequently used words in resumes.
- **TF-IDF Scores:** Visualizing the most important words.
- **Confusion Matrix:** Analyzing model performance.
- **Training Data vs Accuracy:** Observing how training size impacts accuracy.

### 5️⃣ Model Evaluation & Deployment
- Fine-tune the best-performing model.
- Deploy using **Flask/Streamlit** for real-time resume classification.

## 📊 Visualizations
- **Category-wise resume count** (Bar Chart)
- **Frequent words in resumes** (Word Cloud)
- **Feature importance using TF-IDF** (Bar Chart)
- **Confusion Matrix**
- **Model accuracy vs training data size** (Line Plot)

## 💡 Technologies Used
- **Python** (pandas, numpy, seaborn, matplotlib)
- **NLTK & scikit-learn** for text processing & model training
- **TF-IDF** for feature extraction
- **Matplotlib & Seaborn** for visualization
- **Flask/Streamlit** for deployment

## 🚀 How to Run the Project
1. **Install dependencies**
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn nltk wordcloud
   ```
2. **Run the script**
   ```python
   python main.py
   ```
3. **Deploy using Streamlit (optional)**
   ```bash
   streamlit run app.py
   ```

## 🔥 Future Improvements
- Experiment with **deep learning models** (LSTMs, Transformers)
- Implement **resume ranking** based on job descriptions
- Create an **interactive dashboard** for better visualization

## 👨‍💻 Contributors
- **Azmain Abid Khan** (azmain878@gmail.com)

---
Let me know if you need modifications! 🚀

# Resume Screening Using NLP - Q&A Guide

## 1. Introduction
### Q: What is the goal of this project?
A: The goal is to classify resumes into job categories using machine learning (ML) and natural language processing (NLP).

### Q: Why is this problem important?
A: Manual resume screening is time-consuming. Automating this process can help recruiters quickly filter candidates based on job roles.

---

## 2. Data Preprocessing
### Q: Why is data preprocessing necessary in NLP?
A: Text data is unstructured. Preprocessing helps convert it into a format suitable for machine learning models.

### Q: What are the key preprocessing steps?
A:
1. Convert text to lowercase
2. Remove punctuation & special characters
3. Tokenization (splitting text into words)
4. Remove stopwords (common words like 'the', 'is')
5. Lemmatization (reducing words to their base form)

### Q: How do you convert text to lowercase in Python?
A:
```python
df['Resume'] = df['Resume'].str.lower()
```

### Q: How do you remove punctuation and special characters?
A:
```python
import re
df['Resume'] = df['Resume'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
```

### Q: What is tokenization, and how is it performed?
A: Tokenization breaks text into individual words.
```python
from nltk.tokenize import word_tokenize
nltk.download('punkt')
df['Resume'] = df['Resume'].apply(lambda x: word_tokenize(x))
```

### Q: Why do we remove stopwords?
A: Stopwords (e.g., 'is', 'the', 'and') do not add value to the model.
```python
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
df['Resume'] = df['Resume'].apply(lambda x: [word for word in x if word not in stop_words])
```

### Q: What is lemmatization, and how is it performed?
A: Lemmatization converts words to their root forms.
```python
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
df['Resume'] = df['Resume'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
```

---

## 3. Feature Extraction
### Q: Why can’t ML models process raw text directly?
A: ML models require numerical inputs, so text must be converted into numbers.

### Q: What is TF-IDF, and why is it used?
A: TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numerical values, emphasizing important words.
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['Resume'].apply(lambda x: ' '.join(x)))
```

### Q: How do you split data into training and testing sets?
A:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 4. Machine Learning Models
### Q: Why use Logistic Regression?
A: Logistic Regression is simple and effective for text classification.
```python
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
```

### Q: Why use Random Forest?
A: Random Forest is more powerful and reduces overfitting.
```python
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
```

### Q: How do you evaluate the models?
A:
```python
from sklearn.metrics import accuracy_score, classification_report
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
```

---

## 5. Visualization
### Q: How can we visualize the most common words in resumes?
A: Using a word cloud.
```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt
text = " ".join(df['Resume'].apply(lambda x: " ".join(x)))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```

### Q: How can we visualize model performance?
A: Using a confusion matrix.
```python
import seaborn as sns
from sklearn.metrics import confusion_matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()
```

---

## 6. Final Roadmap
### **Step 1: Data Preprocessing**
- Convert text to lowercase
- Remove punctuation, stopwords
- Tokenization & lemmatization

### **Step 2: Convert Text to Numbers**
- Use **TF-IDF Vectorizer** to extract features

### **Step 3: Train Models**
- Train **Logistic Regression** & **Random Forest**

### **Step 4: Evaluate Models**
- Use **Accuracy, Classification Report, Confusion Matrix**

### **Step 5: Visualize Data**
- **Word Cloud**
- **Confusion Matrix**

---

## 7. Summary
- We **clean resumes using NLP techniques** (stopwords, lemmatization, etc.).
- We **convert text to numbers using TF-IDF**.
- We **train models (Logistic Regression & Random Forest)** to predict job categories.
- We **evaluate & visualize results** to understand performance.

Understanding the Output and Real-World Application of Resume Screening
1. What is the output of this project?
The output of this project is a classification of resumes into different categories based on the skills, experience, or domain (e.g., Data Science, Software Engineering, Marketing, etc.). The final output could be:

A predicted job category for each resume.
A ranked list of resumes based on relevance to a job description.
A match percentage showing how well a resume fits a specific role.
For example, given a resume, the model might classify it as:
✅ Category: Data Scientist
✅ Match Score: 85%

2. How can this be used in the real world?
In real-world applications, this resume screening system can be integrated into:

HR and Recruitment Tools: Automatically filtering and ranking candidates.
Applicant Tracking Systems (ATS): Helping companies shortlist candidates faster.
Job Portals: Providing smart recommendations to job seekers.
AI-Based Career Guidance: Suggesting suitable career paths based on resumes.
3. How do Logistic Regression and Random Forest work in this project?
Since resume screening is a classification problem, Logistic Regression and Random Forest are used to classify resumes into predefined categories.

Logistic Regression: A simple yet effective model that works well when the data is linearly separable.
Random Forest: A more robust model that handles complex relationships and performs well with text data when transformed into numerical features (TF-IDF, word embeddings, etc.).

Now you have a **clear roadmap** to build an NLP-based Resume Classifier! 🚀 Let me know if you have any questions. 😊

