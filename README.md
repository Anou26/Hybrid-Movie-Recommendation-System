# 🎬 **Movie Recommendation System (Hybrid Filtering)**  
A **Hybrid Movie Recommendation System** using **Content-Based Filtering, Collaborative Filtering (KNN, SVD, SVD++), and a Popularity-Based Model** to provide personalized movie suggestions.  

---

## **🛠️ Tech Stack**
- **Programming Language**: Python 🐍  
- **Machine Learning & AI**: Scikit-learn, Surprise Library  
- **Data Handling**: Pandas, NumPy  
- **Recommendation Techniques**:  
  - Content-Based Filtering  
  - Collaborative Filtering (KNN, SVD, SVD++)  
  - Matrix Factorization  
  - Hybrid Approach  
- **Evaluation Metrics**: RMSE, MAE, Precision, Recall  
- **Dataset**: **MovieLens 100K**  

---

## **📜 Project Overview**
Modern recommendation systems are essential for reducing **information overload**. This project develops a **Hybrid Recommendation System** based on **MovieLens dataset**, integrating multiple approaches to provide **better accuracy and diverse recommendations**.  

### 🔍 **Key Approaches Used**  
✅ **Content-Based Filtering** → Recommends movies based on genre similarity.  
✅ **Collaborative Filtering** → Uses user-user & item-item similarity (KNN, SVD, SVD++).  
✅ **Matrix Factorization** → Predicts missing ratings using **Singular Value Decomposition (SVD)**.  
✅ **Popularity Model** → Suggests trending movies.  
✅ **Hybrid Model** → Combines all methods for better accuracy.  

---

## **📂 Folder Structure**
```
📦 Movie-Recommendation-System
│── 📜 .gitignore         # Ignore unnecessary files
│── 📜 README.md          # Project documentation (this file)
│── 📜 requirements.txt   # Dependencies for the project
│── 📂 data               # Dataset (MovieLens)
│   ├── movies.csv        # Movie metadata
│   ├── ratings.csv       # User ratings
│── 📂 models
│   ├── content_based.py  # Content-based filtering implementation
│   ├── collaborative.py  # Collaborative filtering (KNN, SVD, SVD++)
│   ├── hybrid.py         # Hybrid recommendation system
│── 📂 notebooks
│   ├── EDA.ipynb         # Exploratory Data Analysis
│   ├── Model_Training.ipynb  # Training & evaluation notebook
│── 📂 utils
│   ├── data_loader.py    # Functions for loading datasets
│   ├── metrics.py        # RMSE, MAE, Precision-Recall functions
│── 📜 main.py            # Script to run recommendations
```

---

## **📊 Dataset**
I used the **MovieLens 100K Dataset**, containing:  
📌 **100,836 ratings**  
📌 **10,000 movies**  
📌 **3,700 tags**  
📌 Features include: **Title, Genre, Ratings, Popularity, Year of Release**  

### **Preprocessing Steps**  
✔️ Remove unnecessary columns  
✔️ Convert categorical features (e.g., genres) to vectors  
✔️ Split data (80% training, 20% testing)  
✔️ Handle missing values  

---

## **🤖 Implemented Models**
### 1️⃣ **Content-Based Filtering**
- Generates recommendations based on **genre similarity**.  
- Uses **TF-IDF Vectorization** to build a movie profile.  

```python
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(movie_features)
```

### 2️⃣ **Collaborative Filtering**
- Uses **User-User & Item-Item Similarity** (KNN).  
- Implements **Singular Value Decomposition (SVD, SVD++)** for rating predictions.  

```python
from surprise import SVD
algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)
```

### 3️⃣ **Hybrid Model**
- Combines **content-based + collaborative filtering** to improve accuracy.  
- Uses **weighted blending of models** to generate final recommendations.  

---

## **📈 Model Performance & Results**
Evaluation Metrics: **RMSE, MAE, Precision, Recall**  
✅ **Content-Based**: MAE = 0.7079, RMSE = 0.9185  
✅ **Collaborative Filtering (KNN)**: Lower error than content-based  
✅ **Matrix Factorization (SVD, SVD++)**: Best accuracy, but slower  
✅ **Hybrid Model**: **Most accurate, balanced recommendations**  

---

## **🎯 Features**
✔️ **Personalized Recommendations**  
✔️ **Cold-Start Problem Solution**  
✔️ **Improved Accuracy using Hybrid Approach**  
✔️ **Matrix Factorization for Better Performance**  
✔️ **Optimized Training with Grid Search**  

---

## **📌 Future Enhancements**
🚀 **Deep Learning (Neural Networks) for recommendations**  
🚀 **Larger dataset integration**  
🚀 **Improving cold-start solutions**  
🚀 **Adding user feedback loop**  
