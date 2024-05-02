### **"AIProject_ResumeAnalyzer"**

**Project Submitted by Jhansi Sreya Jagarapu and Sathwik Sarangam**

**YouTube Link :** https://youtu.be/wOr9IQtB03s

Resume Analyzer is a comprehensive tool designed to enhance your job application process. By aligning your skills with those outlined in the job description, it offers valuable insights into compatibility. Leveraging synonyms and related terms, it ensures thorough skill extraction for a comprehensive analysis of your qualifications. Additionally, it evaluates your resume's quality through a scoring system, pinpointing areas for improvement and optimization. With recommendations on suitable job roles tailored to your skills, it guides you towards positions where success is likely. Furthermore, Resume Analyzer suggests the optimal resume format for each job role, effectively showcasing your qualifications. Finally, it provides quantifiable achievements, offering a clear picture of the number of skills matched to the job description.

### **Datasets used**

**1. modified_jobs.csv :** Processed job list which contains fields such as JobId, Work Type, Job Title, Experience, Qualifications, Salary Range, location, Country, Role, Job Description.
**2. combined_skills.csv :** Dataset comprising a comprehensive list of skills gathered from various sources or datasets.
**3. resumesfolder :** Collection of normal as well as infographic resumes, serving as input for analysis and matching against job descriptions.
**4. resume.pdf :** A resume which is used to match with job list

### **Algorithms used**

**1. Text Processing and Analysis:**
    Tokenization,
    Stemming,
    Stopwords Removal,
    TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization,
    Cosine Similarity
**2. Machine Learning Algorithms:**
    K-Nearest Neighbors (KNN) Classifier,
    Support Vector Machine (SVM) Classifier
**3. Evaluation Metrics:**
    Accuracy,
    Precision,
    Recall,
    F1-Score
**4. Data Visualization:**
    Matplotlib for plotting graphs and visualizations


### **Tools used in the project**

**1. PyPDF2:** For PDF file manipulation.
**2. ftfy:** For fixing text encoding issues.
**3. NLTK:** For natural language processing tasks.
**4. spaCy:** For advanced natural language processing tasks.
**5. Gensim:** For topic modeling and document similarity.
**6. scikit-learn:** For machine learning tasks such as text vectorization and classification.
**7. pandas:** For data manipulation and analysis.
**8. NumPy:** For numerical computing.

### **Tools you can use to run the project**

1. Google Colab (Google Collaboratory)
2. Jupyter Notebook
3. Visual Studio Code (VS Code)
4. PyCharm
5. Spyder
6. Atom
7. Anaconda
8. IDLE (Integrated Development and Learning Environment)
9. Python interpreter in the command line


### **Setup & Installation**

**a. We need to download github code using link**
https://github.com/jhansisreya11/AIProject_ResumeAnalyzer.git

**b. Packages which must be downloaded**

1. PyPDF2 is a Python library used for reading, manipulating, and extracting data from PDF files.
   !pip install PyPDF2

2. ftfy is a Python library designed to fix Unicode text that's been damaged or has encoding issues.
   !pip3 install ftfy

3. This command downloads NLTK's stopwords corpus, containing commonly filtered words used in text processing tasks.
   nltk.download('stopwords')

4. This command downloads the 'punkt' tokenizer models from NLTK, facilitating tokenization of text into individual words or sentences.
   nltk.download('punkt')

5. It initializes an NLP pipeline using spaCy's pre-trained English model for small-scale applications.
   nlp = spacy.load("en_core_web_sm")

6. It loads pre-trained word vectors from the GloVe model trained on Wikipedia and Gigaword data, enabling semantic representations of words in a 300-dimensional vector space."
   word_vectors = api.load("glove-wiki-gigaword-300")

7. It imports various libraries for natural language processing and machine learning tasks, including NLTK, CSV, os, regular expressions, PyPDF2, spaCy, NumPy, pandas, Gensim's API for downloading pre-trained models, and ftfy for fixing text encoding issues. Additionally, it imports classes and functions from scikit-learn for feature extraction, model evaluation, and classification tasks.
    import nltk
    import csv
    import os
    import re
    import PyPDF2
    import spacy
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import gensim.downloader as api
    from collections import Counter
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    from sklearn.svm import SVC
    from ftfy import fix_text

### **Usage**

To use this project

1. Make sure you have all the necessary dependencies installed as described in the "Setup & Installation" section.
2. Clone or download the project repository from GitHub.
3. Prepare combined_skills.csv, modified_jobs.csv, one single resume pdf and a set of resumes folder 
4. Run the provided Python scripts in your preferred environment (e.g., Jupyter Notebook, Python IDE, or command line).
5. Follow the prompts or input the necessary paths/files as required by the scripts.
6. Review the output for insights, recommendations, and matches.

### **Evaluation Methodology**

1. Data Collection and Preparation
2. Feature Extraction
3. Extraction of skills 
4. Using related words or synonyms for skills extraction
5. Matching percentage or matching confidence
6. Using quantifiable achievements for KNN and SVM algorithm implementation
7. Listing of jobs suitable for Resume
8. Predicting accuracy, precision, recall and F1 score

### **Results** 

### **1. Skills extracted**
![image-4](https://github.com/jhansisreya11/AIProject_ResumeAnalyzer/assets/162237187/1aee75c6-2792-4039-b93c-c921a6582588)

### **2. Display of top matches of job with resume and their percentage match, total skills count and matched skills count**
![image-5](https://github.com/jhansisreya11/AIProject_ResumeAnalyzer/assets/162237187/42389444-01a5-4b38-aee5-0072ca3e98ed)

### **3. Output of matching each resume with its highly matched jobs using KNN Algorithm**
![image-6](https://github.com/jhansisreya11/AIProject_ResumeAnalyzer/assets/162237187/4a34b231-dcdb-4356-8ab3-39f5370896b0)

### **4. Accuracy, precision, recall and F1 Score of KNN Classifier**
![image-8](https://github.com/jhansisreya11/AIProject_ResumeAnalyzer/assets/162237187/0d8c3fe5-972c-48c6-9de2-aee8a64c938b)

### **5. Accuracy, precision, recall and F1 Score of SVM Classifier**
![image-7](https://github.com/jhansisreya11/AIProject_ResumeAnalyzer/assets/162237187/34b77899-5530-4650-b901-f719dc1c7901)

### **6. Pie Chart between job title and percentage match**
![image](https://github.com/jhansisreya11/AIProject_ResumeAnalyzer/assets/162237187/d518a5aa-dd28-491e-93cd-9529fed975a0)

### **7. Bar Chart between job title and total skills count**
![image-1](https://github.com/jhansisreya11/AIProject_ResumeAnalyzer/assets/162237187/28f290b0-fe99-4dc3-afab-3625a71c71d4)

### **8. Bar Chart between job title and matched skills count**
![image-2](https://github.com/jhansisreya11/AIProject_ResumeAnalyzer/assets/162237187/67fe5263-9d23-4124-ae03-49e4e63c96a6)

### **9. Bar Chart between job title and percentage match**
![image-3](https://github.com/jhansisreya11/AIProject_ResumeAnalyzer/assets/162237187/603ebcee-1c91-4d9d-a2e3-376f1d6de2a9)


























# AIProject_ResumeAnalyzer
