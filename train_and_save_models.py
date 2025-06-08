import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from madaline import MADALINE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk


# Ensure the necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
data = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")

# List of columns to remove
columns_to_remove = ['number of reviews','Title','Clothing ID', 'Age', 'Division Name', 'Department Name', 'Class Name']

# Drop the specified columns
data.drop(columns_to_remove, axis=1, inplace=True)

# Preprocessing the Review Text column
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        stemmer = SnowballStemmer(language='english')
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
        processed_text = ' '.join(stemmed_tokens)
        return processed_text
    else:
        return ""

# Apply preprocessing to the Review Text column
data['Review Text'] = data['Review Text'].apply(preprocess_text)

# Define features (X) and target (y)
X = data['Review Text']
y = data['Rating'].apply(lambda x: 1 if x > 3 else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=2000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_tfidf, y_train)

# Train a Stochastic Gradient Descent (SGD) model
sgd_model = SGDClassifier()
sgd_model.fit(X_train_tfidf, y_train)

# Train a Support Vector Machine (SVM) model
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)

# Train a K-Nearest Neighbors (KNN) model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_tfidf, y_train)

# Train a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# Train a Decision Tree classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_tfidf, y_train)

# Train a Gradient Boosting classifier
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train_tfidf, y_train)

# Train a Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Train a perceptron model
perceptron_model = Perceptron()
perceptron_model.fit(X_train_tfidf, y_train)

# Train an MLP feedforward neural network
mlp_feedforward_model = MLPClassifier(hidden_layer_sizes=(200,100,50), max_iter=1000, solver='adam')
mlp_feedforward_model.fit(X_train_tfidf, y_train)

# Train an MLP with backpropagation
mlp_backpropagation_model = MLPClassifier(hidden_layer_sizes=(200,100,50), max_iter=1000)
mlp_backpropagation_model.fit(X_train_tfidf, y_train)

# Train an Adaline model
adaline_model = SGDClassifier(loss='perceptron', eta0=0.1, learning_rate='constant', penalty=None)
adaline_model.fit(X_train_tfidf, y_train)

# Train a MADALINE model
madaline_model = MADALINE(input_size=X_train_tfidf.shape[1])
madaline_model.train(X_train_tfidf.toarray(), y_train.to_numpy(), learning_rate=0.1, epochs=100)

# Save the trained models and vectorizer
joblib.dump(logistic_model, 'logistic_model.pkl')
joblib.dump(sgd_model, 'sgd_model.pkl')
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(dt_model, 'dt_model.pkl')
joblib.dump(gb_model, 'gb_model.pkl')
joblib.dump(nb_model, 'nb_model.pkl')
joblib.dump(perceptron_model, 'perceptron_model.pkl')
joblib.dump(mlp_feedforward_model, 'mlp_feedforward_model.pkl')
joblib.dump(mlp_backpropagation_model, 'mlp_backpropagation_model.pkl')
joblib.dump(adaline_model, 'adaline_model.pkl')
joblib.dump(madaline_model, 'madaline_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Function to print evaluation metrics
def print_evaluation(model_name, accuracy, y_test, y_pred):
    print(f"\n{model_name} Model:")
    print("Accuracy:", int(accuracy * 100), '%')
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Predictions and evaluations
models = {
    "Logistic Regression": logistic_model,
    "SGD": sgd_model,
    "SVM": svm_model,
    "KNN": knn_model,
    "Random Forest": rf_model,
    "Decision Tree": dt_model,
    "Gradient Boosting": gb_model,
    "Naive Bayes": nb_model,
    "Perceptron": perceptron_model,
    "MLP Feedforward": mlp_feedforward_model,
    "MLP Backpropagation": mlp_backpropagation_model,
    "Adaline": adaline_model,
    "MADALINE": madaline_model
}

for model_name, model in models.items():
    if model_name == "MADALINE":
        y_pred = model.predict(X_test_tfidf.toarray())
    else:
        y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print_evaluation(model_name, accuracy, y_test, y_pred)

# Confusion Matrix Visualization
def plot_confusion_matrix(cm, labels, title):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(title)
    plt.show()

for model_name, model in models.items():
    if model_name == "MADALINE":
        y_pred = model.predict(X_test_tfidf.toarray())
    else:
        y_pred = model.predict(X_test_tfidf)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels=["Negative", "Positive"], title=f"Confusion Matrix - {model_name}")