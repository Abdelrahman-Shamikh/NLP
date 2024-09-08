import string
import nltk
nltk.download('punkt')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from textblob import Word
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from joblib import dump
from joblib import load
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
nltk.download('stopwords')
from nltk.corpus import stopwords


def load_reviews(folder_path):
    """Load reviews from a specified folder."""
    reviews = []
    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        # Create the full path to the file
        file_path = os.path.join(folder_path, filename)
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            review = file.read()
            # Append the review to the list
            reviews.append(review)
    return reviews


def label_reviews(reviews, label):
    """Label the reviews with a specified label (0 for negative, 1 for positive)."""
    df = pd.DataFrame({'Review': reviews, 'Label': label})
    return df


def combine_and_shuffle(negativeR, positiveR):
    combined_reviews = pd.concat([negativeR, positiveR])
    combined_reviews = combined_reviews.sample(frac=1).reset_index(drop=True)
    return combined_reviews


negative_folder_path = r'E:\Abdelrahman\NLP\NLPproject\neg'
positive_folder_path = r'E:\Abdelrahman\NLP\NLPproject\pos'

negative_reviews = load_reviews(negative_folder_path)
positive_reviews = load_reviews(positive_folder_path)

labeled_negative_reviews = label_reviews(negative_reviews, 0)
labeled_positive_reviews = label_reviews(positive_reviews, 1)

shuffled_reviews = combine_and_shuffle(labeled_negative_reviews, labeled_positive_reviews)

####################################################################################

##Preproccessing!
shuffled_reviews['Review'] = shuffled_reviews['Review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
first_cell = shuffled_reviews.iloc[0, 0]
# print('\n'.join(textwrap.wrap(first_cell, width=130)))  # Adjust width as needed
# print("####################################################################")

stop = stopwords.words('english')
shuffled_reviews['Review'] = shuffled_reviews['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

translation_table = str.maketrans("", "", string.punctuation)
shuffled_reviews['Review'] = shuffled_reviews['Review'].apply(lambda x: x.translate(translation_table))

shuffled_reviews['Review'] = shuffled_reviews['Review'].apply(
    lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

################################################################################################
tfidf_vect = TfidfVectorizer()
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(shuffled_reviews['Review'],
                                                                      shuffled_reviews['Label'])
tfidf_vect.fit(shuffled_reviews['Review'])
train_x_TFIDF = tfidf_vect.transform(train_x)
valid_x_TFIDF = tfidf_vect.transform(valid_x)
# print(train_x_TFIDF)
# naivBayes = naive_bayes.MultinomialNB(alpha=10)
# naivBayes.fit(train_x_TFIDF, train_y)                         #model(naiive bayes)
# prediction = naivBayes.predict(valid_x_TFIDF)
# print(metrics.accuracy_score(valid_y, prediction))
###################################################


# logisticRegrresion = linear_model.LogisticRegression(max_iter=500, C=2)             #model(logistic regression)
# logisticRegrresion.fit(train_x_TFIDF, train_y)
# prediction = logisticRegrresion.predict(valid_x_TFIDF)
# print(metrics.accuracy_score(valid_y, prediction))
# dump(logisticRegrresion, 'logistic_regression_model.joblib')        #save the model!
loaded_model = load('logistic_regression_model.joblib')
# Make predictions on the validation set
predictions_valid = loaded_model.predict(valid_x_TFIDF)
accuracy = metrics.accuracy_score(valid_y, predictions_valid)
print(accuracy)
# Calculate confusion matrix for validation set
cm = confusion_matrix(valid_y, predictions_valid)

# Display the confusion matrix using sklearn's ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.title('Confusion Matrix')
plt.show()


error_rate = 1 - accuracy

# Plotting accuracy and error rate
metrics = ['Accuracy', 'Error Rate']
values = [accuracy, error_rate]

# Create a bar plot
plt.figure()
plt.bar(metrics, values, color=['blue', 'red'])

# Set title and labels
plt.title('Model Accuracy and Error Rate')
plt.ylabel('Value')
plt.ylim([0, 1])  # Set y-axis range from 0 to 1
# Display the plot
plt.show()


# -------------------------------
# Get feature names
feature_names = tfidf_vect.get_feature_names_out()

# Get coefficients from the model
coefficients = loaded_model.coef_[0]

# Create a DataFrame of feature names and their coefficients
feature_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})

# Sort by absolute value of the coefficients
feature_df = feature_df.reindex(feature_df.coefficient.abs().sort_values(ascending=False).index)

# Plot top 20 positive and negative features

top_n = 12
top_positive = feature_df.head(top_n)
# Plot the feature importance
plt.figure(figsize=(12, 8))

# Top positive features
plt.subplot(2, 1, 1)
sns.barplot(x=top_positive['coefficient'], y=top_positive['feature'], color='blue')
plt.title("Important Features")
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.show()
#--------------------------------------


############################################################
# To test the model!!
review = """
The Eternal Sunshine of the Spotless Mind" is a mesmerizing film that delves deep into the complexities of love, memory, and human connection. Directed by Michel Gondry and written by Charlie Kaufman, this unconventional love story challenges traditional narrative structures and offers a fresh perspective on the nature of relationships.

At its core, the film follows Joel Barish (played by Jim Carrey) and Clementine Kruczynski (played by Kate Winslet) as they navigate the painful aftermath of a failed relationship. When Clementine decides to undergo a procedure to erase Joel from her memory, Joel makes the impulsive decision to undergo the same procedure. As the memories of their relationship begin to fade, Joel experiences a profound sense of loss and longing, leading him on a surreal journey through his own subconscious.

What sets "Eternal Sunshine" apart is its innovative storytelling and visual style. Gondry's use of practical effects and non-linear editing creates a dreamlike atmosphere that perfectly mirrors the fragmented nature of memory. The film seamlessly weaves together moments of joy, heartbreak, and introspection, leaving a lasting impression on the viewer.

Carrey and Winslet deliver standout performances, showcasing a range and depth rarely seen in their previous work. Carrey, known for his comedic roles, brings a vulnerability to Joel that is both heartbreaking and relatable. Winslet shines as the free-spirited Clementine, capturing her character's complexity with nuance and charm.

Despite its melancholic undertones, "Eternal Sunshine" ultimately offers a message of hope and redemption. Through its exploration of love and loss, the film reminds us of the enduring power of human connection, even in the face of adversity.

Overall, "The Eternal Sunshine of the Spotless Mind" is a cinematic masterpiece that continues to resonate with audiences long after the credits roll. With its imaginative storytelling, stellar performances, and emotional depth, it's a film that truly deserves its place among the classics of modern cinema.
"""

review = review.lower()
review = " ".join(x for x in review.split() if x not in stop)
review = review.translate(translation_table)
review = " ".join([Word(word).lemmatize() for word in review.split()])

# Transform the preprocessed review
review_TFIDF = tfidf_vect.transform([review])

# Make predictions
prediction = loaded_model.predict(review_TFIDF)
print("Predicted sentiment:", prediction[0])
