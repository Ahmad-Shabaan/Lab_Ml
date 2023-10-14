import pandas as pd  # powerful data manipulation and analysis library  & for data processing
from sklearn.model_selection import train_test_split  # sklearn ,  for building, training, and evaluating machine learning models. 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


#In summary, Pandas is used for data manipulation and preprocessing, 
# Scikit-Learn is used for machine learning model training and evaluation,
#  Matplotlib is used for data visualization, 
# and the Car Evaluation dataset is used to practice classification tasks, such as predicting the acceptability of cars based on their features.

data = pd.read_csv("car.data")
# data.info() ;

#Data Preprocessing:
label_encoder = LabelEncoder()
data['unacc'] = label_encoder.fit_transform(data['unacc'])
# to solve this error :  could not convert string to float: 'low'
data['vhigh'] = label_encoder.fit_transform(data['vhigh'])
data['vhigh.1'] = label_encoder.fit_transform(data['vhigh.1'])
data['2.1'] = label_encoder.fit_transform(data['2.1'])
data['2'] = label_encoder.fit_transform(data['2'])
data['low'] = label_encoder.fit_transform(data['low'])
data['small'] = label_encoder.fit_transform(data['small'])


#Split Data into Training and Testing Sets:
X = data.drop('unacc', axis=1)
y = data['unacc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# first algo : Decision Tree Classifier for classification
clf = DecisionTreeClassifier(class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

#Use the trained model to make predictions on the test data:
y_pred = clf.predict(X_test)



# evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))







