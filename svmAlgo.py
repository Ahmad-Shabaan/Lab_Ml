import pandas as pd 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import svm
data = pd.read_csv("car.data")

#Data Preprocessing:
label_encoder = LabelEncoder()
data['unacc'] = label_encoder.fit_transform(data['unacc'])
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

# Create an SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

#Use the trained model to make predictions on the test data:
y_pred = clf.predict(X_test)

# evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred ,  zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
