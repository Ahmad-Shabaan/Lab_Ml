import pandas as pd  
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("car.data")

label_encoder = LabelEncoder()
for column in data.columns:
    data[column] = label_encoder.fit_transform(data[column])


#Split Data into Training and Testing Sets:
X = data.drop('unacc', axis=1)
y = data['unacc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Create a KNN classifier with k=3 (you can adjust k as needed)
k = 3
clf = KNeighborsClassifier(n_neighbors=k)

clf.fit(X_train, y_train)

#Use the trained model to make predictions on the test data:
y_pred = clf.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


