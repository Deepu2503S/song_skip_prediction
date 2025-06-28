import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix

df = pd.read_csv(r"C:\Users\bsbde\OneDrive\Desktop\Machine Learning\Practise Projects\Spotify Skip\tf_mini.csv")
print(df.describe())
print("Skips : " , df['skip_30s'].value_counts())

"""sns.countplot(x='skip_30s',data=df)
plt.title("Skips vs No Skips")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()"""

# Defining features and target

X = df.drop('skip_30s',axis=1)
Y = df['skip_30s']

scaler = StandardScaler()

# Standardization of data

X_scaled = scaler.fit_transform(X)

#Splitting data

x_tr , x_ts , y_tr , y_ts = train_test_split(X_scaled,Y,test_size=0.2,random_state=42)

#Training the model

model = LogisticRegression()
model.fit(x_tr,y_tr)

#Testing the model

y_prd = model.predict(x_ts)

print(f"The accuracy is {accuracy_score(y_ts,y_prd)*100}%")

#This report gives a brief detailing of how good actually my model is.
print("Classification Report:\n",classification_report(y_ts,y_prd))

#Confusion matrix ith row and jth column tells how much of i's were predicted as j's
print("Confusion Matrix : \n",confusion_matrix(y_ts,y_prd))