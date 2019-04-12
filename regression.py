import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

data = pd.read_csv("./data_normalized.csv")
target = data.Overall
data = data.loc[:, "Crossing":"GKReflexes"]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

LR = LinearRegression()
LR.fit(X_train, y_train)
predictions = LR.predict(X_test)

print(r2_score(y_test, predictions))
print(np.sqrt(mean_squared_error(y_test, predictions)))

plt.figure(figsize=(18,10))
sns.regplot(predictions,y_test,scatter_kws={'alpha':0.3,'color':'blue'},line_kws={'color':'red','alpha':0.5})
plt.xlabel('Predicted Rating')
plt.ylabel('Actual Rating')
plt.title("Linear Prediction of Player Rating")
plt.show()