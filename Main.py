import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the iris dataset
iris = datasets.load_iris()

# Convert to a Pandas dataframe
df = pd.DataFrame(data= iris['data'], columns= iris['feature_names'])

# Train the KMeans model
kmeans = KMeans(n_clusters=3)
kmeans.fit(df)

# Predict the cluster labels for each sample
labels = kmeans.predict(df)

# Visualize the results using a scatter plot
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'], c=labels)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()
