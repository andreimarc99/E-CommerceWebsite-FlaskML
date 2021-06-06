import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

import psycopg2
import csv

query = "SELECT * from review"

con = psycopg2.connect(host = "ec2-54-163-254-204.compute-1.amazonaws.com", 
                    database = "d266vfkcf7d0ss",
                    user = "easgxuchtwrtjq",
                    password = "e93f5b108444dd37ad596ae37fc4f9007f26fb170b21f39fb387a5744383b285")
cur = con.cursor()

with open('reviews.csv', 'w', newline='') as f:
    cur.execute(query)
    writer = csv.writer(f, delimiter=',')
    for row in cur.fetchall():
        writer.writerow(row)

cur.close()
con.close()

plt.style.use("ggplot")

ratings = pd.read_csv('reviews.csv')
ratings.columns = ['review_id', 'message', 'rating', 'customer_id', 'product_id']

ratings = ratings.dropna()

ratings1 = ratings
ratings_utility_matrix = ratings1.pivot_table(values='rating', index='customer_id', columns='product_id', fill_value=0)
ratings_utility_matrix = ratings_utility_matrix.astype(int)

X = ratings_utility_matrix.T
X_sparse = csr_matrix(X)

X1 = X
SVD = TruncatedSVD(n_components=X_sparse.shape[1] - 1)
decomposed_matrix = SVD.fit_transform(X)

correlation_matrix = np.corrcoef(decomposed_matrix)

def recommend(i):
    product_names = list(X.index)
    product_ID = product_names.index(int(i))
    correlation_product_ID = correlation_matrix[product_ID]
    Recommend = list(X.index[correlation_product_ID > 0.75])
    Recommend.remove(int(i))
    return ', '.join(str(e) for e in Recommend)