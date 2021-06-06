from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans
import os
import datetime as DT
from datetime import datetime, date
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask import Response
import io
import psycopg2
import csv

customers_query = "SELECT * from customers"
users_query = "SELECT * from users"
orders_query = "SELECT * from orders"
order_products_query = "SELECT * from order_products"

con = psycopg2.connect(host = "ec2-54-163-254-204.compute-1.amazonaws.com", 
                    database = "d266vfkcf7d0ss",
                    user = "easgxuchtwrtjq",
                    password = "e93f5b108444dd37ad596ae37fc4f9007f26fb170b21f39fb387a5744383b285")
cur = con.cursor()

with open('customers.csv', 'w', newline='') as f:
    cur.execute(customers_query)
    writer = csv.writer(f, delimiter=',')
    for row in cur.fetchall():
        writer.writerow(row)

with open('users.csv', 'w', newline='') as f:
    cur.execute(users_query)
    writer = csv.writer(f, delimiter=',')
    for row in cur.fetchall():
        writer.writerow(row)

with open('orders.csv', 'w', newline='') as f:
    cur.execute(orders_query)
    writer = csv.writer(f, delimiter=',')
    for row in cur.fetchall():
        writer.writerow(row)

with open('order_products.csv', 'w', newline='') as f:
    cur.execute(order_products_query)
    writer = csv.writer(f, delimiter=',')
    for row in cur.fetchall():
        writer.writerow(row)

cur.close()
con.close()

customers = pd.read_csv('customers.csv')
customers.columns = ['customer_id', 'date_joined', 'username']

users = pd.read_csv('users.csv')
users.columns = ['username', 'birth_date', 'cnp', 'first_name', 'gender', 'last_name', 'password', 'role']
customers = customers.merge(users, on='username', how='inner')
del customers['password']

customers['birth_date'] = pd.to_datetime(customers['birth_date']) 
customers['date_joined'] = pd.to_datetime(customers['date_joined']) 

def age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

customers['age'] = customers['birth_date'].apply(age)

orders = pd.read_csv('orders.csv')
orders.columns = ['order_id', 'delivered', 'final_price', 'cart_id', 'voucher_id', 'address_id', 'customer_id']
order_products = pd.read_csv('order_products.csv')
order_products.columns = ['order_id', 'product_id']
orders = orders[['order_id', 'final_price', 'customer_id']]

orders = orders.merge(order_products, on='order_id')

a = orders.groupby('customer_id').sum()['final_price']
a = a.reset_index()
a = a.rename(columns = {'final_price' : 'total_spent'})

customers = customers.merge(a, on='customer_id', how='inner')

del customers['cnp']
del customers['first_name']
del customers['last_name']
del customers['role']
del customers['date_joined']
del customers['username']
del customers['birth_date']

customers['gender'] = customers['gender'].map({'M': 0,'F': 1})
x=customers[['age','total_spent']]

wcss=[]
for i in range(1,10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

kmeans = KMeans(5)
kmeans.fit(x)

identified_clusters=kmeans.fit_predict(x)
table_with_clusters=customers.copy()
table_with_clusters['clusters']=identified_clusters

def get_age_clustering_plot():
    fig = Figure()
    axis = fig.subplots()
    axis.scatter(table_with_clusters['age'], table_with_clusters['total_spent'], c=table_with_clusters['clusters'], cmap='rainbow')
    axis.set_xlabel('Age')
    axis.set_ylabel('Total spent ($)')
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    resp = Response(output.getvalue(), mimetype='image/png')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

y=customers[['gender','total_spent']]
kmeans2 = KMeans(5)
kmeans2.fit(y)
clusters_y=kmeans2.fit_predict(y)
clusters_table_y=customers.copy()
clusters_table_y['clusters']=clusters_y

def get_gender_clustering_plot():
    fig = Figure()
    axis = fig.subplots()
    axis.scatter(clusters_table_y['gender'], clusters_table_y['total_spent'], c=clusters_table_y['clusters'], cmap='rainbow')
    axis.set_xlabel('Gender')
    axis.set_ylabel('Total spent ($)')
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    resp = Response(output.getvalue(), mimetype='image/png')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp