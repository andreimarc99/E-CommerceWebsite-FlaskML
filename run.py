from flask import Flask, render_template
import flask
import numpy
from model import X, recommend
from segmentation import get_age_clustering_plot, get_gender_clustering_plot
from flask import request
import pandas as pd
from matplotlib.figure import Figure
from flask import Response
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import psycopg2
import csv

query = "SELECT * from product"

con = psycopg2.connect(host = "ec2-54-163-254-204.compute-1.amazonaws.com", 
                    database = "d266vfkcf7d0ss",
                    user = "easgxuchtwrtjq",
                    password = "e93f5b108444dd37ad596ae37fc4f9007f26fb170b21f39fb387a5744383b285")
cur = con.cursor()

with open('products.csv', 'w', newline='') as f:
    cur.execute(query)
    writer = csv.writer(f, delimiter=',')
    for row in cur.fetchall():
        writer.writerow(row)

cur.close()
con.close()

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello World'

@app.route('/reviews')
def view_reviews():
    headings=('review_id', 'message', 'rating', 'customer_id', 'product_id')
    table = pd.read_csv("reviews.csv")
    table = table.to_numpy()
    return render_template("table.html", data=table, headings=headings)

@app.route('/popular_products')
def visualize():
    products = pd.read_csv('products.csv')
    products.columns = ['product_id','description','name','number_sold','price','stock','image_id','specs_id']
    del products['description']
    del products['number_sold']
    del products['price']
    del products['stock']
    del products['image_id']
    del products['specs_id']
    ratings = pd.read_csv("reviews.csv")
    ratings.columns = ['review_id', 'message', 'rating', 'customer_id', 'product_id']

    R = pd.DataFrame(ratings.groupby('product_id')['rating'].mean())
    R = R.reset_index()
    C = R['rating'].mean()
    v = pd.DataFrame(ratings.groupby('product_id')['rating'].count())
    v = v.reset_index()
    m = 20
    
    popular_products = pd.DataFrame({'product_id': R['product_id'], 'rating': (R['rating'] * v['rating'] + C * m) / (v['rating'] + m)})
    most_popular = popular_products.sort_values('rating', ascending=False)
    most_popular.plot(kind = "bar")
    x = most_popular[['product_id']]
    y = most_popular[['rating']]

    fig = Figure()
    axis = fig.subplots()
    axis.set_xlabel('Product ID')
    axis.set_ylabel('Rating')
    axis.set_xticks(x.loc[:, 'product_id'].values)
    axis.bar(x.loc[:, 'product_id'].values,y.loc[:, 'rating'].values)
    axis.legend(products[['product_id']].values.flatten().tolist(), products[['name']].values.flatten().tolist())
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    resp = Response(output.getvalue(), mimetype='image/png')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

@app.route('/predict',methods=['GET'])
def predict():
    product = request.args['product']
    prediction = recommend(product)
    response = flask.jsonify({'predictions': prediction})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

def _labels(val):
    return round(val,2)

@app.route('/genders', methods=['GET'])
def gender_segmentation():
    customers = pd.read_csv("./data/customers.csv")
    users = pd.read_csv('./data/users.csv')
    customers = customers.merge(users, on='username', how='inner')
    g = customers['gender'].value_counts()
    genders = pd.DataFrame({'M' : [g.M], 'F': [g.F]})
    x = ['Male Percentage', 'Female Percentage']
    y = [genders.iloc[0]['M'], genders.iloc[0]['F']]

    fig = Figure()
    axis = fig.subplots()
    axis.pie(y, labels=x, autopct=_labels)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    resp = Response(output.getvalue(), mimetype='image/png')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

@app.route('/clustering/age', methods=['GET'])
def age_clustering():
    return get_age_clustering_plot()

@app.route('/clustering/gender', methods=['GET'])
def gender_clustering():
    return get_gender_clustering_plot()

if __name__ == '__main__':
    app.run(debug=True)