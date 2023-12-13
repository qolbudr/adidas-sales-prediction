from flask import Flask, Response, request, render_template
import numpy as np
import pandas as pd
import os
import joblib

app = Flask(__name__)

def parse_data():
    df=pd.read_excel('data.xlsx')
    df = df.drop('Unnamed: 0',axis=1)
    df = df.drop(labels=[0,1,2], axis=0)
    df.columns = df.iloc[0]
    df = df.drop(labels=3, axis=0)
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
    return df

def parse_data_analyze(df):
    df2=df.copy()
    df2['Region']=pd.factorize(df2.Region)[0]
    df2['State']=pd.factorize(df2.State)[0]
    df2['City']=pd.factorize(df2.City)[0]
    df2['Product']=pd.factorize(df2.Product)[0]
    df2['Retailer']=pd.factorize(df2.Retailer)[0]

    # mengganti nama kolom dalam DataFrame df2 dari 'Sales Method' menjadi 'Method'
    df2.rename(columns = {'Sales Method':'Method'}, inplace = True)

    df2['Method']=pd.factorize(df2.Method)[0] # mengubah nilai-nilai dalam kolom 'Method' dari DataFrame df2 menjadi indeks numerik yang sesuai.
    df2 = df2.drop('Retailer ID',axis=1) # untuk menghapus kolom 'Retailer ID' dari DataFrame df2
    df2 = df2.drop('Invoice Date',axis=1) # untuk menghapus kolom 'invoice date' dari DataFrame df2

    df2.head() # ntuk melihat contoh data dalam DataFrame

    # mengubah tipe data (data type) dari beberapa kolom dalam DataFrame df2 menjadi tipe data integer (int).
    df2['Units Sold'] = df2['Units Sold'].astype(int)
    df2['Total Sales'] = df2['Total Sales'].astype(int)
    df2['Operating Profit'] = df2['Operating Profit'].astype(int)
    df2['Retailer'] = df2['Retailer'].astype(int)
    return df2


@app.route("/")
def home():
    return render_template('home.html')

@app.route("/api/product_menu")
def get_product_menu():
    selected_month = request.args.get('month')
    
    df=parse_data()

    #select where month
    df = df[df['Invoice Date'].dt.to_period('M') == selected_month]

    result = df[['Product','Price per Unit']].drop_duplicates('Product')
    result = result.rename(columns={"Product": "product", "Price per Unit": "price"})
    return Response(result.to_json(orient="records"), mimetype='application/json')

@app.route("/api/unit_sold")
def unit_sold():
    selected_month = request.args.get('month')
    
    df=parse_data()

    #select where month
    df = df[df['Invoice Date'].dt.to_period('M') == selected_month]

    result = df[['Product','Units Sold']].groupby('Product', as_index=False).sum()
    result = result.rename(columns={"Product": "product", "Units Sold": "unit"})
    return Response(result.to_json(orient="records"), mimetype='application/json')

@app.route("/api/total_sales")
def total_sales():
    selected_month = request.args.get('month')
    
    df=parse_data()

    #select where month
    df = df[df['Invoice Date'].dt.to_period('M') == selected_month]

    result = df[['Product','Total Sales']].groupby('Product', as_index=False).sum()
    result = result.rename(columns={"Product": "product", "Total Sales": "total_sales"})
    return Response(result.to_json(orient="records"), mimetype='application/json')    

@app.route("/api/predict")
def predict():
    selected_month = request.args.get('month')
    ascending = False if request.args.get('ascending') == "false" else True
    
    df=parse_data()

    #select where month
    df = df[df['Invoice Date'].dt.to_period('M') == selected_month]
    df2 = parse_data_analyze(df)

    X = df2.values[:,(0,1,2,3,4,5,6,8,9,10)]
    Y = df2.values[:, 7]

    lr = joblib.load('model.pkl')

    df['sales_prediction'] = lr.predict(X)
    df['Invoice Date'] = df['Invoice Date'].dt.strftime('%m/%d/%Y')
    result = df[['Retailer', 'Region', 'State', 'City', 'Product', 'sales_prediction', 'Invoice Date']]
    result = result.rename(columns={"Retailer": "retailer", "Region": "region", "State": "state", "City": "city", "Product": "product", "Invoice Date": "invoice_date"})
    
    result = result.sort_values(by=['sales_prediction'], ascending=ascending)

    return Response(result.to_json(orient="records"), mimetype='application/json')    