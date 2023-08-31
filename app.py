from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd  
import nltk
from nltk.stem.porter import PorterStemmer
from promptify import Prompter,OpenAI, Pipeline
import random
from flask_cors import CORS, cross_origin
import requests
import numpy as np
import os


ps = PorterStemmer()
def stem(text):
  y = []

  for i in text.split():
    i = i.replace("'", "")
    y.append(ps.stem(i))

  return " ".join(set(y))


app = Flask(__name__)
CORS(app, support_credentials=True)

fk = pd.read_csv('cleanFlipkartData.csv')
fk = fk.drop_duplicates()
fk['Price'] = fk['Price'].apply(lambda x: str(x))
fk['Rating'] = fk['Rating'].apply(lambda x: str(x))

clean_fk = fk.copy(deep=True)
clean_fk['Brand'] = clean_fk['Brand'].apply(lambda x: x.replace(" ", "").lower())

clean_fk['Name'] = clean_fk['Name'] + clean_fk['BreadCrumbs']
clean_fk['Name'] = clean_fk['Name'].apply(stem)

cvv = TfidfVectorizer(max_features=5000, stop_words="english")

# OPENAI vars
api_key = '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'

model        = OpenAI(api_key)
prompter     = Prompter('ner.jinja')
pipe         = Pipeline(prompter , model)

def recommend(data=None):

    item_name = ''
    item_brand = ''
    item_price = -1
    item_rating = ''
    
    if(data != None):
        item_name = data.get('Labels', data.get('Name', ''))
        item_brand = data.get('Brand', '').replace(" ", "")
        item_price = int(data.get('Price', -1))
        item_rating = str(data.get('Rating', ''))

    item_name = stem(item_name)
    item_brand = item_brand.lower() 

    new_fk = clean_fk.copy(deep=True)
    new_fk = pd.concat([new_fk, pd.DataFrame([{'Brand': item_brand, 'Name': item_name, 'Price': item_price, 'Rating': item_rating}])], ignore_index=True)

    name_vec = cvv.fit_transform(new_fk['Name']).toarray()
    brand_vec = cvv.fit_transform(new_fk['Brand']).toarray()

    price_vec = []
    if (item_price != -1):
        for i in new_fk['Price'].tolist():
            price_vec.append(abs(1 - (abs(item_price - int(i)) / int(i))))

    name_similarity = cosine_similarity(name_vec)
    brand_similarity = cosine_similarity(brand_vec)

    overall_similarity = name_similarity[-1] + brand_similarity[-1] 

    if item_price != -1:
        overall_similarity += np.array(list(map(lambda x: x * 0.4, price_vec)))[-1]
    
    recom_indices = sorted(range(len(overall_similarity)), key=lambda i: overall_similarity[i], reverse=True)[1:31]
    
    recommendations = [{'Brand': fk.iloc[idx]['Brand'], 'Image': fk.iloc[idx]['Image'], 'Name':  fk.iloc[idx]['Name'], 'Rating':  fk.iloc[idx]['Rating'], 'BreadCrumbs':  fk.iloc[idx]['BreadCrumbs'], 'Reviews':  fk.iloc[idx]['Reviews'], 'Price':  fk.iloc[idx]['Price'], 'Index': idx} for idx in recom_indices]
    return recommendations

def recommend2():
    result = requests.get(url='https://flipkart-backend-qn4j.onrender.com/api/flipkart/getpurchased') 
    res = result.json()

    recommend_indexes = []

    interested = res["interested"]
    purchased = res['purchased']

    new_fk = clean_fk.copy(deep=True)

    k = []

    for i in interested:
        try:
            res_name = stem(str(i["name"]) + str(i["breadcrumbs"])) 
            res_brand = stem(str(i["brand"]).lower())
            res_price = i["price"]
            if res_name not in k:
                k.append(res_name)
            new_fk = pd.concat([new_fk, pd.DataFrame([{'Brand': res_brand, 'Name': res_name, 'Price': res_price, 'Rating': ''}])], ignore_index=True)
        except:
            pass
    for j in purchased:
        try:
            res_name = stem(str(j["name"]) + str(j["breadcrumbs"]))
            res_brand = stem(str(j["brand"]).lower())
            res_price = j["price"]
            if res_name not in k:
                k.append(res_name)
            new_fk = pd.concat([new_fk, pd.DataFrame([{'Brand': res_brand, 'Name': res_name, 'Price': res_price, 'Rating': ''}])], ignore_index=True)
        except:
            pass

    name_vec = cvv.fit_transform(new_fk['Name']).toarray()
    brand_vec = cvv.fit_transform(new_fk['Brand']).toarray()

    name_similarity = cosine_similarity(name_vec)
    brand_similarity = cosine_similarity(brand_vec)

    for i in range(1, len(interested)+len(purchased) + 1):
        overall_similarity = name_similarity[-i] + brand_similarity[-i] 
        t = sorted(range(len(clean_fk)), key=lambda i: overall_similarity[i], reverse=True)[3]
        if t not in recommend_indexes: 
            recommend_indexes.append(t)
    
    for i in range(15 - len(recommend_indexes)):
        recommend_indexes.append(random.randint(1, 4400))

    recom = [{"Brand": fk.iloc[idx]['Brand'], "Image": fk.iloc[idx]['Image'], "Name":  fk.iloc[idx]['Name'], "Rating":  fk.iloc[idx]['Rating'], "BreadCrumbs":  fk.iloc[idx]['BreadCrumbs'], "Reviews":  fk.iloc[idx]['Reviews'], "Price":  fk.iloc[idx]['Price'], "Index": idx} for idx in recommend_indexes]
    return recom

@app.route('/recommend', methods=['GET'])
@cross_origin(supports_credentials=True, origins=["*"])
def recommend_route(data=None):
    try:
        data = jsonify(recommend2())
        return data
    except:
        return "error", 500

@app.route('/chat', methods=['POST'])
@cross_origin(supports_credentials=True)
def chat_route():
    item = request.json
    message = item.get('message', None)

    if message is None:
       return "Input Message can't be null", 400
    
    example1 = ["I am a 20 year old woman looking for a Diwali outfit in Mumbai",
    [
        {'E': 'Labels', 'W': 'women'},
        {'E': 'Labels', 'W': 'saree'},
        {'E': 'Labels', 'W': 'kurta'},
    ]
    ]

    example2 = ["boys black tshirts around 500 and nike",
    [
        {'E': 'Labels', 'W': 'men'},
        {'E': 'Labels', 'W': 'boys'},
        {'E': 'Labels', 'W': 'black'},
        {'E': 'Labels', 'W': 'tshirts'},
        {'E': 'Brand', 'W': 'nike'},
        {'E': 'Price', 'W': '500'}
    ]
    ]
    result = pipe.fit(message,
                  domain  = 'fashion',
                  examples = [example1, example2],
                  description = "Below paragraph is the input from a user. The paragraph may describe some information about user and about the clothes and fashion items he or she is interested to buy.",
                  labels  = ['Brand', 'Labels', 'Price'])
    
    try:
        res = result[0].get('parsed').get('data').get('completion')
        data = {}
        for i in res[0]:
           try:
              data[i['E']] = data.get(i['E'], '') + ' ' + str(i['W'])
           except:
              print(i)

        resp = recommend(data)
        return jsonify(resp[:5])
    except Exception as e:
        print(e)
        return "Error", 400

@app.route('/', methods=['GET'])
@cross_origin(supports_credentials=True, origins=["*"])
def ko():
    return "Home"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
