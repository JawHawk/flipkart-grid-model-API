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
# api_key = ''
api_key = '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'

model        = OpenAI(api_key)
prompter     = Prompter('ner.jinja')
pipe         = Pipeline(prompter , model)

def recommend(data=None):
    res = requests.get('http://localhost:5001/api/flipkart/getpurchased').json()

    item_name = ''
    item_brand = ''
    item_price = ''
    item_rating = ''

    res_name = ''
    res_brand = ''

    interested = res["interested"]
    purchased = res['purchased']

    for i in interested[:3]:
       res_name += stem(str(i["name"])) + " "
       res_brand += stem(str(i["brand"])) + " "
    
    for j in purchased[:3]:
       res_name = stem(str(j["name"]) + res_name) + " "
       res_brand = stem(str(j["brand"]) + res_brand) + " "
    
    if(data != None):
        item_name += data.get('Labels', data.get('Name', ''))
        item_brand += data.get('Brand', '').replace(" ", "")
        item_price += str(data.get('Price', '')).replace(",", "")
        item_rating += str(data.get('Rating', ''))

    item_name = stem(item_name)
    item_brand = item_brand.lower() 

    new_fk = clean_fk.copy(deep=True)
    new_fk = pd.concat([new_fk, pd.DataFrame([{'Brand': item_brand, 'Name': item_name, 'Price': item_price, 'Rating': item_rating}])], ignore_index=True)

    name_vec = cvv.fit_transform(new_fk['Name']).toarray()
    brand_vec = cvv.fit_transform(new_fk['Brand']).toarray()
    price_vec = cvv.fit_transform(new_fk['Price']).toarray()

    name_similarity = cosine_similarity(name_vec)
    brand_similarity = cosine_similarity(brand_vec)
    price_similarity = cosine_similarity(price_vec)

    overall_similarity = name_similarity[-1] + brand_similarity[-1] 

    recom_indices = sorted(range(len(overall_similarity)), key=lambda i: overall_similarity[i], reverse=True)[1:31]
    
    recommendations = [{'Brand': fk.iloc[idx]['Brand'], 'Image': fk.iloc[idx]['Image'], 'Name':  fk.iloc[idx]['Name'], 'Rating':  fk.iloc[idx]['Rating'], 'BreadCrumbs':  fk.iloc[idx]['BreadCrumbs'], 'Reviews':  fk.iloc[idx]['Reviews'], 'Price':  fk.iloc[idx]['Price'], 'Index': idx} for idx in recom_indices]
    return recommendations

@app.route('/recommend', methods=['GET'])
@cross_origin(supports_credentials=True)
def recommend_route(data=None):
    return jsonify(recommend())

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

if __name__ == '__main__':
    app.run(debug=True)
