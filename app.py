from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd  
import nltk
from nltk.stem.porter import PorterStemmer
from promptify import Prompter,OpenAI, Pipeline
import random

app = Flask(__name__)

fk = pd.read_csv('cleanFlipkartData.csv')
clean_fk = fk.drop_duplicates()
clean_fk['Brand'] = clean_fk['Brand'].apply(lambda x: x.replace(" ", ""))
clean_fk['Rating'] = clean_fk['Rating'].apply(lambda x: str(x))
clean_fk['Price'] = clean_fk['Price'].apply(lambda x: str(x))

cvv = TfidfVectorizer(max_features=5000, stop_words="english")

ps = PorterStemmer()
def stem(text):
  y = []

  for i in text.split():
    y.append(ps.stem(i))

  return " ".join(y)

# OPENAI vars
api_key = ''
# api_key = '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'

model        = OpenAI(api_key)
prompter     = Prompter('ner.jinja')
pipe         = Pipeline(prompter , model)

def recommend(data):
    item_name = ' '.join(list(map(stem, data.get('Labels', '').split(" "))))
    item_brand = data.get('Brand', '').replace(" ", "").lower()
    item_price = str(data.get('Price', '')).replace(",", "")
    item_rating = str(data.get('Rating', ''))

    if not (item_brand or item_name or item_price or item_rating ):
       return [random.randint(1,len(clean_fk) - 1) for x in range(5)]

    clean_fk['Name'] = clean_fk['Name'] + clean_fk['BreadCrumbs']
    new_fk = pd.concat([clean_fk, pd.DataFrame([{'Brand': item_brand, 'Name': item_name, 'Price': item_price, 'Rating': item_rating}])], ignore_index=True)

    name_vec = cvv.fit_transform(new_fk['Name']).toarray()
    brand_vec = cvv.fit_transform(new_fk['Brand']).toarray()
    price_vec = cvv.fit_transform(new_fk['Price']).toarray()

    name_similarity = cosine_similarity(name_vec)
    brand_similarity = cosine_similarity(brand_vec)
    price_similarity = cosine_similarity(price_vec)

    overall_similarity = name_similarity[-1] + brand_similarity[-1] + list(map(lambda x: x * 0.65, price_similarity[-1]))

    recom_indices = sorted(range(len(overall_similarity)), key=lambda i: overall_similarity[i], reverse=True)[1:6]
    recommendations = [idx for idx in recom_indices]
    return recommendations

@app.route('/recommend', methods=['POST'])
def recommend_route(data=None):
    return jsonify(recommend(request.json))

@app.route('/chat', methods=['POST'])
def chat_route():
    item = request.json
    message = item.get('message', None)
    # return jsonify(recommend({"Brand": "nike"}))
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
           data[i['E']] = data.get(i['E'], '') + ' ' + str(i['W'])

        print(recommend(data))
        return jsonify(recommend(data))
    except Exception as e:
        print(e)
        return "Error", 400

if __name__ == '__main__':
    app.run(debug=True)
