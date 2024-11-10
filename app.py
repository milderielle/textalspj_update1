from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# โหลดข้อมูลและสร้างโมเดล
files = ['INFP_cleaned.csv', 'INFJ_cleaned.csv', 'INTJ_cleaned.csv', 'INTP_cleaned.csv']
dfs = [pd.read_csv(file) for file in files]
data = pd.concat(dfs, ignore_index=True)

mbti_to_animal = {'INTP': 'Owl', 'INTJ': 'Cat', 'INFP': 'Dolphin', 'INFJ': 'Wolf'}
data['animal'] = data['mbti_type'].map(mbti_to_animal)
data['comment'].fillna('', inplace=True)

# สร้าง pipeline สำหรับโมเดล
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['comment'], data['animal'], test_size=0.2, random_state=42)

# ฝึกโมเดล
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

pipeline.fit(X_train, y_train)

# ฟังก์ชันสำหรับทำนายประเภทสัตว์และคำนวณเปอร์เซ็นต์
def predict_animal_with_prob(comment):
    prob = pipeline.predict_proba([comment])[0]  # รับผลความน่าจะเป็นของแต่ละประเภท
    animals = ['Owl', 'Cat', 'Dolphin', 'Wolf']
    result = {animals[i]: prob[i] * 100 for i in range(len(animals))}  # แปลงเป็นเปอร์เซ็นต์
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_comment = request.form['comment']
        result = predict_animal_with_prob(user_comment)
        
        # คำนวณหาสัตว์ที่มีเปอร์เซ็นต์สูงสุด
        highest_animal = max(result, key=result.get)
        highest_percentage = result[highest_animal]
        
        return render_template('index.html', result=result, highest_animal=highest_animal, highest_percentage=round(highest_percentage, 2))

if __name__ == '__main__':
    app.run(debug=True)
