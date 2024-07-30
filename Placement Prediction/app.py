from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load and prepare the dataset
da = pd.read_csv("C:\\Users\\lenovo\\OneDrive\\Documents\\Desktop\\collegePlace.csv")
df = pd.DataFrame(da)

columns_to_keep = ['Age', 'CGPA', 'Internships', 'PlacedOrNot', 'HistoryOfBacklogs']
df = df[columns_to_keep]

X = df.drop(columns=['PlacedOrNot'])
y = df['PlacedOrNot']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    cgpa = float(request.form['cgpa'])
    internships = int(request.form['internships'])
    backlogs = int(request.form['backlogs'])

    new_data = {
        'Age': [age],
        'CGPA': [cgpa],
        'Internships': [internships],
        'HistoryOfBacklogs': [backlogs]
    }

    new_df = pd.DataFrame(new_data)
    placement_prediction = model.predict(new_df)
    placement_prediction_label = 'Yes' if placement_prediction[0] == 1 else 'No'
    
    return render_template('result.html', prediction=placement_prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
