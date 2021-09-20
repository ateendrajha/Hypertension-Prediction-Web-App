import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from flask import Flask, render_template, request


app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template("health_index.html")

@app.route("/result", methods= ["POST", 'GET'])
def result():
    Rawdf = pd.read_csv("C:\\Users\\ateen\\Google Drive\\Python\\BP\\cardio.csv", sep=";")
    Rawdf['AgeinYr'] = (Rawdf.age)//365
    Rawdf['BP_Status'] = [1 if i > 140 or j > 90 else 0 for i,j in zip(Rawdf.ap_hi, Rawdf.ap_lo)  ]
    Finaldf = Rawdf[['gender','height','weight','cholesterol','alco','active','cardio','AgeinYr', 'BP_Status']]
    Finaldf['gender'] = Finaldf.gender.map({1:0,2:1})  # 1 - Male 0 - Female
    X_train = Finaldf.drop('BP_Status', axis=1)
    y_train = Finaldf[['BP_Status']]
    lr1 = sm.GLM(y_train, X_train, family= sm.families.Binomial())
    lm1 = lr1.fit()
    output = request.form.to_dict()
    X_test = pd.DataFrame({'gender':int(output['gender']), 'height':int(output['height']),	'weight':int(output['weight']),	
                            'cholesterol':int(output['chol']),	'alco':int(output['alcohol']),	'active':int(output['active']),	
                            'cardio':int(output['cardio']),	'AgeinYr':int(output['age'])}, index=[0])
    y_test_pred = lm1.predict(X_test)
    if y_test_pred.values > 0.218:
        prediction = 'Chances of Hypertension'
    else:
        prediction = 'Low Chance of Hypertension'
    return render_template("health_index.html", prediction = prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5001)



