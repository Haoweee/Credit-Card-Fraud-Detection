from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('../models/xgboost_model.pkl')

@app.route('/')
def home():
    return render_template('main.html')


@app.route('/transactions')
def transactions():
    return render_template('transactions.html')


@app.route('/analysis')
def analysis():
    return render_template('analysis.html')


@app.route('/predict', methods=['POST'])
def predict():
    print('inside predict')
    form_data = {
        'category': request.form['category'],
        'amt': float(request.form['amt']),
        'state': request.form['state'],
        'zip': int(request.form['zip']),
        'job_category': request.form['job_category'],
        'trans_day': int(request.form['trans_day']),
        'trans_month': int(request.form['trans_month']),
        'trans_hour': int(request.form['trans_hour']),
        'age_at_transaction': int(request.form['age_at_transaction']),
        'city_pop_group': request.form['city_pop_group'],
        'time_since_last_trans': float(request.form['time_since_last_trans']),
        'avg_transaction_amount': float(request.form['avg_transaction_amount']),
        'category_transaction_count': int(request.form['category_transaction_count']),
        'unique_transactions_day': int(request.form['unique_transactions_day']),
        'user_merchant_distance': float(request.form['user_merchant_distance'])
    }

    form_df = pd.DataFrame([form_data])
    prediction = model.predict(form_df)
    prediction_proba = model.predict_proba(form_df)

    confidence = prediction_proba[0][1] if prediction == 1 else prediction_proba[0][0]
    confidence_percentage = f"{confidence * 100:.2f}%"

    print('prediction', prediction, flush=True)
    print('prediction_proba', prediction_proba, flush=True)
    print('confidence', confidence, flush=True)
    print('confidence_percentage', confidence_percentage, flush=True)

    return render_template('result.html', prediction=prediction, confidence=confidence_percentage)


if __name__ == '__main__':
    app.run(debug=True)
