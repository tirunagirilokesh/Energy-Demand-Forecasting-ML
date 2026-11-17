import os
import joblib
import pandas as pd
import numpy as np

# ---- IMPORTANT: non-GUI backend ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from django.shortcuts import render
from django.conf import settings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ================= BASIC PAGES =================
def index(request):
    return render(request, 'index.html')


def overview(request):
    return render(request, 'overview.html')


# ================= TRAIN MODELS =================
def train_models(request):

    df = pd.read_csv(r'media/energy_demand_dataset.csv')

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    df['Weekend'] = (df['DayOfWeek'] >= 5).astype(int)

    feature_cols = [
        'Temperature', 'Humidity', 'Wind_Speed',
        'Avg_Past_Consumption',
        'Lag_1', 'Lag_24',
        'Rolling_Mean_3', 'Rolling_Mean_24',
        'Hour', 'DayOfWeek', 'Month', 'Weekend'
    ]

    X = df[feature_cols]
    y = df['Electricity_Consumed']

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model_dir = settings.MEDIA_ROOT / 'models'
    graph_dir = settings.MEDIA_ROOT / 'graphs'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    # -------- RANDOM FOREST --------
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, model_dir / 'rf.pkl')

    # -------- GRADIENT BOOSTING --------
    gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    gb.fit(X_train, y_train)
    joblib.dump(gb, model_dir / 'gb.pkl')

    # -------- SVM --------
    svm = SVR()
    svm.fit(X_train_s, y_train)
    joblib.dump(svm, model_dir / 'svm.pkl')

    # -------- ANN --------
    ann = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_s.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    ann.compile(optimizer='adam', loss='mse')
    ann.fit(X_train_s, y_train, epochs=40, verbose=0)
    ann.save(model_dir / 'ann.h5')

    joblib.dump(scaler, model_dir / 'scaler.pkl')

    # -------- PREDICTIONS --------
    rf_pred = rf.predict(X_test)
    gb_pred = gb.predict(X_test)
    svm_pred = svm.predict(X_test_s)
    ann_pred = ann.predict(X_test_s).flatten()

    # -------- METRICS FUNCTION --------
    def metrics(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        return {
            'MAE': round(mean_absolute_error(y_true, y_pred), 4),
            'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
            'R2': round(r2, 4),
            'accuracy': round(r2 * 100, 2)
        }

    # -------- CALCULATE RESULTS --------
    results = {
        'rf': metrics(y_test, rf_pred),
        'gb': metrics(y_test, gb_pred),
        'svm': metrics(y_test, svm_pred),
        'ann': metrics(y_test, ann_pred),
    }

    # ================= GRAPHS =================
    plt.figure(figsize=(8,4))
    plt.plot(y_test.values[:200], label='Actual')
    plt.plot(gb_pred[:200], label='Predicted')
    plt.xlabel("Time Index")
    plt.ylabel("Electricity Consumption")
    plt.title("Actual vs Predicted ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_dir / 'actual_vs_pred.png')
    plt.close()

    residuals = y_test.values - gb_pred

    plt.figure(figsize=(8,4))
    plt.plot(residuals[:200])
    plt.axhline(0)
    plt.xlabel("Time Index")
    plt.ylabel("Residual Error")
    plt.title("Residual Errors")
    plt.tight_layout()
    plt.savefig(graph_dir / 'residuals.png')
    plt.close()

    plt.figure(figsize=(8,4))
    plt.hist(residuals, bins=40)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.tight_layout()
    plt.savefig(graph_dir / 'error_dist.png')
    plt.close()

    plt.figure(figsize=(8,6))
    plt.barh(feature_cols, gb.feature_importances_)
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(graph_dir / 'feature_importance.png')
    plt.close()

    return render(request, 'train.html', {
        'rf': results['rf'],
        'gb': results['gb'],
        'svm': results['svm'],
        'ann': results['ann'],
        'g1': settings.MEDIA_URL + 'graphs/actual_vs_pred.png',
        'g2': settings.MEDIA_URL + 'graphs/residuals.png',
        'g3': settings.MEDIA_URL + 'graphs/error_dist.png',
        'g4': settings.MEDIA_URL + 'graphs/feature_importance.png',
    })


# ================= PREDICTION =================
def predict_energy(request):

    if request.method == 'POST':

        ts = pd.to_datetime(request.POST['timestamp'])

        temp_real = float(request.POST['temp'])
        humidity_real = float(request.POST['humidity'])
        wind_real = float(request.POST['wind'])
        past_real = float(request.POST['past'])

        TEMP_MIN, TEMP_MAX = 10, 45
        HUM_MIN, HUM_MAX = 20, 100
        WIND_MIN, WIND_MAX = 0, 30
        PAST_MIN, PAST_MAX = 100, 500

        def scale(value, min_val, max_val):
            scaled = (value - min_val) / (max_val - min_val)
            return max(0, min(1, scaled))

        temp = scale(temp_real, TEMP_MIN, TEMP_MAX)
        humidity = scale(humidity_real, HUM_MIN, HUM_MAX)
        wind = scale(wind_real, WIND_MIN, WIND_MAX)
        past = scale(past_real, PAST_MIN, PAST_MAX)

        hour = ts.hour / 23
        day = ts.dayofweek / 6
        month = (ts.month - 1) / 11
        weekend = 1 if ts.dayofweek >= 5 else 0

        input_data = [[
            temp, humidity, wind,
            past,
            past, past,
            past, past,
            hour, day, month, weekend
        ]]

        model = joblib.load(settings.MEDIA_ROOT / 'models' / 'gb.pkl')
        prediction_scaled = model.predict(input_data)[0]

        prediction_kwh = prediction_scaled * PAST_MAX

        df = pd.read_csv(r'media/energy_demand_dataset.csv')
        low_threshold = df['Electricity_Consumed'].quantile(0.33)
        high_threshold = df['Electricity_Consumed'].quantile(0.66)

        if prediction_scaled < low_threshold:
            level = "Low Demand"
        elif prediction_scaled < high_threshold:
            level = "Medium Demand"
        else:
            level = "High / Peak Demand"

        if level == "Low Demand":
            recommendation = "Demand is low. Utilize surplus energy for storage or reduce generation."
        elif level == "Medium Demand":
            recommendation = "Demand is moderate. Maintain supply-demand balance and monitor grid."
        else:
            recommendation = "High demand detected. Activate peak load management and backup systems."

        return render(request, 'result.html', {
            'prediction': round(prediction_kwh, 2),
            'level': level,
            'timestamp': ts,
            'recommendation': recommendation,
        })

    return render(request, 'predict.html')

# import os
# import joblib
# import pandas as pd
# import numpy as np

# # ---- IMPORTANT: non-GUI backend ----
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# from django.shortcuts import render
# from django.conf import settings

# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense


# def index(request):
#     return render(request, 'index.html')


# def overview(request):
#     return render(request, 'overview.html')


# def train_models(request):
#     # -------- LOAD DATA --------
#     df = pd.read_csv(r'media/energy_demand_dataset.csv')

#     df['Timestamp'] = pd.to_datetime(df['Timestamp'])
#     df['Hour'] = df['Timestamp'].dt.hour
#     df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
#     df['Month'] = df['Timestamp'].dt.month
#     df['Weekend'] = (df['DayOfWeek'] >= 5).astype(int)

#     feature_cols = [
#         'Temperature', 'Humidity', 'Wind_Speed',
#         'Avg_Past_Consumption',
#         'Lag_1', 'Lag_24',
#         'Rolling_Mean_3', 'Rolling_Mean_24',
#         'Hour', 'DayOfWeek', 'Month', 'Weekend'
#     ]

#     X = df[feature_cols]
#     y = df['Electricity_Consumed']

#     split = int(len(df) * 0.8)
#     X_train, X_test = X[:split], X[split:]
#     y_train, y_test = y[:split], y[split:]

#     scaler = StandardScaler()
#     X_train_s = scaler.fit_transform(X_train)
#     X_test_s = scaler.transform(X_test)

#     model_dir = settings.MEDIA_ROOT / 'models'
#     graph_dir = settings.MEDIA_ROOT / 'graphs'
#     os.makedirs(model_dir, exist_ok=True)
#     os.makedirs(graph_dir, exist_ok=True)

#     # -------- RANDOM FOREST --------
#     rf = RandomForestRegressor(random_state=42)
#     rf.fit(X_train, y_train)
#     joblib.dump(rf, model_dir / 'rf.pkl')

#     # -------- GRADIENT BOOSTING  --------
#     gb = GradientBoostingRegressor(
#         n_estimators=300,
#         learning_rate=0.05,
#         max_depth=4,
#         random_state=42
#     )
#     gb.fit(X_train, y_train)
#     joblib.dump(gb, model_dir / 'gb.pkl')

#     # -------- SVM --------
#     svm = SVR()
#     svm.fit(X_train_s, y_train)
#     joblib.dump(svm, model_dir / 'svm.pkl')

#     # -------- ANN --------
#     ann = Sequential([
#         Dense(64, activation='relu', input_shape=(X_train_s.shape[1],)),
#         Dense(32, activation='relu'),
#         Dense(1)
#     ])
#     ann.compile(optimizer='adam', loss='mse')
#     ann.fit(X_train_s, y_train, epochs=40, verbose=0)
#     ann.save(model_dir / 'ann.h5')

#     joblib.dump(scaler, model_dir / 'scaler.pkl')

#     # -------- PREDICTIONS --------
#     rf_pred  = rf.predict(X_test)
#     gb_pred  = gb.predict(X_test)
#     svm_pred = svm.predict(X_test_s)
#     ann_pred = ann.predict(X_test_s).flatten()

#     def metrics(y_t, y_p):
#         return {
#             'MAE': round(mean_absolute_error(y_t, y_p), 4),
#             'RMSE': round(np.sqrt(mean_squared_error(y_t, y_p)), 4),
#             'R2': round(r2_score(y_t, y_p), 4)
#         }

#     results = {
#         'rf': metrics(y_test, rf_pred),
#         'gb': metrics(y_test, gb_pred),
#         'svm': metrics(y_test, svm_pred),
#         'ann': metrics(y_test, ann_pred),
#     }

#     # ================= GRAPHS  =================

#     # 1️⃣ Actual vs Predicted
#     plt.figure(figsize=(8,4))
#     plt.plot(y_test.values[:200], label='Actual Consumption')
#     plt.plot(gb_pred[:200], label='Predicted Consumption')
#     plt.xlabel("Time Index")
#     plt.ylabel("Electricity Consumption (kWh)")
#     plt.title("Actual vs Predicted Energy Consumption (Gradient Boosting)")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(graph_dir / 'actual_vs_pred.png')
#     plt.close()


#     # 2️⃣ Residuals Plot
#     residuals = y_test.values - gb_pred

#     plt.figure(figsize=(8,4))
#     plt.plot(residuals[:200])
#     plt.axhline(0)
#     plt.xlabel("Time Index")
#     plt.ylabel("Residual Error")
#     plt.title("Residual Errors Over Time")
#     plt.tight_layout()
#     plt.savefig(graph_dir / 'residuals.png')
#     plt.close()


#     # 3️⃣ Error Distribution
#     plt.figure(figsize=(8,4))
#     plt.hist(residuals, bins=40)
#     plt.xlabel("Prediction Error")
#     plt.ylabel("Frequency")
#     plt.title("Distribution of Prediction Errors")
#     plt.tight_layout()
#     plt.savefig(graph_dir / 'error_dist.png')
#     plt.close()


#     # 4️⃣ Feature Importance (Gradient Boosting)
#     plt.figure(figsize=(8,6))
#     plt.barh(feature_cols, gb.feature_importances_)
#     plt.xlabel("Importance Score")
#     plt.ylabel("Features")
#     plt.title("Feature Importance ")
#     plt.tight_layout()
#     plt.savefig(graph_dir / 'feature_importance.png')
#     plt.close()

#     return render(request, 'train.html', {
#         'rf': results['rf'],
#         'gb': results['gb'],
#         'svm': results['svm'],
#         'ann': results['ann'],

#         'g1': settings.MEDIA_URL + 'graphs/actual_vs_pred.png',
#         'g2': settings.MEDIA_URL + 'graphs/residuals.png',
#         'g3': settings.MEDIA_URL + 'graphs/error_dist.png',
#         'g4': settings.MEDIA_URL + 'graphs/feature_importance.png',
#     })
# def predict_energy(request):
#     if request.method == 'POST':

#         ts = pd.to_datetime(request.POST['timestamp'])


#         temp_real = float(request.POST['temp'])        # °C
#         humidity_real = float(request.POST['humidity'])  # %
#         wind_real = float(request.POST['wind'])        # km/h
#         past_real = float(request.POST['past'])        # kWh


#         TEMP_MIN, TEMP_MAX = 10, 45
#         HUM_MIN, HUM_MAX = 20, 100
#         WIND_MIN, WIND_MAX = 0, 30
#         PAST_MIN, PAST_MAX = 100, 500


#         def scale(value, min_val, max_val):
#             scaled = (value - min_val) / (max_val - min_val)
#             return max(0, min(1, scaled))  

#         temp = scale(temp_real, TEMP_MIN, TEMP_MAX)
#         humidity = scale(humidity_real, HUM_MIN, HUM_MAX)
#         wind = scale(wind_real, WIND_MIN, WIND_MAX)
#         past = scale(past_real, PAST_MIN, PAST_MAX)

#         hour = ts.hour / 23
#         day = ts.dayofweek / 6
#         month = (ts.month - 1) / 11
#         weekend = 1 if ts.dayofweek >= 5 else 0

#         lag_1 = past
#         lag_24 = past
#         rolling_3 = past
#         rolling_24 = past

#         input_data = [[
#             temp, humidity, wind,
#             past,
#             lag_1, lag_24,
#             rolling_3, rolling_24,
#             hour, day, month, weekend
#         ]]

#         # -------------------------------
#         # 4️⃣ LOAD MODEL
#         # -------------------------------
#         model = joblib.load(settings.MEDIA_ROOT / 'models' / 'gb.pkl')
#         prediction_scaled = model.predict(input_data)[0]


#         prediction_kwh = prediction_scaled * PAST_MAX

#         df = pd.read_csv(r'media/energy_demand_dataset.csv')
#         low_threshold = df['Electricity_Consumed'].quantile(0.33)
#         high_threshold = df['Electricity_Consumed'].quantile(0.66)

#         if prediction_scaled < low_threshold:
#             level = "Low Demand"
#         elif prediction_scaled < high_threshold:
#             level = "Medium Demand"
#         else:
#             level = "High / Peak Demand"


#         if level == "Low Demand":
#             recommendation = (
#                 "Demand is low. Consider utilizing surplus energy for battery "
#                 "storage, reducing generation load, or scheduling maintenance activities."
#             )

#         elif level == "Medium Demand":
#             recommendation = (
#                 "Demand is moderate. Maintain balanced supply, monitor grid "
#                 "stability, and prepare for possible peak fluctuations."
#             )

#         else:
#             recommendation = (
#                 "High demand detected. Activate peak load management strategies "
#                 "such as demand response programs, backup generators, or load shifting."
#             )

#         return render(request, 'result.html', {
#             'prediction': round(prediction_kwh, 2),
#             'level': level,
#             'timestamp': ts,
#             'recommendation': recommendation,
#         })

#     return render(request, 'predict.html')