import os
from datetime import datetime

import joblib
import pandas as pd
import psycopg2
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


DB_CONFIG = {
    "dbname": "projekt_db",
    "user": "admin",
    "password": "admin",
    "host": "postgres",
    "port": "5432"
}

MODEL_PATH = "models/parking_occupancy_model.joblib"


def connect_to_database():
    return psycopg2.connect(**DB_CONFIG)


def prepare_training_table(conn):
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_training_runs (
            id SERIAL PRIMARY KEY,
            trained_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            rows_count INT NOT NULL,
            mae NUMERIC(10, 4) NOT NULL,
            rmse NUMERIC(10, 4) NOT NULL,
            model_path VARCHAR(255) NOT NULL
        );
    """)

    conn.commit()
    cursor.close()


def load_hourly_data(conn):
    query = """
        SELECT
            date_trunc('hour', event_time) AS hour_window,
            COUNT(*) FILTER (WHERE action = 'in') AS entries_count,
            COUNT(*) FILTER (WHERE action = 'out') AS exits_count,
            AVG(occupied_spaces) AS avg_occupied_spaces,
            MAX(occupied_spaces) AS max_occupied_spaces,
            MIN(free_spaces) AS min_free_spaces
        FROM parking_events_history
        GROUP BY date_trunc('hour', event_time)
        ORDER BY hour_window;
    """

    df = pd.read_sql_query(query, conn)

    if df.empty:
        return df

    df["hour_window"] = pd.to_datetime(df["hour_window"])

    return df


def add_weather_demo_data(df):
    """
    Wersja demonstracyjna danych pogodowych.

    W docelowej wersji tutaj można byłoby pobrać pogodę z zewnętrznego API.
    Na potrzeby projektu zostawiamy stałe wartości, żeby pipeline ML był kompletny:
    historia parkingu -> cechy pogodowe -> model -> predykcja.
    """
    df = df.copy()

    df["temperature"] = 20.0
    df["precipitation"] = 0.0
    df["cloud_cover"] = 30.0

    return df


def prepare_features(df):
    df = df.copy()

    df["hour"] = df["hour_window"].dt.hour
    df["day_of_week"] = df["hour_window"].dt.dayofweek
    df["month"] = df["hour_window"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Target: przewidujemy średnie obłożenie w kolejnej godzinie.
    df["target_next_hour_occupied"] = df["avg_occupied_spaces"].shift(-1)

    # Ostatni rekord nie ma targetu, bo nie znamy "następnej godziny".
    df = df.dropna()

    feature_columns = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "entries_count",
        "exits_count",
        "avg_occupied_spaces",
        "max_occupied_spaces",
        "min_free_spaces",
        "temperature",
        "precipitation",
        "cloud_cover"
    ]

    X = df[feature_columns]
    y = df["target_next_hour_occupied"]

    return X, y, feature_columns


def train_model(X, y):
    if len(X) < 10:
        raise ValueError(
            "Za mało danych do trenowania modelu. "
            "Uruchom producer i consumer dłużej, żeby zebrać minimum 10 godzin historii."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    return model, mae, rmse


def save_model(model, feature_columns):
    os.makedirs("models", exist_ok=True)

    model_package = {
        "model": model,
        "feature_columns": feature_columns,
        "trained_at": datetime.now().isoformat()
    }

    joblib.dump(model_package, MODEL_PATH)


def save_training_run(conn, rows_count, mae, rmse):
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO model_training_runs (
            rows_count,
            mae,
            rmse,
            model_path
        )
        VALUES (%s, %s, %s, %s);
    """, (
        rows_count,
        float(mae),
        float(rmse),
        MODEL_PATH
    ))

    conn.commit()
    cursor.close()


def main():
    print("[+] Start batchowego treningu modelu ML")

    conn = connect_to_database()

    try:
        prepare_training_table(conn)

        df = load_hourly_data(conn)

        if df.empty:
            print("[!] Brak danych w parking_events_history")
            return

        print(f"[+] Liczba rekordów godzinowych: {len(df)}")

        df = add_weather_demo_data(df)

        X, y, feature_columns = prepare_features(df)

        if X.empty:
            print("[!] Za mało danych po przygotowaniu cech")
            return

        model, mae, rmse = train_model(X, y)

        save_model(model, feature_columns)
        save_training_run(conn, len(X), mae, rmse)

        print("[+] Model zapisany")
        print(f"[+] Ścieżka modelu: {MODEL_PATH}")
        print(f"[+] MAE: {mae:.2f}")
        print(f"[+] RMSE: {rmse:.2f}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()