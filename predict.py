from datetime import datetime, timedelta

import joblib
import pandas as pd
import psycopg2


DB_CONFIG = {
    "dbname": "projekt_db",
    "user": "admin",
    "password": "admin",
    "host": "postgres",
    "port": "5432"
}

MODEL_PATH = "models/parking_occupancy_model.joblib"

PARKING_ID = "PARKING_A"
MAX_SPOTS = 100


def connect_to_database():
    return psycopg2.connect(**DB_CONFIG)


def prepare_prediction_table(conn):
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parking_predictions (
            id SERIAL PRIMARY KEY,
            parking_id VARCHAR(50) NOT NULL,
            predicted_for TIMESTAMP NOT NULL,
            predicted_occupied_spaces NUMERIC(10, 2) NOT NULL,
            predicted_free_spaces NUMERIC(10, 2) NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    cursor.close()


def load_model():
    return joblib.load(MODEL_PATH)


def load_current_status(conn):
    cursor = conn.cursor()

    cursor.execute("""
        SELECT occupied_spaces, free_spaces
        FROM parking_status
        WHERE parking_id = %s;
    """, (PARKING_ID,))

    row = cursor.fetchone()
    cursor.close()

    if row is None:
        raise ValueError("Brak rekordu parkingu w tabeli parking_status")

    return int(row[0]), int(row[1])


def load_last_hour_stats(conn):
    """
    Pobiera statystyki z ostatniej godziny dostępnej w danych historycznych.

    Nie używamy NOW() - INTERVAL '1 hour', bo producer generuje czas wirtualny,
    np. od 2023-01-01. Gdybyśmy użyli NOW(), zapytanie mogłoby zwrócić puste dane.
    """
    query = """
        WITH latest_hour AS (
            SELECT date_trunc('hour', MAX(event_time)) AS hour_window
            FROM parking_events_history
        )
        SELECT
            COUNT(*) FILTER (WHERE h.action = 'in') AS entries_count,
            COUNT(*) FILTER (WHERE h.action = 'out') AS exits_count,
            AVG(h.occupied_spaces) AS avg_occupied_spaces,
            MAX(h.occupied_spaces) AS max_occupied_spaces,
            MIN(h.free_spaces) AS min_free_spaces
        FROM parking_events_history h
        CROSS JOIN latest_hour lh
        WHERE date_trunc('hour', h.event_time) = lh.hour_window;
    """

    df = pd.read_sql_query(query, conn)

    if df.empty:
        return {
            "entries_count": 0,
            "exits_count": 0,
            "avg_occupied_spaces": 0,
            "max_occupied_spaces": 0,
            "min_free_spaces": MAX_SPOTS
        }

    row = df.iloc[0]

    return {
        "entries_count": int(row["entries_count"] or 0),
        "exits_count": int(row["exits_count"] or 0),
        "avg_occupied_spaces": float(row["avg_occupied_spaces"] or 0),
        "max_occupied_spaces": float(row["max_occupied_spaces"] or 0),
        "min_free_spaces": float(row["min_free_spaces"] or MAX_SPOTS)
    }


def load_latest_event_time(conn):
    cursor = conn.cursor()

    cursor.execute("""
        SELECT MAX(event_time)
        FROM parking_events_history;
    """)

    row = cursor.fetchone()
    cursor.close()

    if row is None or row[0] is None:
        return datetime.now().replace(minute=0, second=0, microsecond=0)

    return row[0].replace(minute=0, second=0, microsecond=0)


def load_stats_for_hour(conn, target_hour):
    """
    Pobiera historyczne statystyki dla konkretnej godziny dnia.

    Przykład:
    jeśli target_hour = 14, to bierzemy wszystkie historyczne rekordy
    z godziny 14:00 i liczymy ich średnie statystyki.

    Dzięki temu predykcje dla różnych godzin mogą dostać różne dane wejściowe.
    """
    query = """
        SELECT
            AVG(entries_count) AS entries_count,
            AVG(exits_count) AS exits_count,
            AVG(avg_occupied_spaces) AS avg_occupied_spaces,
            AVG(max_occupied_spaces) AS max_occupied_spaces,
            AVG(min_free_spaces) AS min_free_spaces
        FROM (
            SELECT
                date_trunc('hour', event_time) AS hour_window,
                COUNT(*) FILTER (WHERE action = 'in') AS entries_count,
                COUNT(*) FILTER (WHERE action = 'out') AS exits_count,
                AVG(occupied_spaces) AS avg_occupied_spaces,
                MAX(occupied_spaces) AS max_occupied_spaces,
                MIN(free_spaces) AS min_free_spaces
            FROM parking_events_history
            WHERE EXTRACT(HOUR FROM event_time) = %s
            GROUP BY date_trunc('hour', event_time)
        ) hourly_stats;
    """

    df = pd.read_sql_query(query, conn, params=(target_hour,))

    if df.empty:
        return None

    row = df.iloc[0]

    if pd.isna(row["avg_occupied_spaces"]):
        return None

    return {
        "entries_count": float(row["entries_count"] or 0),
        "exits_count": float(row["exits_count"] or 0),
        "avg_occupied_spaces": float(row["avg_occupied_spaces"] or 0),
        "max_occupied_spaces": float(row["max_occupied_spaces"] or 0),
        "min_free_spaces": float(row["min_free_spaces"] or MAX_SPOTS)
    }


def build_features(conn, feature_columns, forecast_hours):
    """
    Budujemy cechy dla kolejnych godzin względem ostatniego czasu z historii.

    Dla każdej prognozowanej godziny próbujemy pobrać historyczne statystyki
    dla tej konkretnej godziny dnia.

    Przykład:
    - predykcja na 14:00 bierze historyczne statystyki z godziny 14:00,
    - predykcja na 15:00 bierze historyczne statystyki z godziny 15:00.

    Jeśli dla danej godziny nie ma historii, używamy statystyk z ostatniej godziny.
    """
    base_time = load_latest_event_time(conn)

    current_occupied, current_free = load_current_status(conn)
    last_hour_stats = load_last_hour_stats(conn)

    rows = []

    for i in range(1, forecast_hours + 1):
        predicted_for = base_time + timedelta(hours=i)
        target_hour = predicted_for.hour

        hour_stats = load_stats_for_hour(conn, target_hour)

        if hour_stats is None:
            hour_stats = last_hour_stats

        row = {
            "hour": predicted_for.hour,
            "day_of_week": predicted_for.weekday(),
            "month": predicted_for.month,
            "is_weekend": 1 if predicted_for.weekday() in [5, 6] else 0,

            "entries_count": hour_stats["entries_count"],
            "exits_count": hour_stats["exits_count"],
            "avg_occupied_spaces": hour_stats["avg_occupied_spaces"] or current_occupied,
            "max_occupied_spaces": hour_stats["max_occupied_spaces"] or current_occupied,
            "min_free_spaces": hour_stats["min_free_spaces"] or current_free,

            # Demo pogody.
            # W wersji produkcyjnej można tu podpiąć realne API pogodowe.
            "temperature": 20.0,
            "precipitation": 0.0,
            "cloud_cover": 30.0,

            "predicted_for": predicted_for
        }

        rows.append(row)

    features_df = pd.DataFrame(rows)
    X = features_df[feature_columns]

    return features_df, X


def save_predictions(conn, predictions_df):
    cursor = conn.cursor()

    for _, row in predictions_df.iterrows():
        cursor.execute("""
            INSERT INTO parking_predictions (
                parking_id,
                predicted_for,
                predicted_occupied_spaces,
                predicted_free_spaces
            )
            VALUES (%s, %s, %s, %s);
        """, (
            PARKING_ID,
            row["predicted_for"],
            float(row["predicted_occupied_spaces"]),
            float(row["predicted_free_spaces"])
        ))

    conn.commit()
    cursor.close()


def main():
    forecast_hours = 6

    print("[+] Start predykcji obłożenia parkingu")

    conn = connect_to_database()

    try:
        prepare_prediction_table(conn)

        model_package = load_model()
        model = model_package["model"]
        feature_columns = model_package["feature_columns"]

        features_df, X = build_features(
            conn=conn,
            feature_columns=feature_columns,
            forecast_hours=forecast_hours
        )

        predicted_occupied = model.predict(X)

        features_df["predicted_occupied_spaces"] = predicted_occupied
        features_df["predicted_occupied_spaces"] = features_df["predicted_occupied_spaces"].clip(
            lower=0,
            upper=MAX_SPOTS
        )

        features_df["predicted_free_spaces"] = MAX_SPOTS - features_df["predicted_occupied_spaces"]

        save_predictions(conn, features_df)

        print("[+] Predykcje zapisane do tabeli parking_predictions")
        print(features_df[[
            "predicted_for",
            "predicted_occupied_spaces",
            "predicted_free_spaces"
        ]])

    finally:
        conn.close()


if __name__ == "__main__":
    main()