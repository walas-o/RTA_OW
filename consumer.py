import json
import time
from datetime import datetime

import psycopg2
from confluent_kafka import Consumer


MAX_SPOTS = 100
PARKING_ID = "PARKING_A"

DB_CONFIG = {
    "dbname": "projekt_db",
    "user": "admin",
    "password": "admin",
    "host": "postgres",
    "port": "5432"
}

KAFKA_CONFIG = {
    "bootstrap.servers": "kafka:29092",
    "group.id": "parking-consumers-advanced",
    "auto.offset.reset": "latest"
}


def connect_to_database():
    while True:
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            print("[+] Połączono z bazą danych PostgreSQL")
            return conn
        except Exception as error:
            print(f"[WARN] PostgreSQL jeszcze niegotowy: {error}")
            time.sleep(3)


def prepare_database(conn):
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parking_status (
            parking_id VARCHAR(50) PRIMARY KEY,
            occupied_spaces INT NOT NULL,
            free_spaces INT NOT NULL,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS spot_status (
            spot_id INT PRIMARY KEY,
            status VARCHAR(10) NOT NULL,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parking_events_history (
            id SERIAL PRIMARY KEY,
            parking_id VARCHAR(50) NOT NULL,
            spot_id INT NOT NULL,
            action VARCHAR(10) NOT NULL,
            spot_status VARCHAR(10) NOT NULL,
            occupied_spaces INT NOT NULL,
            free_spaces INT NOT NULL,
            event_timestamp BIGINT,
            virtual_time_str VARCHAR(10),
            event_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("""
        INSERT INTO parking_status (
            parking_id,
            occupied_spaces,
            free_spaces,
            updated_at
        )
        VALUES (%s, 0, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (parking_id)
        DO NOTHING;
    """, (PARKING_ID, MAX_SPOTS))

    conn.commit()
    cursor.close()

    print("[+] Tabele przygotowane w PostgreSQL")


def load_current_occupied(conn):
    cursor = conn.cursor()

    cursor.execute("""
        SELECT occupied_spaces
        FROM parking_status
        WHERE parking_id = %s;
    """, (PARKING_ID,))

    row = cursor.fetchone()
    cursor.close()

    if row is None:
        return 0

    return int(row[0])


def validate_event(data):
    if "action" not in data:
        raise ValueError("Brak pola action w evencie")

    if "spot_id" not in data:
        raise ValueError("Brak pola spot_id w evencie")

    action = data["action"]
    spot_id = int(data["spot_id"])

    if action not in ["in", "out"]:
        raise ValueError(f"Niepoprawna akcja: {action}")

    if spot_id <= 0 or spot_id > MAX_SPOTS:
        raise ValueError(f"spot_id poza zakresem: {spot_id}")

    return action, spot_id


def get_event_time(data):
    timestamp = data.get("timestamp")

    if timestamp is None:
        return datetime.now()

    return datetime.fromtimestamp(int(timestamp))


def save_event(
    conn,
    spot_id,
    action,
    spot_status,
    current_occupied,
    free_spaces,
    event_timestamp,
    virtual_time_str,
    event_time
):
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE parking_status
        SET
            occupied_spaces = %s,
            free_spaces = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE parking_id = %s;
    """, (current_occupied, free_spaces, PARKING_ID))

    cursor.execute("""
        INSERT INTO spot_status (
            spot_id,
            status,
            updated_at
        )
        VALUES (%s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (spot_id)
        DO UPDATE SET
            status = EXCLUDED.status,
            updated_at = CURRENT_TIMESTAMP;
    """, (spot_id, spot_status))

    cursor.execute("""
        INSERT INTO parking_events_history (
            parking_id,
            spot_id,
            action,
            spot_status,
            occupied_spaces,
            free_spaces,
            event_timestamp,
            virtual_time_str,
            event_time
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
    """, (
        PARKING_ID,
        spot_id,
        action,
        spot_status,
        current_occupied,
        free_spaces,
        event_timestamp,
        virtual_time_str,
        event_time
    ))

    conn.commit()
    cursor.close()


def main():
    conn = connect_to_database()
    prepare_database(conn)

    current_occupied = load_current_occupied(conn)

    consumer = Consumer(KAFKA_CONFIG)
    consumer.subscribe(["parking-events"])

    print("[+] Consumer uruchomiony")
    print(f"[+] Stan startowy: zajęte={current_occupied}, wolne={MAX_SPOTS - current_occupied}")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                print(f"[BŁĄD KAFKI]: {msg.error()}")
                continue

            try:
                data = json.loads(msg.value().decode("utf-8"))

                action, spot_id = validate_event(data)
                event_timestamp = data.get("timestamp")
                virtual_time_str = data.get("virtual_time_str")
                event_time = get_event_time(data)

                spot_status = "OCCUPIED" if action == "in" else "FREE"

                if action == "in":
                    if current_occupied >= MAX_SPOTS:
                        print("[WARN] Parking pełny. Pomijam wjazd.")
                        continue

                    current_occupied += 1

                elif action == "out":
                    if current_occupied <= 0:
                        print("[WARN] Parking pusty. Pomijam wyjazd.")
                        continue

                    current_occupied -= 1

                free_spaces = MAX_SPOTS - current_occupied

                save_event(
                    conn=conn,
                    spot_id=spot_id,
                    action=action,
                    spot_status=spot_status,
                    current_occupied=current_occupied,
                    free_spaces=free_spaces,
                    event_timestamp=event_timestamp,
                    virtual_time_str=virtual_time_str,
                    event_time=event_time
                )

                print(
                    f"[{virtual_time_str}] Miejsce {spot_id} -> {spot_status} | "
                    f"zajęte={current_occupied}, wolne={free_spaces}"
                )

            except Exception as error:
                conn.rollback()
                print(f"[BŁĄD PRZETWARZANIA]: {error}")

    except KeyboardInterrupt:
        print("[+] Zatrzymano consumer")

    finally:
        consumer.close()
        conn.close()
        print("[+] Połączenia zamknięte")


if __name__ == "__main__":
    main()