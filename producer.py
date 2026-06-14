import time
import json
import random
from confluent_kafka import Producer
from datetime import datetime, timedelta

config = {'bootstrap.servers': 'kafka:29092'}
producer = Producer(config)
topic_name = 'parking-events'

MAX_SPOTS = 100
current_cars = 0

virtual_time = datetime(2023, 1, 1, 0, 0, 0)

print(f"Rozpoczęto generowanie mądrych danych IoT. Pojemność parkingu: {MAX_SPOTS} miejsc.")

try:
    while True:
        
        virtual_time += timedelta(minutes=1)
        hour = virtual_time.hour
        
      
        if 7 <= hour < 10:
            prob_in = 0.8  
        elif 10 <= hour < 15:
            prob_in = 0.5  
        elif 15 <= hour < 19:
            prob_in = 0.2  
        else:
            prob_in = 0.5  
            
        
        if (hour >= 22 or hour < 6) and random.random() > 0.15:
            time.sleep(0.5) 
            continue

        if current_cars >= MAX_SPOTS:
            action = "out"
        elif current_cars <= 0:
            action = "in"
        else:
            action = "in" if random.random() < prob_in else "out"

        if action == "in":
            current_cars += 1
        else:
            current_cars -= 1
            
        event = {
            "spot_id": random.randint(1, MAX_SPOTS),
            "action": action,
            "timestamp": int(virtual_time.timestamp()),
            "virtual_time_str": virtual_time.strftime("%H:%M") 
        }
        
        payload = json.dumps(event)
        producer.produce(topic=topic_name, value=payload)
        producer.flush()
        
        print(f"[{event['virtual_time_str']}] Wysłano: {payload} | Aktualne obłożenie: {current_cars}/{MAX_SPOTS}")
        
        time.sleep(random.uniform(0.5, 1.5))

except KeyboardInterrupt:
    print("\nZatrzymano generator.")
