"""
Script de prueba para el sistema MQTT.
Publica un mensaje de sensor y escucha el resultado.

Uso:
    python test_mqtt.py
"""

import json
import time
import threading
from datetime import datetime

import paho.mqtt.client as mqtt

BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC_SENSOR = "sensores/confeccion"
TOPIC_RESULTADO = "resultados/confeccion"

resultado_recibido = None


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"[+] Conectado a {BROKER}:{PORT}")
        client.subscribe(TOPIC_RESULTADO, qos=1)
        print(f"[+] Suscrito a: {TOPIC_RESULTADO}")
    else:
        print(f"[-] Error de conexion: {reason_code}")


def on_message(client, userdata, msg):
    global resultado_recibido
    topic = msg.topic
    payload = msg.payload.decode("utf-8")
    print(f"\n[!] RESULTADO RECIBIDO en {topic}:")
    try:
        data = json.loads(payload)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except:
        print(payload)
    resultado_recibido = payload


def main():
    global resultado_recibido

    print("=" * 50)
    print("  Prueba del Sistema MQTT - Confecciones")
    print("=" * 50)
    print(f"\nBroker: {BROKER}:{PORT}")
    print(f"Topico sensor: {TOPIC_SENSOR}")
    print(f"Topico resultado: {TOPIC_RESULTADO}")

    # Crear cliente
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"test-confecciones-{int(time.time())}",
    )

    client.on_connect = on_connect
    client.on_message = on_message

    # Conectar
    print(f"\n[*] Conectando a {BROKER}...")
    client.connect(BROKER, PORT, keepalive=60)
    client.loop_start()

    # Esperar conexion
    time.sleep(3)

    # Mensaje de prueba del sensor
    sensor_data = {
        "barcode": "7501234567890",
        "sensor_id": "sensor-prueba-001",
        "producto": "Confeccion de prueba",
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n[*] Enviando mensaje de sensor...")
    print(f"    Topico: {TOPIC_SENSOR}")
    print(f"    Payload: {json.dumps(sensor_data, indent=2)}")

    client.publish(TOPIC_SENSOR, json.dumps(sensor_data), qos=1)

    print("\n[*] Esperando resultado (max 30 segundos)...")
    print("    (El sistema debe: capturar foto -> ML -> publicar resultado)")

    # Esperar resultado
    timeout = 30
    start = time.time()
    while time.time() - start < timeout:
        if resultado_recibido:
            print("\n[+] Prueba completada exitosamente!")
            break
        time.sleep(1)
        elapsed = int(time.time() - start)
        if elapsed % 5 == 0 and elapsed > 0:
            print(f"    ... esperando ({elapsed}s)")
    else:
        print("\n[-] Timeout - No se recibio resultado")
        print("    Verifica que:")
        print("    1. El servidor esta corriendo")
        print("    2. MQTT esta conectado (ver /mqtt)")
        print("    3. La camara ID 1 esta disponible")

    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()
