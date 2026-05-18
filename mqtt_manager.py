"""
Gestor de MQTT para triggers automaticos.
Conecta a un broker MQTT, se suscribe a topicos configurados,
y activa camaras cuando recibe mensajes.

Uso:
    mqtt_manager.start()    # conectar y suscribirse
    mqtt_manager.stop()     # desconectar
    mqtt_manager.reload()   # recargar configuracion
"""

import json
import threading
import logging
from datetime import datetime

import paho.mqtt.client as mqtt

from database import get_setting_sync, get_all_mqtt_triggers

logger = logging.getLogger("mqtt")


class MQTTManager:
    """Gestor singleton del cliente MQTT."""

    def __init__(self):
        self.client = None
        self._thread = None
        self._connected = False
        self._trigger_callback = None  # se asigna desde app.py
        self._last_error = None

    def set_trigger_callback(self, callback):
        """Asigna la funcion que ejecuta el trigger (viene de app.py)."""
        self._trigger_callback = callback

    @property
    def is_connected(self):
        return self._connected

    @property
    def last_error(self):
        return self._last_error

    def start(self):
        """Inicia la conexion MQTT en un hilo daemon."""
        if self._connected:
            return

        host = get_setting_sync("mqtt_host") or ""
        port = int(get_setting_sync("mqtt_port") or 1883)
        user = get_setting_sync("mqtt_user") or ""
        password = get_setting_sync("mqtt_pass") or ""
        enabled = get_setting_sync("mqtt_enabled") or "false"

        if enabled != "true" or not host:
            logger.info("MQTT deshabilitado o sin host configurado")
            return

        try:
            self.client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"confecciones-{datetime.now().strftime('%H%M%S')}",
            )

            if user:
                self.client.username_pw_set(user, password)

            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message

            logger.info(f"Conectando a MQTT {host}:{port}")
            self.client.connect_async(host, port, keepalive=60)

            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Error iniciando MQTT: {e}")

    def _loop(self):
        """Bucle principal del cliente MQTT (en hilo daemon)."""
        try:
            self.client.loop_forever()
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Error en bucle MQTT: {e}")
            self._connected = False

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        """Callback cuando se conecta al broker."""
        if reason_code == 0:
            self._connected = True
            self._last_error = None
            logger.info("MQTT conectado")
            self._subscribe_triggers()
        else:
            self._connected = False
            self._last_error = f"Conexion rechazada: {reason_code}"
            logger.error(f"MQTT conexion rechazada: {reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        """Callback cuando se desconecta."""
        self._connected = False
        logger.warning(f"MQTT desconectado: {reason_code}")

    def _on_message(self, client, userdata, msg):
        """Callback cuando llega un mensaje en un topico suscrito."""
        topic = msg.topic
        payload = msg.payload.decode("utf-8", errors="replace")
        logger.info(f"MQTT recibido: {topic} -> {payload}")

        triggers = get_all_mqtt_triggers()
        for trigger in triggers:
            if trigger["topic"] == topic and trigger["active"] == "true":
                camera_id = trigger["camera_id"]
                result_topic = trigger["result_topic"]
                payload_vars = trigger.get("payload_vars", "")
                logger.info(f"Trigger activado: topico={topic}, camara={camera_id}")
                self._process_trigger(camera_id, topic, payload, result_topic, payload_vars)
                break

    def _extract_vars(self, payload_str, payload_vars_str):
        """Extrae variables del payload segun la configuracion.

        payload_vars_str: nombres separados por coma, ej: "barcode,sensor_id,linea"
        Retorna dict con las variables extraidas.
        """
        extracted = {}
        var_names = [v.strip() for v in payload_vars_str.split(",") if v.strip()]

        if not var_names:
            return extracted

        # Intentar parsear como JSON
        try:
            data = json.loads(payload_str)
            if isinstance(data, dict):
                for var in var_names:
                    if var in data:
                        extracted[var] = data[var]
            else:
                # JSON pero no es dict (ej: array, string)
                extracted["payload"] = data
        except (json.JSONDecodeError, TypeError):
            # No es JSON - payload plano
            if len(var_names) == 1:
                extracted[var_names[0]] = payload_str
            else:
                extracted["raw"] = payload_str

        return extracted

    def _process_trigger(self, camera_id, topic, payload, result_topic, payload_vars=""):
        """Ejecuta el trigger de camara y publica resultado enriquecido."""
        if not self._trigger_callback:
            logger.warning("No hay callback de trigger asignado")
            return

        try:
            # Extraer variables del payload entrante
            extracted = self._extract_vars(payload, payload_vars)

            # Ejecutar trigger (foto + ML)
            ml_result = self._trigger_callback(camera_id)

            # Construir resultado combinado
            combined = {
                "timestamp": datetime.now().isoformat(),
                "source": {
                    "topic": topic,
                    "payload_raw": payload,
                    "variables": extracted,
                },
                "ml": ml_result,
            }

            # Resolver topico resultado (soporta {variable} templates)
            resolved_topic = result_topic
            if result_topic and extracted:
                try:
                    resolved_topic = result_topic.format(**extracted)
                except (KeyError, ValueError):
                    pass  # template no coincide, usar topic tal cual

            if resolved_topic:
                self.publish(resolved_topic, json.dumps(combined, ensure_ascii=False))
                logger.info(f"Resultado publicado en: {resolved_topic}")

        except Exception as e:
            logger.error(f"Error procesando trigger MQTT: {e}")
            if result_topic:
                error_msg = {"error": str(e), "topic": topic, "payload": payload}
                self.publish(result_topic, json.dumps(error_msg))

    def _subscribe_triggers(self):
        """Se suscribe a todos los topicos de triggers activos."""
        triggers = get_all_mqtt_triggers()
        for trigger in triggers:
            if trigger["active"] == "true" and trigger["topic"]:
                self.client.subscribe(trigger["topic"], qos=1)
                logger.info(f"Suscrito a: {trigger['topic']}")

    def publish(self, topic, payload):
        """Publica un mensaje en un topico."""
        if self.client and self._connected:
            result = self.client.publish(topic, payload, qos=1)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        return False

    def stop(self):
        """Detiene el cliente MQTT."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self._connected = False
            logger.info("MQTT detenido")

    def reload(self):
        """Recarga la configuracion y reconecta."""
        self.stop()
        self.start()


# Instancia global
mqtt_manager = MQTTManager()
