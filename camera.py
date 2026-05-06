"""
Gestor de camaras para captura de fotos del dataset.
Soporta camaras USB (via OpenCV) y camaras IP (via RTSP/MJPEG).
"""

import cv2
import uuid
import threading
import time
from pathlib import Path
from datetime import datetime
from config import DATASET_DIR


class CameraManager:
    """Gestor de camaras USB e IP."""

    def __init__(self):
        self.cameras = {}  # id -> {"type": "usb"|"ip", "source": ..., "name": ...}
        self._active_captures = {}  # camera_id -> VideoCapture
        self._lock = threading.Lock()
        self._next_id = 1

    def scan_usb_cameras(self) -> list[dict]:
        """Escanea camaras USB conectadas al servidor."""
        # Liberar capturas activas antes de escanear
        self.release_all()
        found = []
        for index in range(5):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                # Forzar formato MJPEG para obtener color
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                time.sleep(0.2)
                ret, frame = cap.read()
                if ret and frame is not None:
                    cam_id = f"usb_{index}"
                    cam_info = {
                        "id": cam_id,
                        "type": "usb",
                        "source": index,
                        "name": f"Camara USB {index}",
                        "resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
                    }
                    self.cameras[cam_id] = cam_info
                    found.append(cam_info)
                cap.release()
        return found

    def add_ip_camera(self, name: str, url: str) -> dict:
        """Anade una camara IP por URL RTSP o MJPEG."""
        cam_id = f"ip_{self._next_id}"
        self._next_id += 1

        # Probar conexion
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            cap.release()
            return {"error": f"No se puede conectar a: {url}"}

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return {"error": "Conectado pero sin imagen. Verifica la URL."}

        cam_info = {
            "id": cam_id,
            "type": "ip",
            "source": url,
            "name": name,
            "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
        }
        self.cameras[cam_id] = cam_info
        return cam_info

    def remove_camera(self, cam_id: str):
        """Elimina una camara del gestor."""
        self.release_capture(cam_id)
        self.cameras.pop(cam_id, None)

    def get_capture(self, cam_id: str):
        """Obtiene o crea una captura de video para la camara."""
        with self._lock:
            if cam_id in self._active_captures:
                return self._active_captures[cam_id]

            cam = self.cameras.get(cam_id)
            if not cam:
                return None

            cap = cv2.VideoCapture(cam["source"])
            if cap.isOpened():
                # Forzar formato MJPEG (color) en lugar de YUYV/YUV (gris)
                if cam["type"] == "usb":
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    # Pequeña pausa para que el driver aplique el formato
                    time.sleep(0.2)
                self._active_captures[cam_id] = cap
                return cap
            return None

    def release_capture(self, cam_id: str):
        """Libera la captura de una camara."""
        with self._lock:
            cap = self._active_captures.pop(cam_id, None)
            if cap:
                cap.release()

    def release_all(self):
        """Libera todas las capturas."""
        with self._lock:
            for cap in self._active_captures.values():
                cap.release()
            self._active_captures.clear()

    def get_frame(self, cam_id: str):
        """Captura un frame de la camara. Retorna (frame, error)."""
        cap = self.get_capture(cam_id)
        if not cap:
            return None, "Camara no disponible"

        ret, frame = cap.read()
        if not ret or frame is None:
            self.release_capture(cam_id)
            return None, "Error leyendo frame"

        # Corregir frames en escala de grises (2 canales o 3 canales iguales)
        if len(frame.shape) == 2:
            # Frame monocanal -> convertir a BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            # Frame con 1 canal expandido
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            # Frame BGRA -> BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        return frame, None

    def capture_and_save(self, cam_id: str, class_name: str) -> dict:
        """Captura un frame y lo guarda en la carpeta de la clase del dataset."""
        frame, error = self.get_frame(cam_id)
        if error:
            return {"error": error}

        class_dir = DATASET_DIR / class_name
        class_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cam_{timestamp}_{uuid.uuid4().hex[:6]}.jpg"
        filepath = class_dir / filename

        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return {
            "success": True,
            "filename": filename,
            "class": class_name,
            "path": str(filepath),
            "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
        }

    def scan_qr(self, cam_id: str) -> list[dict]:
        """Escanea el frame actual en busca de codigos QR y barras."""
        frame, error = self.get_frame(cam_id)
        if error:
            return []

        results = []

        # QR codes
        try:
            qr = cv2.QRCodeDetector()
            retval, decoded_info, points, _ = qr.detectAndDecodeMulti(frame)
            if retval and decoded_info:
                for i, text in enumerate(decoded_info):
                    if text:
                        pts = points[i].tolist() if points is not None and i < len(points) else []
                        results.append({"type": "QR", "data": text})
        except Exception:
            # Fallback: single QR detection
            try:
                qr = cv2.QRCodeDetector()
                decoded, points, _ = qr.detectAndDecode(frame)
                if decoded:
                    results.append({"type": "QR", "data": decoded})
            except Exception:
                pass

        # Barcodes (1D) - EAN, UPC, Code128, etc.
        try:
            barcode = cv2.barcode_BarcodeDetector()
            decoded, points, _ = barcode.detectAndDecode(frame)
            if decoded:
                results.append({"type": "BARCODE", "data": decoded})
        except Exception:
            pass

        return results

    def get_frame_jpeg(self, cam_id: str, quality: int = 70) -> bytes | None:
        """Captura un frame y lo retorna como bytes JPEG (para streaming)."""
        frame, error = self.get_frame(cam_id)
        if error:
            return None
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buffer.tobytes()

    def list_cameras(self) -> list[dict]:
        """Lista todas las camaras registradas."""
        return list(self.cameras.values())


# Instancia global
camera_manager = CameraManager()
