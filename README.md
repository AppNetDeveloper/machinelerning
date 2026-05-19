# Confecciones de Frutas - Sistema de Vision Industrial v3.1

Sistema de vision por computadora de bajo costo para clasificacion de cajas de frutas en entornos industriales. Pipeline IoT completo: sensor fisico → captura de camara → clasificacion ML + escaneo QR → publicacion MQTT / webhook callback.

---

## Tabla de Contenidos

1. [Caracteristicas](#características)
2. [Requisitos](#requisitos)
3. [Instalacion](#instalación)
4. [Primer Inicio](#primer-inicio)
5. [Configuracion](#configuración)
6. [Uso del Panel Web](#uso-del-panel-web)
7. [Camaras](#cámaras)
8. [Entrenamiento del Modelo](#entrenamiento-del-modelo)
9. [Predicciones](#predicciones)
10. [API REST](#api-rest)
11. [Integracion IoT / MQTT](#integración-iot--mqtt)
12. [Escaner QR / Codigos de Barras](#escáner-qr--códigos-de-barras)
13. [Seguridad](#seguridad)
14. [Solucion de Problemas](#solución-de-problemas)
15. [Licencia](#licencia)

---

## Caracteristicas

- **Clasificacion ML** con EfficientNet-B0 (transfer learning), EMA, MixUp, TTA determinista
- **Entrenamiento en 2 fases** con early stopping, warmup + cosine annealing, gradient clipping
- **Panel web** dark theme con login, dashboard, dataset, entrenamiento en vivo (SSE), predicciones
- **Captura de fotos** via camaras USB e IP (RTSP, HTTP, MJPEG)
- **Escaner QR/codigos de barras** integrado con deteccion en tiempo real
- **API REST** para integracion con sistemas externos
- **Triggers IoT** por camara con webhook callbacks y API key auth
- **MQTT** para integracion con brokers IoT (publicacion de resultados)
- **Historial completo** de entrenamientos y predicciones en SQLite
- **Seguridad** bcrypt, CSRF (HMAC tokens), sesiones firmadas, auto-migracion SHA-256→bcrypt

---

## Requisitos

### Software

| Componente | Version minima |
|------------|----------------|
| Python | 3.10+ |
| pip | 21.0+ |
| Sistema operativo | Linux (recomendado), Windows, macOS |

### Hardware Recomendado

| Componente | Minimo | Recomendado |
|------------|--------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| GPU | No requerido | NVIDIA con CUDA (entrenamiento mas rapido) |
| Almacenamiento | 2 GB libre | 10+ GB (para datasets grandes) |

### Camaras Compatibles

- **USB**: Cualquier camara UVC compatible con OpenCV (indices 0-4)
- **IP/Red**: Camaras con streaming RTSP, HTTP MJPEG, o HTTPS MJPEG
  - Ejemplo: Camaras IP Hikvision, Dahua, Reolink, etc.
  - URL tipica RTSP: `rtsp://usuario:password@192.168.1.100:554/stream1`

---

## Instalacion

### 1. Clonar el repositorio

```bash
git clone https://github.com/AppNetDeveloper/machinelerning.git
cd machinelerning
```

### 2. Crear entorno virtual

```bash
python3 -m venv venv
```

### 3. Activar entorno virtual

**Linux / macOS:**
```bash
source venv/bin/activate
```

**Windows (CMD):**
```cmd
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

Esto instalara:
- PyTorch + torchvision (modelos ML)
- FastAPI + uvicorn (servidor web)
- OpenCV (captura de camara)
- aiosqlite (base de datos async)
- bcrypt (hashing de passwords)
- paho-mqtt (cliente MQTT)
- Pillow, matplotlib (procesamiento de imagenes)

### 5. Verificar instalacion

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
```

---

## Primer Inicio

### Iniciar el servidor

```bash
# Asegurar que el entorno virtual esta activo
source venv/bin/activate

# Iniciar la aplicacion
python app.py
```

El servidor mostrara:

```
==================================================
  Panel de Confecciones de Frutas v3.1
  http://localhost:8000
  Acceso: admin / 123456789
==================================================
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Acceder al panel

1. Abre tu navegador en `http://localhost:8000`
2. Ingresa las credenciales:
   - **Usuario:** `admin`
   - **Password:** `123456789`
3. Haz clic en "Iniciar Sesion"

### Cambiar password por defecto

Despues del primer login, cambia el password desde **Configuracion → Gestion de Usuarios**.

---

## Configuracion

### Configuracion via Panel Web

La mayoria de ajustes se configuran desde el panel web en **Configuracion** (`/settings`):

| Ajuste | Descripcion |
|--------|-------------|
| Puerto del servidor | Puerto HTTP (default: 8000) |
| Host de escucha | Interfaz de red (default: 0.0.0.0) |
| API Key de triggers | Clave para proteger endpoints IoT |
| Gestion de usuarios | Crear, editar, eliminar usuarios |

Los cambios se guardan en `data/runtime_config.json`.

### Variables de Entorno

| Variable | Descripcion | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Clave para firmar sesiones (hex) | Generada aleatoriamente por arranque |

Para mantener sesiones entre reinicios:
```bash
export SECRET_KEY="tu-clave-secreta-larga-aqui"
python app.py
```

### Parametros ML (config.py)

Estos parametros se pueden ajustar editando `config.py`:

| Parametro | Valor Default | Descripcion |
|-----------|---------------|-------------|
| `IMG_SIZE` | 224 | Tamano de imagen de entrada (px) |
| `BATCH_SIZE` | 32 | Tamano de lote de entrenamiento |
| `EPOCHS_HEAD` | 10 | Epocas fase 1 (solo clasificador) |
| `EPOCHS_FINETUNE` | 50 | Epocas fase 2 (fine-tuning) |
| `LR_HEAD` | 0.001 | Learning rate fase 1 |
| `LR_FINETUNE` | 0.0002 | Learning rate fase 2 |
| `EMA_DECAY` | 0.999 | Decay del Exponential Moving Average |
| `MIXUP_ALPHA` | 0.2 | Intensidad de MixUp augmentation |
| `LABEL_SMOOTHING` | 0.1 | Suavizado de etiquetas |
| `EARLY_STOP_PATIENCE` | 10 | Epocas sin mejora para detener |
| `TTA_AUGMENTATIONS` | 7 | Numero de augmentaciones TTA |
| `GRAD_CLIP_MAX_NORM` | 1.0 | Maximo norma del gradiente |
| `WARMUP_EPOCHS` | 5 | Epocas de warmup del LR |

### Estructura de Directorios

```
machinelerning/
├── app.py                 # Punto de entrada, monta routers
├── config.py              # Configuracion centralizada
├── database.py            # Capa de acceso a SQLite
├── auth.py                # Autenticacion y CSRF
├── ml_model.py            # Gestor del modelo ML
├── train.py               # Pipeline de entrenamiento
├── camera.py              # Gestor de camaras
├── mqtt_manager.py        # Cliente MQTT
├── routers/               # Rutas FastAPI
│   ├── auth.py            # Login/logout
│   ├── pages.py           # Dashboard, historial
│   ├── dataset.py         # Gestion de dataset
│   ├── training.py        # Entrenamiento y progreso SSE
│   ├── predictions.py     # Predicciones
│   ├── settings.py        # Configuracion
│   ├── cameras.py         # Camaras y QR
│   ├── api.py             # API REST
│   └── mqtt.py            # Configuracion MQTT
├── templates/             # HTML Jinja2
├── static/                # CSS, JS
├── dataset/               # Imagenes por clase
│   ├── clase1/
│   ├── clase2/
│   └── ...
├── data/                  # Base de datos y config runtime
│   ├── app.db
│   └── runtime_config.json
├── modelo_confecciones.pth  # Modelo entrenado
└── clases.json            # Lista de clases
```

---

## Uso del Panel Web

### Dashboard (`/`)

Muestra resumen del sistema:
- Estado del modelo ML (cargado/no cargado)
- Numero de clases y imagenes en el dataset
- Ultimo entrenamiento y prediccion
- Camaras registradas

### Dataset (`/dataset`)

Gestiona las imagenes de entrenamiento:

1. **Crear nueva clase**: Escribe nombre y clic "Crear"
2. **Subir imagenes**: Selecciona clase → arrastra archivos o clic para subir
3. **Eliminar imagenes**: Clic en imagen → confirmar eliminacion
4. **Eliminar clase**: Clic en papelera junto al nombre de clase

**Organizacion del dataset:**
```
dataset/
├── manzana/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── naranja/
│   └── ...
└── platano/
    └── ...
```

**Recomendaciones:**
- Minimo 50 imagenes por clase para resultados decentes
- 200+ imagenes por clase para precision alta
- Variar angulos, iluminacion y fondos
- Incluir imagenes con defectos si aplica

### Camaras (`/camera`)

Ver seccion [Camaras](#cámaras) mas abajo.

### Entrenamiento (`/train`)

Ver seccion [Entrenamiento del Modelo](#entrenamiento-del-modelo) mas abajo.

### Predicciones (`/predict`)

Ver seccion [Predicciones](#predicciones) mas abajo.

### QR Escaner (`/qr`)

Ver seccion [Escaner QR](#escáner-qr--códigos-de-barras) mas abajo.

### Historial (`/history`)

Muestra historial completo de:
- Entrenamientos (fecha, epocas, precision)
- Predicciones (imagen, clase, confianza)

### Configuracion (`/settings`)

- **Servidor**: Puerto, host, API key de triggers
- **Usuarios**: Crear, editar, eliminar usuarios admin
- **Backup**: Exportar/importar base de datos

### API Docs (`/api-docs`)

Documentacion interactiva de la API REST (Swagger UI).

---

## Cámaras

### Camaras USB

1. Conecta la camara USB al servidor
2. Ve a **Camaras** (`/camera`)
3. Clic en **"Escanear camaras USB"**
4. Se mostraran las camaras detectadas (indice 0-4)
5. Clic en **"Registrar"** junto a la camara deseada
6. Asigna nombre y (opcional) URL de callback

### Camaras IP

1. Ve a **Camaras** (`/camera`)
2. En el panel **"Anadir Camara IP"**:
   - **Nombre**: Identificador descriptivo
   - **Protocolo**: RTSP, HTTP, o HTTPS
   - **IP/Host**: Direccion IP de la camara
   - **Puerto**: Puerto de streaming (default: 554 para RTSP, 80 para HTTP)
   - **Ruta/Stream**: Ruta del stream (ej: `/stream1`)
   - **Usuario**: (opcional) Usuario de la camara
   - **Password**: (opcional) Password de la camara
3. Clic en **"Registrar Camara IP"**

**URLs comunes de camaras IP:**

| Marca | URL RTSP |
|-------|----------|
| Hikvision | `rtsp://admin:password@ip:554/Streaming/Channels/101` |
| Dahua | `rtsp://admin:password@ip:554/cam/realmonitor?channel=1&subtype=0` |
| Reolink | `rtsp://admin:password@ip:554/h264Preview_01_main` |
| Generica | `rtsp://admin:password@ip:554/stream1` |

### Gestion de Camaras Registradas

En la seccion **"Camaras Registradas"** puedes:

- **Ver** (icono ojo): Selecciona la camara para vista previa en tiempo real
- **Editar** (icono lapiz): Modifica nombre, URL, callback
- **Eliminar** (icono papelera): Elimina la camara

### Captura de Fotos

1. Selecciona una camara haciendo clic en el icono de ojo
2. La vista previa aparecera en el panel derecho
3. Selecciona la clase destino en el dropdown
4. Clic en **"CAPTURAR AHORA"**
5. La foto se guarda en `dataset/[clase]/cam_[timestamp].jpg`

### Preview en Tiempo Real

Al seleccionar una camara, se inicia streaming MJPEG en el panel de vista previa. Puedes:
- Tomar snapshots individuales con **"Ver frame"**
- Capturar multiples fotos rapidamente con **"CAPTURAR AHORA"**

---

## Entrenamiento del Modelo

### Preparar el Dataset

1. Ve a **Dataset** (`/dataset`)
2. Crea las clases necesarias (ej: "manzana", "naranja", "platano")
3. Sube imagenes a cada clase (minimo 50 por clase recomendado)
4. Verifica que las imagenes sean variadas (angulos, iluminacion)

### Iniciar Entrenamiento

1. Ve a **Entrenamiento** (`/train`)
2. Revisa las estadisticas del dataset
3. Clic en **"Iniciar Entrenamiento"**
4. Observa el progreso en tiempo real via SSE:
   - Barra de progreso por fase
   - Loss y precision en tiempo real
   - Graficos de convergencia

### Fases del Entrenamiento

**Fase 1 - Entrenamiento del clasificador (Head)**
- Epocas: 10 (default)
- Learning rate: 0.001
- Solo entrena la ultima capa (clasificador)
- Congela el backbone de EfficientNet-B0

**Fase 2 - Fine-tuning**
- Epocas: 50 (default)
- Learning rate: 0.0002
- Descongela capas superiores del backbone
- Early stopping con paciencia de 10 epocas

### Caracteristicas del Entrenamiento

- **EMA (Exponential Moving Average)**: Promedia pesos del modelo para mejor generalizacion
- **MixUp**: Mezcla imagenes para augmentacion de datos
- **Label Smoothing**: Suaviza etiquetas para evitar sobreajuste
- **WeightedRandomSampler**: Balancea clases desbalanceadas automaticamente
- **Warmup**: Incrementa LR gradualmente al inicio
- **CosineAnnealing**: Reduce LR durante el entrenamiento
- **Gradient Clipping**: Previene explosiones de gradiente
- **AMP**: Mixed precision para GPU (mas rapido, menos memoria)

### Guardado del Modelo

Al finalizar el entrenamiento:
- Modelo guardado en `modelo_confecciones.pth`
- Clases guardadas en `clases.json`
- Historial guardado en la base de datos

### Entrenamiento via CLI

Tambien puedes entrenar directamente via linea de comandos:

```bash
python train.py
```

Esto inicia el mismo pipeline de entrenamiento que el panel web.

---

## Predicciones

### Via Panel Web

1. Ve a **Predicciones** (`/predict`)
2. Sube una imagen o arrastrala al area de carga
3. Clic en **"Predecir"**
4. Se mostrara:
   - Clase predicha
   - Nivel de confianza (%)
   - Grafico de probabilidades por clase
   - Indicador de confiabilidad (verde/rojo)

### Via API

```bash
curl -X POST http://localhost:8000/api/predecir -F "file=@imagen.jpg"
```

Respuesta:
```json
{
    "clase": "manzana",
    "confianza": 92.5,
    "probabilidades": {
        "manzana": 92.5,
        "naranja": 5.2,
        "platano": 2.3
    },
    "confiable": true
}
```

### TTA (Test-Time Augmentation)

Las predicciones usan TTA para mayor precision:
- 7 augmentaciones por imagen (default)
- Promedia las probabilidades de todas las augmentaciones
- Resultado mas estable y confiable

---

## API REST

### Endpoints de Clasificacion ML

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/api/status` | GET | Estado del modelo |
| `/api/clases` | GET | Lista de clases disponibles |
| `/api/predecir` | POST | Clasificar imagen (campo: `file`) |
| `/api/predecir-lote` | POST | Clasificar multiples imagenes |
| `/api/modelo/status` | GET | Estado detallado del modelo |
| `/api/dataset/stats` | GET | Estadisticas del dataset |
| `/api/entrenamientos` | GET | Historial de entrenamientos |
| `/api/predicciones` | GET | Historial de predicciones |

### Endpoints de Escaner QR

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/api/qr/camaras` | GET | Camaras disponibles para QR |
| `/api/qr/escanear` | POST | Escanea frame de camara (`camera_id`) |
| `/api/qr/escanear-imagen` | POST | Escanea imagen subida (`file`) |
| `/api/qr/historial` | GET | Historial de escaneos QR/barcode |

### Endpoints de Triggers IoT

Cada camara registrada tiene su propio endpoint de trigger. Requiere API key si esta configurada.

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/api/disparar/{slug}` | POST | Captura → ML + QR → resultado o webhook |
| `/api/disparar` | POST | Lista endpoints disponibles |

### Ejemplos de Uso

**Predecir imagen:**
```bash
curl -X POST http://localhost:8000/api/predecir -F "file=@foto.jpg"
```

**Disparar camara por slug:**
```bash
curl -X POST http://localhost:8000/api/disparar/camara-almacen
```

**Con API key:**
```bash
curl -X POST http://localhost:8000/api/disparar/camara-almacen \
  -H "X-API-Key: tu-api-key"
```

**Escanear QR desde camara:**
```bash
curl -X POST http://localhost:8000/api/qr/escanear \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "1"}'
```

### Respuesta de Trigger

```json
{
    "timestamp": "2026-05-18T22:30:00",
    "camera_slug": "camara-almacen",
    "camera_name": "Camara Almacen",
    "ml": {
        "confeccion": "confeccion2",
        "confianza": 89.86,
        "probabilidades": {
            "confeccion2": 89.86,
            "confeccion3": 7.21,
            "confeccion1": 2.93
        },
        "confiable": true
    },
    "qr": [
        {"type": "QR", "data": "https://example.com"}
    ],
    "qr_count": 1
}
```

---

## Integración IoT / MQTT

### Configurar MQTT

1. Ve a **Configuracion → MQTT** (`/mqtt`)
2. Ingresa los datos del broker:
   - **Host**: Direccion del broker MQTT (ej: `mqtt://192.168.1.100`)
   - **Puerto**: Puerto del broker (default: 1883)
   - **Usuario**: (opcional) Credenciales
   - **Password**: (opcional) Credenciales
3. Clic en **"Conectar"**

### Configurar Triggers

1. Ve a **Camaras** (`/camera`)
2. Edita una camara (icono lapiz)
3. Configura:
   - **Callback URL**: URL que recibira el resultado via POST
   - **Callback Activa**: Activa/desactiva el webhook
4. Guarda los cambios

### Flujo de Trigger

```
Sensor fisico (GPIO/webhook)
        ↓
POST /api/disparar/{slug}
        ↓
Captura frame de camara
        ↓
┌───────────────────┐
│ Clasificacion ML  │
│ Escaneo QR/Barcode│
└───────────────────┘
        ↓
┌───────────────────┐
│ Publicar MQTT     │
│ Webhook callback  │
└───────────────────┘
```

### Ejemplo con Node-RED

```json
{
    "id": "trigger-camara",
    "type": "http request",
    "url": "http://localhost:8000/api/disparar/camara-almacen",
    "method": "POST",
    "headers": {
        "X-API-Key": "tu-api-key"
    }
}
```

---

## Escáner QR / Códigos de Barras

### Via Panel Web

1. Ve a **QR Escaner** (`/qr`)
2. Selecciona una camara
3. Clic en **"Escanear QR"**
4. Los resultados se muestran en pantalla
5. El historial se guarda automaticamente

### Codigos Soportados

- **QR Code**: Codigos QR 2D
- **EAN-13**: Codigos de barras de productos
- **EAN-8**: Codigos de barras compactos
- **UPC-A**: Codigos de productos USA
- **Code 128**: Codigos alfanumericos
- **Code 39**: Codigos alfanumericos legacy

### Via API

**Escanear desde camara:**
```bash
curl -X POST http://localhost:8000/api/qr/escanear \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "1"}'
```

**Escanear imagen:**
```bash
curl -X POST http://localhost:8000/api/qr/escanear-imagen \
  -F "file=@qr_image.jpg"
```

---

## Seguridad

### Autenticacion

- Passwords hasheados con **bcrypt** (salt automatico)
- Migracion automatica de passwords SHA-256 legacy a bcrypt
- Sesiones firmadas con `itsdangerous` (token en cookie)
- Duracion de sesion: 7 dias

### CSRF Protection

- Tokens CSRF generados via HMAC-SHA256
- Verificacion en todos los POST (excepto `/login`, `/api/*`, `/camera/stream/*`)
- JS en `base.html` inyecta header `X-CSRF-Token` automaticamente

### API Key para Triggers

Para proteger los endpoints de trigger IoT:

1. Ve a **Configuracion** (`/settings`)
2. En "API Key de Triggers", genera o ingresa una clave
3. Guarda los cambios
4. Todos los requests a `/api/disparar/*` requeriran el header `X-API-Key`

### Mejores Practicas

- Cambia el password admin por defecto inmediatamente
- Usa `SECRET_KEY` como variable de entorno para mantener sesiones
- Configura API key si expones el servidor a la red
- Usa HTTPS via proxy inverso (nginx) en produccion

---

## Solución de Problemas

### El servidor no inicia

**Error: Puerto en uso**
```bash
# Verificar que no hay otro proceso en el puerto 8000
lsof -i :8000

# Matar proceso existente
kill -9 <PID>
```

**Error: Modulo no encontrado**
```bash
# Reinstalar dependencias
source venv/bin/activate
pip install -r requirements.txt
```

### La camara no funciona

**Camara USB no detectada:**
- Verifica que esta conectada: `ls /dev/video*`
- Prueba con otro indice USB
- Verifica permisos: `sudo usermod -aG video $USER`

**Camara IP no conecta:**
- Verifica la URL RTSP con VLC: `vlc rtsp://admin:password@ip:554/stream1`
- Verifica que la IP es accesible: `ping 192.168.1.100`
- Verifica firewall y puertos

**Error HEVC en logs:**
```
[hevc @ 0x...] Could not find ref with POC X
[hevc @ 0x...] Error constructing the frame RPS.
```
Estos errores son normales con camaras H.265/HEVC. No afectan la funcionalidad.

### Entrenamiento falla

**"Sin imagenes en el dataset":**
- Verifica que las carpetas de clase tienen imagenes
- Formatos soportados: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

**"CUDA out of memory":**
- Reduce `BATCH_SIZE` en `config.py` (ej: 16 o 8)
- Cierra otras aplicaciones que usen GPU

**"Modelo no converge":**
- Agrega mas imagenes al dataset
- Verifica que las clases estan bien separadas
- Aumenta epocas de entrenamiento

### Predicciones inexactas

- Entrena con mas imagenes (200+ por clase ideal)
- Verifica calidad de imagenes (no borrosas, buena iluminacion)
- Aumenta `TTA_AUGMENTATIONS` para predicciones mas estables
- Revisa que el modelo esta actualizado (re-entrenar si agregaste clases)

### Permisos de archivos

```bash
# Si hay errores de permisos
chmod -R 755 dataset/
chmod -R 755 data/
chmod -R 755 templates/
chmod -R 755 static/
```

---

## Licencia

[Licencia Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE)

**No se permite el uso comercial.**

---

## Creditos

Desarrollado como sistema de vision industrial de bajo costo para clasificacion de cajas de frutas.

Stack: FastAPI + PyTorch + OpenCV + Bootstrap 5
