# Informe: MVP de monitoreo de mascotas (`perception_lab`)

- **Fecha:** 2026-11-01
- **Rama:** `feat/pet-activity-monitoring` (desde `main`, tras el PR #1)
- **Estado:** 15 commits locales, **sin push**
- **Plan de referencia:** `docs/plans/2026-11-01-pet-monitoring-plan.md`
- **Documentación de uso:** `docs/pet-monitoring.md`

## Resultado de validación

- `uv run pytest`: 224 tests pasan (80 del `main` y 144 nuevos).
- `uv run ruff check .` y `uv run ruff format --check .`: sin errores.
- `uv run lab eval` sobre la escena sintética: precisión 1.000, recall 0.833 (F1 0.909). La regla `resting_or_sleeping` no se dispara con el umbral por defecto de 300 s; con `stationary_s = 30` sí se dispara.
- Con ruido controlado (20 % de detecciones perdidas, 4 px de jitter): recall de eventos de 4/6 a 6/6 tras las correcciones, sin falsos eventos.
- Con ruido fuerte (40 % perdidas, 8 px de jitter): recall 0.833; se pierde `interacting`.

**Nada de esto es una medición real.** No hubo cámara, GPU ni modelos reales.

## Qué contiene

| Área | Contenido |
|---|---|
| Config y tipos | TOML validado a dataclasses con valores por defecto locales; `FrameBundle`, `Detection`, `SpatialObs`, `TrackObservation` |
| Fuentes | RealSense en vivo, `.bag` con grabación y sidecar JSON, video, escena sintética con etiquetas exactas; `discover` con avisos de USB 2 |
| Visión | Detector YOLOv8 ONNX (reutiliza `depth_volumetry.segmentation`); tracker estilo ByteTrack con emparejamiento por distancia de centros |
| Espacial | Plano de piso (RANSAC con límite de inclinación); zonas en polígonos de imagen con rango de elevación; profundidad robusta por animal con `depth_valid_ratio` y banderas de calidad |
| Eventos | 7 reglas deterministas; histéresis, cooldown y tolerancia a parpadeos; buffer de evidencia en memoria |
| VLM | Esquema JSON estricto con `unknown` permitido; prompt versionado; proveedores Ollama y OpenAI-compatible; worker asíncrono con cola acotada |
| Almacenamiento | SQLite con FTS5 (fallback LIKE); JPEG de evidencia por día; retención y `purge` |
| Notificaciones | `notify-send`, webhook solo en loopback, log |
| Dashboard | Servidor estándar en 127.0.0.1; estado, vista en vivo, eventos con evidencia, búsqueda, revisión correcto/incorrecto |
| Evaluación | Etiquetas, métricas por actividad, falsos eventos por día, precisión del VLM con `unknown` separado, id switches, `run.json` por corrida |
| CLI `lab` | `init-config`, `run`, `serve`, `discover`, `record`, `replay`, `calibrate`, `eval`, `bench detector`, `bench vlm`, `purge` |

## Correcciones durante el desarrollo

1. **Identidad de tracks en movimientos rápidos.** Con saltos o carreras la caja no se solapaba con la predicción y se creaba un ID nuevo. Se agregó emparejamiento por distancia de centros. En la escena sintética pasó de varios cambios de ID a 0 con 5 y 15 fps.
2. **Elevación con caja imprecisa.** Si la banda inferior de la caja veía el fondo, la altura cambiaba de un frame a otro. Ahora se usa la profundidad del cuerpo y se marca `elevation_from_body_depth`.
3. **Reglas que se reiniciaban con parpadeos.** Una detección perdida en un frame cortaba eventos largos. Se agregó `rules.flicker_tolerance_s`.
4. **Estacionario dentro del comedero.** La regla de reposo se disparaba mientras el animal comía. Ahora la deja a la regla de zona.

## Commits

```
da93a9d fix(events): tolerate missed detections and brief flicker in rules
bbe5571 fix(spatial): estimate elevation from body depth when the feet are not resolved
f28da94 feat(eval): add repeatable evaluation and on-machine benchmarks
fbd0fa0 feat(web): add a local dashboard for status, events and review
b98e768 fix(vision): keep track identity through fast motion at low frame rates
cb05325 feat(pets): wire the pet monitoring pipeline, notifications and CLI
c0e1be0 feat(vlm): add event-gated interpretation through local runtimes
7a68abd feat(store): add SQLite event history with search and retention
b23b585 feat(events): add event model, temporal rules and evidence buffer
6a8f1df feat(vision): add animal detection and multi-object tracking
76cb2d8 feat(sources): add synthetic, video and RealSense frame sources
abe7cfa feat(spatial): add floor plane, zones and per-track depth facts
69bf744 feat(lab): scaffold the perception_lab package
77f8131 refactor(realsense): expose the lazy pyrealsense2 loader
a10d342 docs: document the pet monitoring lab
```

## Pendiente antes de usar en casa

- Verificar en la PC: `nvidia-smi`, `lab discover`, `lab record`, y la versión de `pyrealsense2` para Python 3.12.
- Confirmar el nombre del modelo VLM (`qwen3-vl:2b`) y su licencia en la página oficial.
- Medir detector y VLM con `lab bench` y decidir CPU/GPU; si la GPU, cambiar a `onnxruntime-gpu`.
- Calibrar el plano de piso y las zonas con cinta métrica.
- Grabar el conjunto de validación y etiquetarlo para `lab eval`.
- Decidir si se hace push de la rama.

## Decisiones pendientes

- Dirección local de Ollama en la config (`127.0.0.1:11434`): se mantiene por ser local, pero `CLAUDE.md` pide no poner endpoints en el repo.
- Licencia de los pesos del detector (AGPL si son de Ultralytics).
