#!/usr/bin/env bash
# Поднимает ComfyUI в фоне и отдаёт управление хендлеру RunPod.
set -e

echo "worker: старт"

# Диагностика тома. Если Network Volume не подключён к эндпоинту, ComfyUI
# поднимется, но не найдёт ни одной модели, и ошибка вылезет только на первой
# генерации - в логах будет невнятное "model not found". Проверяем сразу.
MODELS=/runpod-volume/ComfyUI/models/diffusion_models
if [ ! -d "$MODELS" ]; then
  echo "worker: ВНИМАНИЕ - $MODELS не найден."
  echo "worker: причины по убыванию вероятности:"
  echo "worker:   1) Network Volume не подключён к эндпоинту"
  echo "worker:   2) регион эндпоинта не совпадает с регионом тома"
  echo "worker:   3) install_wan22.sh запускали без COMFY_ROOT=/runpod-volume"
  echo "worker: что реально лежит на томе:"
  ls -1 /runpod-volume 2>/dev/null | head || echo "worker:   том не смонтирован вообще"
else
  echo "worker: модели на томе:"
  ls -1 "$MODELS" | head
fi

python /comfyui/main.py \
    --listen 127.0.0.1 --port 8188 \
    --use-sage-attention \
    --disable-auto-launch --disable-metadata &

exec python -u /handler.py
