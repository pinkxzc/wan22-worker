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

  # Воркер ожидает чекпоинты v3.0. Если на томе лежат только v2.0, генерация
  # упадёт с "model not found" уже на первом сегменте - предупреждаем заранее.
  for f in Wan2.2_Remix_NSFW_i2v_14b_high_lighting_fp8_e4m3fn_v3.0.safetensors \
           Wan2.2_Remix_NSFW_i2v_14b_low_lighting_fp8_e4m3fn_v3.0.safetensors; do
    if [ ! -f "$MODELS/$f" ]; then
      echo "worker: ВНИМАНИЕ - нет $f"
      echo "worker:   скачайте с huggingface.co/FX-FeiHou/wan2.2-Remix (папка NSFW)"
      echo "worker:   либо верните имена v2.0 в wan22_i2v_api.json"
    fi
  done
fi

# Прогрев страничного кэша.
#
# Зачем. В логах каждое переключение эксперта high<->low даёт ~85 секунд
# "Model Initializing": 13626 МБ делить на 85 с - это ровно 160 МБ/с, скорость
# сетевого тома. Веса перечитываются по сети при каждом свопе, а свопятся они
# всегда: high 13.6 + low 13.6 + текстовый энкодер 10.8 = 38 ГБ при 24 ГБ VRAM.
#
# Если оперативной памяти хватает, один раз протащим файлы через кэш - дальше
# ComfyUI будет брать их из RAM, а не с сети. Нужно ~38 ГБ свободной памяти;
# при меньшем объёме прогрев бесполезен (кэш вытеснится), поэтому проверяем.
if [ "${WARM_MODELS:-1}" = "1" ] && [ -d "$MODELS" ]; then
  FREE_GB=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  NEED_GB=$(du -scBG "$MODELS"/*.safetensors 2>/dev/null | tail -1 | tr -dc '0-9')
  NEED_GB=${NEED_GB:-99}
  if [ "$FREE_GB" -gt "$NEED_GB" ]; then
    echo "worker: прогрев кэша (${NEED_GB} ГБ моделей, ${FREE_GB} ГБ свободно)"
    time cat "$MODELS"/*.safetensors > /dev/null 2>&1 || true
    echo "worker: прогрев завершён"
  else
    echo "worker: прогрев пропущен - нужно ${NEED_GB} ГБ, доступно ${FREE_GB} ГБ."
    echo "worker: первый сегмент будет медленным (~85 с на подкачку эксперта)."
    echo "worker: варианты - взять воркер с большей RAM либо запечь модели в образ."
  fi
fi

python /comfyui/main.py \
    --listen 127.0.0.1 --port 8188 \
    --use-sage-attention \
    --disable-auto-launch --disable-metadata &

exec python -u /handler.py
