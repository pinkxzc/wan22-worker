# Serverless-воркер: ComfyUI + Wan 2.2 Remix I2V fp8 + чейнинг сегментов.
#
# Три вещи здесь критичны:
#
# 1. PyTorch 2.11+cu128, НЕ базовый 2.4.1.
#    На 2.4.1 нет DynamicVRAM, ComfyUI сваливается в legacy ModelPatcher,
#    и наложение LoRA на fp8-веса даёт OOM в requantize_from_float.
#    Версия ЗАФИКСИРОВАНА: раньше здесь стоял безверсионный `pip install torch`,
#    который тянул текущий latest, и сборки в разные дни отличались.
#
# 2. Именно cu128, не cu130.
#    Драйвер на подах RunPod - CUDA 12.8. Сборка под cu130 требует более
#    новый драйвер и полностью отключает CUDA.
#
# 3. Модели НЕ в образе, а на Network Volume (/runpod-volume).
#    Образ ~10 ГБ вместо 30, но каждый холодный старт тянет веса с сетевого
#    диска (~3 минуты).
#
# Сборка:
#   docker build -t ВАШ_ЛОГИН/wan22-worker:latest .
#   docker push ВАШ_ЛОГИН/wan22-worker:latest

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    COMFY=/comfyui

ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------- PyTorch: главный шаг
# Ставим до ComfyUI, чтобы его requirements не перетянули старую версию.
RUN pip install --no-cache-dir --force-reinstall \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
        --index-url https://download.pytorch.org/whl/cu128

# ------------------------------------------------------------------ ComfyUI
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git $COMFY
WORKDIR $COMFY
# --no-deps на torch-пакеты, иначе requirements.txt откатит версию обратно
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --force-reinstall --no-deps \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
        --index-url https://download.pytorch.org/whl/cu128

# --------------------------------------------------------------- custom nodes
# ComfyUI-GGUF оставлен на случай перехода на квантованные веса; в текущем
# workflow не используется (мёртвая нода UnetLoaderGGUF из него убрана).
RUN cd custom_nodes \
    && git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git \
    && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    && pip install --no-cache-dir gguf

RUN pip install --no-cache-dir opencv-python-headless imageio-ffmpeg
RUN pip install --no-cache-dir sageattention || echo "sageattention не собрался, работаем без него"

# ------------------------------------------------------- зависимости хендлера
RUN pip install --no-cache-dir runpod requests numpy boto3

# Модели на смонтированном томе. Добавлены clip: и unet: - CLIPLoader в части
# версий ComfyUI ищет в models/clip, а не только в models/text_encoders.
RUN printf '%s\n' \
    'comfyui:' \
    '  base_path: /runpod-volume/ComfyUI/' \
    '  diffusion_models: models/diffusion_models' \
    '  unet: models/diffusion_models' \
    '  loras: models/loras' \
    '  text_encoders: models/text_encoders' \
    '  clip: models/text_encoders' \
    '  vae: models/vae' \
    > $COMFY/extra_model_paths.yaml

# ------------------------------------------- ОПЦИЯ: модели внутрь образа
# Логи показывают ~85 секунд на каждую подкачку эксперта: 13626 МБ с сетевого
# тома на 160 МБ/с. Свопы неизбежны (high 13.6 + low 13.6 + энкодер 10.8 = 38 ГБ
# при 24 ГБ VRAM), поэтому единственный радикальный способ - положить веса на
# локальный диск воркера.
#
# Цена: образ вырастает с ~10 до ~40 ГБ, первый pull на новом воркере долгий.
# Выигрыш: минус ~170 секунд на каждый сегмент.
#
# Чтобы включить - положите модели рядом с Dockerfile в папку models/ и
# раскомментируйте. Тогда extra_model_paths.yaml ниже уже не нужен.
#
# COPY models/diffusion_models /comfyui/models/diffusion_models
# COPY models/text_encoders    /comfyui/models/text_encoders
# COPY models/loras            /comfyui/models/loras
# COPY models/vae              /comfyui/models/vae

COPY handler.py /handler.py
COPY workflow_config.json /workflow_config.json
COPY wan22_i2v_api.json /workflow.json
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Проверка на этапе сборки: если torch не тот, узнаем сейчас, а не на воркере
RUN python -c "import torch, sys; v=torch.__version__; print('torch', v); \
    sys.exit(0 if v.startswith('${TORCH_VERSION}') else 1)"

CMD ["/start.sh"]
