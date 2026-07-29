# Serverless-воркер: ComfyUI + Wan 2.2 I2V fp8 + Lightning LoRA + чейнинг.
#
# Собран по конфигурации, которая реально заработала на RTX 4090 после
# долгой отладки. Три вещи здесь критичны, и все три выяснены на живом поде:
#
# 1. PyTorch 2.11+cu128, НЕ базовый 2.4.1.
#    На 2.4.1 нет DynamicVRAM, ComfyUI сваливается в legacy ModelPatcher,
#    и наложение LoRA на fp8-веса гарантированно даёт OOM в
#    requantize_from_float. С 2.11 включается ленивое применение патчей
#    ("400 patches attached") и всё влезает в 24 ГБ.
#
# 2. Именно cu128, не cu130.
#    Драйвер на подах RunPod - CUDA 12.8. Сборка под cu130 требует более
#    новый драйвер и полностью отключает CUDA.
#
# 3. Модели НЕ в образе, а на Network Volume (/runpod-volume).
#    Компромисс: образ остаётся ~10 ГБ вместо 30, но каждый холодный старт
#    тянет 20 ГБ весов с сетевого диска (~3 минуты). См. ROADMAP.md, там
#    описан вариант с запеканием моделей внутрь, если холодные старты
#    станут дороже генерации.
#
# Сборка:
#   docker build -t ВАШ_ЛОГИН/wan22-worker:latest .
#   docker push ВАШ_ЛОГИН/wan22-worker:latest

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    COMFY=/comfyui

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------- PyTorch: главный шаг
# Ставим до ComfyUI, чтобы его requirements не перетянули старую версию.
RUN pip install --no-cache-dir --force-reinstall \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# ------------------------------------------------------------------ ComfyUI
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git $COMFY
WORKDIR $COMFY
# --no-deps на torch-пакеты, иначе requirements.txt откатит версию обратно
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --force-reinstall --no-deps \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# --------------------------------------------------------------- custom nodes
RUN cd custom_nodes \
    && git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git \
    && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    && pip install --no-cache-dir gguf

# cv2 нужен VideoHelperSuite: без него нода не импортируется и mp4 не сохранить
RUN pip install --no-cache-dir opencv-python-headless imageio-ffmpeg
RUN pip install --no-cache-dir sageattention || echo "sageattention не собрался, работаем без него"

# ------------------------------------------------------- зависимости хендлера
RUN pip install --no-cache-dir runpod requests numpy boto3

# Модели на смонтированном томе. Путь обязан совпадать с тем, куда их положил
# install_ascii.sh: он запускался с COMFY_ROOT=/workspace на поде, значит внутри
# тома лежит подкаталог ComfyUI. В serverless тот же том монтируется
# в /runpod-volume, поэтому итоговый путь такой:
RUN printf '%s\n' \
    'comfyui:' \
    '  base_path: /runpod-volume/ComfyUI/' \
    '  diffusion_models: models/diffusion_models' \
    '  loras: models/loras' \
    '  text_encoders: models/text_encoders' \
    '  vae: models/vae' \
    > $COMFY/extra_model_paths.yaml

# Все файлы лежат в корне репозитория - так их можно просто перетащить
# в веб-интерфейс GitHub, не создавая подпапок.
COPY handler.py /handler.py
COPY workflow_config.json /workflow_config.json
COPY wan22_i2v_api.json /workflow.json
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Проверка на этапе сборки: если torch не тот, узнаем сейчас, а не на воркере
RUN python -c "import torch, sys; v=torch.__version__; print('torch', v); \
    sys.exit(0 if v.startswith(('2.8','2.9','2.10','2.11','2.12')) else 1)"

CMD ["/start.sh"]
