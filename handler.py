"""
RunPod Serverless handler: фото + промт -> видео любой длины.

ГЛАВНОЕ АРХИТЕКТУРНОЕ РЕШЕНИЕ
-----------------------------
Весь чейнинг из 6 сегментов происходит ВНУТРИ одного вызова handler.
Соблазн сделать иначе — послать 6 отдельных запросов в эндпоинт — обойдётся
на 71% дороже: каждый запрос заново поднимает воркер и грузит 22 ГБ весов.

    чейнинг внутри одного запроса : $0.193 за 30-сек ролик
    6 отдельных запросов          : $0.330

Внутри одного вызова модель загружается один раз и остаётся в VRAM
между сегментами. Это же даёт и выигрыш по времени: 6 сегментов подряд
идут без пауз на переинициализацию.

ЧТО ВАЖНО ВЫСТАВИТЬ В НАСТРОЙКАХ ЭНДПОИНТА
------------------------------------------
  Execution Timeout >= 1200 сек. Дефолтные 600 не хватит: 30-секундный
  ролик занимает ~10 минут вместе с холодным стартом, и задача будет
  убита ровно перед возвратом результата.
  Idle Timeout = 30 сек. Больше — платите за простой, меньше — теряете
  переиспользование воркера на серии запросов.

Вход:
    {"input": {"image": "<url или base64>", "prompt": "...", "seconds": 30}}
Выход:
    {"video_url": "..."} если настроен S3, иначе {"video_base64": "..."}
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np
import requests
import runpod

COMFY = "http://127.0.0.1:8188"
WORKFLOW = Path(os.environ.get("WORKFLOW_PATH", "/workflow.json"))
# Соответствие "роль ноды -> её ID" собирает inspect_workflow.py.
# Ищем по ID, а не по Title: схема Wan 2.2 - сабграф, при экспорте в API
# ноды разворачиваются в плоский список с составными ID вида "130:110",
# и заголовки туда доезжают не всегда.
CONFIG = Path(os.environ.get("WORKFLOW_CONFIG", "/workflow_config.json"))

FPS = 16                      # нативная частота Wan 2.2 A14B, менять нельзя
FRAMES_PER_SEGMENT = 81       # 81/16 = 5.06 сек. Должно быть вида 4n+1
PIXEL_BUDGET = 832 * 480      # ~400k пикселей. Держим постоянным: пропорции берём
                              # из фото, но площадь не меняем, иначе время и цена
                              # генерации поехали бы вслед за форматом снимка
DIM_STEP = 16                 # Wan требует размеры, кратные 16
MIN_DIM, MAX_DIM = 320, 1280
STEPS = 4                     # с Lightning LoRA. Без неё было бы 20 и в 5 раз дольше
CFG = 1.0                     # ровно 1.0, иначе картинка пережжённая
TAIL_OFFSET = 3               # опорный кадр берём за 3 до конца: последний самый мыльный
REFRESH_EVERY = 3             # каждые N сегментов подмешиваем исходное фото
REFRESH_BLEND = 0.15
MAX_SEGMENTS = 24             # предохранитель: 24 x 5.06 = ~2 минуты

S3_BUCKET = os.environ.get("S3_BUCKET")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")


# ------------------------------------------------------------- ожидание ComfyUI

def fit_dimensions(image_path: Path) -> tuple[int, int]:
    """
    Подбирает размер кадра под пропорции исходного фото.

    Раньше здесь было жёстко 832x480, из-за чего вертикальные снимки и кадры
    16:9 обрезались. Теперь берём соотношение сторон из самого фото, а площадь
    оставляем прежней - так ничего не режется, но время генерации не скачет
    от формата к формату.

    Примеры того, что получается:
        4:3   ->  736 x 544
        16:9  ->  864 x 480
        9:16  ->  480 x 864
        1:1   ->  640 x 640
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return 832, 480
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return 832, 480

    aspect = w / h
    new_h = (PIXEL_BUDGET / aspect) ** 0.5
    new_w = new_h * aspect

    def snap(v: float) -> int:
        v = int(round(v / DIM_STEP) * DIM_STEP)
        return max(MIN_DIM, min(MAX_DIM, v))

    return snap(new_w), snap(new_h)


def wait_for_comfy(timeout: int = 300) -> None:
    """Воркер стартует раньше, чем ComfyUI успевает подняться."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{COMFY}/system_stats", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError("ComfyUI не поднялся за отведённое время")


# ------------------------------------------------------------------ ComfyUI API

def node_by_role(wf: dict, cfg: dict, role: str, required: bool = True) -> dict | None:
    """Достаёт ноду по роли из workflow_config.json."""
    nid = cfg.get(role)
    if nid is None or nid not in wf:
        if required:
            raise RuntimeError(
                f"В workflow_config.json нет роли '{role}' либо такой ноды нет в схеме. "
                f"Прогоните inspect_workflow.py на экспортированном JSON."
            )
        return None
    return wf[nid]


def upload_image(path: Path) -> str:
    with open(path, "rb") as f:
        r = requests.post(
            f"{COMFY}/upload/image",
            files={"image": (path.name, f, "image/png")},
            data={"overwrite": "true"},
            timeout=120,
        )
    r.raise_for_status()
    return r.json()["name"]


def run_segment(template: dict, cfg: dict, prompt: str, start: Path,
                seed: int, workdir: Path, size: tuple[int, int]) -> Path:
    wf = json.loads(json.dumps(template))  # глубокая копия: шаблон не мутируем

    node_by_role(wf, cfg, "prompt")["inputs"]["text"] = prompt
    node_by_role(wf, cfg, "start_image")["inputs"]["image"] = upload_image(start)

    video = node_by_role(wf, cfg, "video_cfg")
    video["inputs"]["length"] = FRAMES_PER_SEGMENT
    video["inputs"]["width"], video["inputs"]["height"] = size

    # Шаги и cfg проставляем принудительно. Схема ComfyUI по умолчанию идёт
    # в режиме 10+10 шагов без ускорения - это пятикратная переплата по времени.
    high = node_by_role(wf, cfg, "sampler_high")
    low = node_by_role(wf, cfg, "sampler_low")
    for s in (high, low):
        s["inputs"]["steps"] = STEPS
        s["inputs"]["cfg"] = CFG          # ровно 1.0: Lightning обучена без CFG
    high["inputs"]["start_at_step"] = 0
    high["inputs"]["end_at_step"] = STEPS // 2
    low["inputs"]["start_at_step"] = STEPS // 2
    low["inputs"]["end_at_step"] = 10000

    # сид меняем на каждом сегменте, иначе движение будет буквально повторяться
    for n in wf.values():
        for key in ("seed", "noise_seed"):
            if key in n.get("inputs", {}):
                n["inputs"][key] = seed

    r = requests.post(f"{COMFY}/prompt", json={"prompt": wf}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"ComfyUI отклонил workflow: {r.text[:400]}")
    pid = r.json()["prompt_id"]

    while True:
        time.sleep(2)
        hist = requests.get(f"{COMFY}/history/{pid}", timeout=30).json()
        if pid not in hist:
            continue
        entry = hist[pid]
        if entry.get("status", {}).get("status_str") == "error":
            raise RuntimeError(f"Ошибка генерации: {json.dumps(entry['status'])[:400]}")
        meta = extract_output(entry)
        if meta:
            dst = workdir / f"seg_{uuid.uuid4().hex[:8]}.mp4"
            dst.write_bytes(requests.get(f"{COMFY}/view", params=meta, timeout=300).content)
            return dst


def extract_output(entry: dict) -> dict | None:
    for node in entry.get("outputs", {}).values():
        for key in ("gifs", "videos", "images"):
            if node.get(key):
                f = node[key][0]
                if f["filename"].endswith((".mp4", ".webm")):
                    return {"filename": f["filename"],
                            "subfolder": f.get("subfolder", ""),
                            "type": f.get("type", "output")}
    return None


# ----------------------------------------------------------- кадры, цвет, сборка

def grab_anchor(video: Path, out: Path) -> Path:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1 - TAIL_OFFSET))
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Не удалось прочитать кадр из {video}")
    cv2.imwrite(str(out), frame)
    return out


def refresh_anchor(anchor: Path, original: Path) -> None:
    """Подмешивает исходное фото — сбрасывает накопленный дрейф личности."""
    a, o = cv2.imread(str(anchor)), cv2.imread(str(original))
    if a is None or o is None:
        return
    o = cv2.resize(o, (a.shape[1], a.shape[0]))
    cv2.imwrite(str(anchor), cv2.addWeighted(a, 1 - REFRESH_BLEND, o, REFRESH_BLEND, 0))


def match_exposure(video: Path, reference: Path, out: Path) -> Path:
    """Выравнивает яркость сегмента по первому — убирает ступеньки на стыках."""
    def mid_lightness(v: Path) -> float:
        cap = cv2.VideoCapture(str(v))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return 128.0
        return float(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0].mean())

    delta = float(np.clip((mid_lightness(reference) - mid_lightness(video)) / 255.0, -0.3, 0.3))
    if abs(delta) < 0.01:            # в пределах шума — не трогаем
        shutil.copy(video, out)
        return out

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
        "-vf", f"eq=brightness={delta:.4f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", str(out),
    ], check=True)
    return out


def concat_and_upscale(segments: list[Path], workdir: Path, upscale: bool) -> Path:
    lst = workdir / "concat.txt"
    lst.write_text("".join(f"file '{s.resolve()}'\n" for s in segments))
    joined = workdir / "joined.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-vf", "format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-movflags", "+faststart", str(joined),
    ], check=True)

    if not upscale:
        return joined

    # Увеличиваем в 1.5 раза по обеим сторонам, а не до фиксированной высоты 720.
    # Иначе вертикальное видео 480x848 после "scale=-2:720" стало бы 408x720,
    # то есть уменьшилось бы вместо апскейла.
    out = workdir / "final_up.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(joined),
        "-vf", "scale=iw*1.5:ih*1.5:flags=lanczos,"
               "scale=trunc(iw/2)*2:trunc(ih/2)*2,unsharp=5:5:0.6",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-movflags", "+faststart", str(out),
    ], check=True)
    return out


# ------------------------------------------------------------------- ввод/вывод

def purge_comfy_dirs() -> None:
    """
    Чистит рабочие папки ComfyUI после генерации.

    Наш собственный workdir убирается в finally, а вот ComfyUI складывает
    результат каждого сегмента в /comfyui/output, а каждый загруженный опорный
    кадр - в /comfyui/input, и сам их никогда не удаляет. Пока воркер тёплый и
    обслуживает запрос за запросом, там копится около 20 МБ на каждый
    30-секундный ролик.

    При остановке воркера контейнер стирается целиком, так что это подстраховка
    на случай долгой непрерывной нагрузки, когда воркер живёт часами.
    """
    for sub in ("output", "input", "temp"):
        d = Path("/comfyui") / sub
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.name.startswith("."):      # служебные файлы ComfyUI не трогаем
                continue
            try:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
            except Exception:
                pass


def fetch_input_image(src: str, dst: Path) -> Path:
    if src.startswith(("http://", "https://")):
        urllib.request.urlretrieve(src, dst)
    else:
        dst.write_bytes(base64.b64decode(src.split(",")[-1]))
    return dst


def deliver(path: Path) -> dict:
    """S3 если настроен, иначе base64. 30-сек ролик в 480p весит ~2-4 МБ."""
    if S3_BUCKET:
        import boto3
        key = f"videos/{uuid.uuid4().hex}.mp4"
        boto3.client("s3", endpoint_url=S3_ENDPOINT).upload_file(
            str(path), S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"}
        )
        base = S3_ENDPOINT or f"https://{S3_BUCKET}.s3.amazonaws.com"
        return {"video_url": f"{base.rstrip('/')}/{S3_BUCKET}/{key}"
                if S3_ENDPOINT else f"{base}/{key}"}
    return {"video_base64": base64.b64encode(path.read_bytes()).decode()}


# --------------------------------------------------------------------- handler

def handler(job: dict) -> dict:
    inp = job.get("input", {})
    prompt = inp.get("prompt")
    image_src = inp.get("image")
    if not prompt or not image_src:
        return {"error": "нужны поля 'image' и 'prompt'"}

    seconds = float(inp.get("seconds", 30))
    seed = int(inp.get("seed", 42))
    upscale = bool(inp.get("upscale", True))

    seg_len = FRAMES_PER_SEGMENT / FPS
    n = max(1, min(MAX_SEGMENTS, round(seconds / seg_len)))

    wait_for_comfy()
    template = json.loads(WORKFLOW.read_text())
    cfg = json.loads(CONFIG.read_text())

    workdir = Path(tempfile.mkdtemp(prefix="wan_"))
    try:
        original = fetch_input_image(image_src, workdir / "original.png")
        anchor = workdir / "anchor_000.png"
        shutil.copy(original, anchor)

        # Пропорции кадра берём из фото, чтобы 16:9 и вертикальные снимки
        # не обрезались. Клиент может задать размер явно через width/height.
        size = (int(inp["width"]), int(inp["height"])) \
            if inp.get("width") and inp.get("height") else fit_dimensions(original)

        started = time.time()
        raw: list[Path] = []

        # весь чейнинг здесь: модель загружена один раз и живёт в VRAM
        for i in range(n):
            raw.append(run_segment(template, cfg, prompt, anchor, seed + i, workdir, size))
            if i < n - 1:
                anchor = grab_anchor(raw[-1], workdir / f"anchor_{i + 1:03d}.png")
                if (i + 1) % REFRESH_EVERY == 0:
                    refresh_anchor(anchor, original)

        fixed = [raw[0]] + [
            match_exposure(s, raw[0], workdir / f"fixed_{i:03d}.mp4")
            for i, s in enumerate(raw[1:], start=1)
        ]
        final = concat_and_upscale(fixed, workdir, upscale)

        gpu_sec = time.time() - started
        result = deliver(final)
        result.update({
            "segments": n,
            "duration_sec": round(n * seg_len, 2),
            "gpu_seconds": round(gpu_sec, 1),
            # без холодного старта — его RunPod считает отдельно
            "est_cost_usd": round(gpu_sec * 1.10 / 3600, 4),
        })
        return result

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        purge_comfy_dirs()


runpod.serverless.start({"handler": handler})
