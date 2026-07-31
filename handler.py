"""
RunPod Serverless handler: фото + промт -> видео любой длины.

ИСПРАВЛЕННАЯ ВЕРСИЯ. Что изменено против оригинала и почему:

1. QUALITY приведён к рекомендациям автора чекпоинта Remix (8 шагов / split 4
   без Lightning LoRA, 4 / 2 с ней). Было 20 шагов при cfg 3.5 по умолчанию -
   это пережигало дистиллированные веса: кислотный цвет, пластиковая кожа.

2. Граница high/low больше не steps // 2. Она задаётся уровнем сигмы (0.90 для
   I2V), а не долей шагов, и зависит от shift. См. moe_split(). При 4 шагах
   старая формула случайно давала верный ответ, при 20 - промахивалась на два
   шага, отдавая доводку деталей высокошумному эксперту.

3. Разрешение снапится к нативным бакетам Wan (1280x720 и т.д.), а не считается
   из постоянной площади. Было ~400k пикселей (480p) с произвольными сторонами
   вроде 736x544 - вне обучающих бакетов и вдвое ниже дефолта автора модели.

4. Промежуточные файлы больше не пережимаются. Раньше при 6 сегментах первый
   проходил семь поколений libx264: выравнивание экспозиции, пять попарных
   xfade и апскейл. Теперь вся сборка - один filter_complex и один энкод.

5. unsharp убран из финального фильтра: поверх lanczos он давал ореолы.

6. Опорный кадр берётся не из готового mp4, а прямо с выхода VAE отдельной
   веткой SaveImage - без промежуточной компрессии в цепочке чейнинга.

7. lora=0 действительно обходит ноду (перекоммутация графа), а не грузит её
   с нулевым весом.

8. Убрано неверное утверждение из старой шапки о том, что модель грузится один
   раз и остаётся в VRAM между сегментами. Логи это опровергают: в каждом
   промте есть "Model WAN21 ... 13626MB Staged". На 24 ГБ 4090 не помещаются
   high 13.6 ГБ + low 13.6 ГБ + текстовый энкодер 10.8 ГБ = 38 ГБ, поэтому
   эксперты свопаются всегда, и заявленная экономия 71% не достигается.

9. Дефолт качества переведён с high на balanced. По логам на 4090 разница
   между "4 шага + LoRA" и "8 шагов без LoRA" - от 60-70 до 250-460 секунд на
   сегмент. Почти всё это не шаги, а перечитывание весов с сетевого тома.

Вход:
    {"input": {"image": "<url или base64>", "prompt": "...", "seconds": 30,
               "quality": "balanced", "resolution": "720p"}}
Выход:
    {"video_url": "..."} если настроен S3, иначе {"video_base64": "..."},
    плюс "timings" - секунды на каждый сегмент, чтобы видеть распределение
    времени не залезая в логи воркера.

НАСТРОЙКИ ЭНДПОИНТА
    Execution Timeout >= 3000 сек. При quality=high сегмент занимает до 460 с,
    и шесть сегментов упираются ровно в 1800.
    Idle Timeout = 30 сек.
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
CONFIG = Path(os.environ.get("WORKFLOW_CONFIG", "/workflow_config.json"))

FPS = 16                      # нативная частота Wan 2.2 A14B, менять нельзя
FRAMES_PER_SEGMENT = 81       # 81/16 = 5.06 сек. Должно быть вида 4n+1
DIM_STEP = 16

# Нативные бакеты Wan 2.2. Модель обучена на них; произвольные размеры она
# переваривает, но артефактов заметно больше. Выбираем ближайший по пропорциям.
BUCKETS = {
    "720p": [(1280, 720), (720, 1280), (960, 960)],
    "480p": [(832, 480), (480, 832), (640, 640)],
}
DEFAULT_RESOLUTION = "720p"

# Режимы качества.
#
# Чекпоинты Wan2.2 Remix *_lighting_* - ускоренные сборки. Автор модели даёт
# две проверенные точки: 8 шагов при split 4 без Lightning LoRA, либо 4 шага
# при split 2 с ней. Всё, что сильно выше по шагам и cfg, работает хуже, а не
# лучше: дистилляция уже в весах, и высокий guidance её пережигает.
#
# cfg держим в районе 1.0. Если промт слушается плохо, поднимайте до 2.0-2.5
# по одному шагу и сравнивайте на фиксированном сиде - запас есть, но узкий.
#
# split=None означает "посчитать по сигме" (moe_split). Там, где стоит число,
# это рекомендация автора чекпоинта, она приоритетнее расчёта.
#
# ВРЕМЯ ЗАМЕРЕНО ПО ЛОГАМ RTX 4090 с моделями на сетевом томе:
#   4 шага + LoRA  ->  60-70 с/сегмент
#   8 шагов без LoRA -> 250-460 с/сегмент
# Разница в 4-6 раз, и она почти вся не в шагах, а в подкачке весов: без LoRA
# ComfyUI гоняет по 13.6 ГБ на каждое переключение эксперта. Поэтому дефолт -
# balanced (LoRA остаётся), а не high. См. раздел "Скорость" в README.
QUALITY = {
    #  имя         шаги  cfg   сила LoRA  split   замер на 4090 / сегмент
    "fast":     dict(steps=4,  cfg=1.0, lora=1.0, split=2),   # ~60-70 с
    "balanced": dict(steps=6,  cfg=1.0, lora=1.0, split=3),   # ~90-110 с
    "high":     dict(steps=8,  cfg=1.0, lora=0.0, split=4),   # ~250-460 с
    "max":      dict(steps=10, cfg=2.0, lora=0.0, split=None),# ~300-500 с
}
DEFAULT_QUALITY = "balanced"

SHIFT = 5.0                   # ModelSamplingSD3 в workflow
MOE_BOUNDARY = 0.90           # граница экспертов Wan 2.2: 0.90 для I2V, 0.875 для T2V

OVERLAP_FRAMES = 12           # 0.75 сек при 16 fps
REFRESH_EVERY = 3
REFRESH_BLEND = 0.15
MAX_SEGMENTS = 24

S3_BUCKET = os.environ.get("S3_BUCKET")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")


# ------------------------------------------------------------------ разрешение

def moe_split(steps: int, shift: float = SHIFT, boundary: float = MOE_BOUNDARY) -> int:
    """
    Сколько первых шагов ведёт высокошумный эксперт.

    Wan 2.2 - MoE из двух экспертов, и граница между ними задана уровнем шума,
    а не номером шага: sigma >= boundary - высокошумный, ниже - низкошумный.
    Расписание сигм зависит от shift в ModelSamplingSD3:

        sigma(t) = shift * t / (1 + (shift - 1) * t),   t_i = 1 - i / steps

    Считаем, сколько шагов стартуют выше границы. Для shift=5, boundary=0.90:

        steps=4  -> 2      steps=8  -> 3      steps=20 -> 8
        steps=6  -> 3      steps=10 -> 4

    Старое steps // 2 совпадает с этим только при 4 и 6 шагах; дальше оно
    отдаёт высокошумному эксперту лишние шаги, на которых он мылит детали.
    """
    n_high = 0
    for i in range(steps):
        t = 1.0 - i / steps
        sigma = shift * t / (1.0 + (shift - 1.0) * t)
        if sigma >= boundary:
            n_high += 1
    return max(1, min(steps - 1, n_high))


def fit_dimensions(image_path: Path, tier: str = DEFAULT_RESOLUTION) -> tuple[int, int]:
    """
    Подбирает ближайший нативный бакет Wan под пропорции фото.

    Раньше здесь сохранялась постоянная площадь ~400k пикселей при произвольных
    сторонах (736x544, 640x640 и т.п.). Пропорции держались, но размеры уходили
    из обучающих бакетов, и вдобавок это было 480p там, где автор чекпоинта
    рекомендует короткую сторону 720.
    """
    buckets = BUCKETS.get(tier, BUCKETS[DEFAULT_RESOLUTION])
    img = cv2.imread(str(image_path))
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        return buckets[0]
    aspect = img.shape[1] / img.shape[0]
    return min(buckets, key=lambda b: abs(b[0] / b[1] - aspect))


def wait_for_comfy(timeout: int = 300) -> None:
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
    nid = cfg.get(role)
    if nid is None or nid not in wf:
        if required:
            raise RuntimeError(
                f"В workflow_config.json нет роли '{role}' либо такой ноды нет в схеме."
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
                seed: int, workdir: Path, size: tuple[int, int],
                quality: dict) -> tuple[Path, Path | None]:
    """Возвращает (видео сегмента, опорный кадр в png или None)."""
    wf = json.loads(json.dumps(template))

    node_by_role(wf, cfg, "prompt")["inputs"]["text"] = prompt
    node_by_role(wf, cfg, "start_image")["inputs"]["image"] = upload_image(start)

    video = node_by_role(wf, cfg, "video_cfg")
    video["inputs"]["length"] = FRAMES_PER_SEGMENT
    video["inputs"]["width"], video["inputs"]["height"] = size

    steps = quality["steps"]
    split = quality.get("split") or moe_split(steps)
    split = max(1, min(steps - 1, split))

    high = node_by_role(wf, cfg, "sampler_high")
    low = node_by_role(wf, cfg, "sampler_low")
    for s in (high, low):
        s["inputs"]["steps"] = steps
        s["inputs"]["cfg"] = quality["cfg"]
    high["inputs"]["start_at_step"] = 0
    high["inputs"]["end_at_step"] = split
    low["inputs"]["start_at_step"] = split
    low["inputs"]["end_at_step"] = 10000

    # Lightning LoRA. При нулевой силе обходим ноду перекоммутацией: подаём
    # ModelSamplingSD3 напрямую с загрузчика UNET. Установка strength_model=0
    # ноду не выключает - файл всё равно грузится и занимает VRAM.
    if quality["lora"] <= 0.0:
        for samp_role, unet_role in (("sampling_high", "unet_high"),
                                     ("sampling_low", "unet_low")):
            samp = node_by_role(wf, cfg, samp_role, required=False)
            unet_id = cfg.get(unet_role)
            if samp is not None and unet_id:
                samp["inputs"]["model"] = [unet_id, 0]
    else:
        for role in ("lora_high", "lora_low"):
            node = node_by_role(wf, cfg, role, required=False)
            if node is not None:
                node["inputs"]["strength_model"] = quality["lora"]

    # Опорный кадр забираем прямо с выхода VAE, до сохранения в mp4.
    picker = node_by_role(wf, cfg, "anchor_pick", required=False)
    if picker is not None:
        picker["inputs"]["batch_index"] = max(0, FRAMES_PER_SEGMENT - 1 - OVERLAP_FRAMES)

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
        if not meta:
            continue

        tag = uuid.uuid4().hex[:8]
        dst = workdir / f"seg_{tag}.mp4"
        dst.write_bytes(requests.get(f"{COMFY}/view", params=meta, timeout=300).content)

        anchor = None
        amet = extract_anchor(entry, cfg.get("anchor_save"))
        if amet:
            anchor = workdir / f"anchor_{tag}.png"
            anchor.write_bytes(
                requests.get(f"{COMFY}/view", params=amet, timeout=300).content)
        return dst, anchor


def _meta(f: dict) -> dict:
    return {"filename": f["filename"],
            "subfolder": f.get("subfolder", ""),
            "type": f.get("type", "output")}


def extract_output(entry: dict) -> dict | None:
    for node in entry.get("outputs", {}).values():
        for key in ("gifs", "videos", "images"):
            for f in node.get(key, []):
                if f["filename"].endswith((".mp4", ".webm")):
                    return _meta(f)
    return None


def extract_anchor(entry: dict, node_id: str | None) -> dict | None:
    """Опорный кадр — PNG с ноды SaveImage, а не кадр из сжатого mp4."""
    if not node_id:
        return None
    node = entry.get("outputs", {}).get(node_id, {})
    for f in node.get("images", []):
        if f["filename"].lower().endswith(".png"):
            return _meta(f)
    return None


# ----------------------------------------------------------- кадры, цвет, сборка

def grab_anchor_fallback(video: Path, out: Path) -> Path:
    """Запасной путь, если в workflow нет ветки SaveImage для опорного кадра."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1 - OVERLAP_FRAMES))
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
    """Подмешивает исходное фото — сбрасывает накопленный дрейф внешности."""
    a, o = cv2.imread(str(anchor)), cv2.imread(str(original))
    if a is None or o is None:
        return
    o = cv2.resize(o, (a.shape[1], a.shape[0]))
    cv2.imwrite(str(anchor), cv2.addWeighted(a, 1 - REFRESH_BLEND, o, REFRESH_BLEND, 0))


def mid_lightness(v: Path) -> float:
    cap = cv2.VideoCapture(str(v))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return 128.0
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0].mean())


def assemble(segments: list[Path], workdir: Path, upscale: bool) -> Path:
    """
    Собирает все сегменты за ОДИН вызов ffmpeg и ОДИН энкод.

    В оригинале выравнивание экспозиции и каждая пара xfade кодировались
    отдельно, и для 30-секундного ролика первый сегмент проходил семь
    поколений libx264 crf 18-20. На градиентах это бандинг, на коже - блоки.

    Здесь выравнивание яркости (eq), все кроссфейды и опциональный апскейл
    собраны в один filter_complex. Ролик кодируется ровно один раз.
    """
    out = workdir / "final.mp4"
    seg_dur = FRAMES_PER_SEGMENT / FPS
    fade = OVERLAP_FRAMES / FPS

    # Яркость каждого сегмента подтягиваем к первому.
    ref = mid_lightness(segments[0])
    deltas = [0.0] + [
        float(np.clip((ref - mid_lightness(s)) / 255.0, -0.3, 0.3))
        for s in segments[1:]
    ]

    filters: list[str] = []
    labels: list[str] = []
    for i, d in enumerate(deltas):
        if abs(d) < 0.01:
            labels.append(f"{i}:v")
        else:
            filters.append(f"[{i}:v]eq=brightness={d:.4f}[e{i}]")
            labels.append(f"e{i}")

    prev = labels[0]
    for k in range(1, len(segments)):
        nxt = f"x{k}"
        filters.append(
            f"[{prev}][{labels[k]}]xfade=transition=fade"
            f":duration={fade:.3f}:offset={k * (seg_dur - fade):.3f}[{nxt}]"
        )
        prev = nxt

    # Апскейл без unsharp: поверх lanczos он рисовал ореолы по контурам.
    tail = "scale=iw*1.5:ih*1.5:flags=lanczos," if upscale else ""
    filters.append(f"[{prev}]{tail}scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p[out]")

    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for s in segments:
        cmd += ["-i", str(s)]
    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[out]",
        "-c:v", "libx264", "-preset", "slow", "-crf", "16",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


# ------------------------------------------------------------------- ввод/вывод

def purge_comfy_dirs() -> None:
    for sub in ("output", "input", "temp"):
        d = Path("/comfyui") / sub
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.name.startswith("."):
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

def spread_prompts(prompts: list[str], n: int) -> list[str]:
    """
    Раскладывает k промтов-сцен по n сегментам.

    Клиент мыслит сценами по ~5 секунд, а сегментов из-за перекрытия получается
    больше: 30 секунд это 7 сегментов, а не 6. Поэтому не требуем совпадения, а
    растягиваем что дали:

        1 промт  на 7 сегментов -> все одинаковые (поведение как раньше)
        3 промта на 7 сегментов -> [0,0,0,1,1,2,2]
        7 промтов на 7          -> один к одному
    """
    if not prompts:
        return []
    k = len(prompts)
    return [prompts[min(k - 1, i * k // n)] for i in range(n)]


def handler(job: dict) -> dict:
    inp = job.get("input", {})
    image_src = inp.get("image")

    # 'prompts' - список сцен, 'prompt' - одна строка на весь ролик.
    # Поддерживаем оба: старые клиенты продолжают слать 'prompt'.
    scenes = inp.get("prompts")
    if isinstance(scenes, str):
        scenes = [scenes]
    if not scenes:
        scenes = [inp["prompt"]] if inp.get("prompt") else []
    scenes = [s.strip() for s in scenes if isinstance(s, str) and s.strip()]

    if not scenes or not image_src:
        return {"error": "нужны поля 'image' и 'prompt' (или 'prompts')"}
    prompt = scenes[0]

    seconds = float(inp.get("seconds", 30))
    seed = int(inp.get("seed", 42))
    upscale = bool(inp.get("upscale", False))   # на 720p наивный апскейл не нужен
    tier = str(inp.get("resolution") or DEFAULT_RESOLUTION)

    seg_len = FRAMES_PER_SEGMENT / FPS
    step_len = (FRAMES_PER_SEGMENT - OVERLAP_FRAMES) / FPS
    n = 1 if seconds <= seg_len else 1 + round((seconds - seg_len) / step_len)
    n = max(1, min(MAX_SEGMENTS, n))

    quality = QUALITY.get(str(inp.get("quality") or DEFAULT_QUALITY),
                          QUALITY[DEFAULT_QUALITY])

    wait_for_comfy()
    template = json.loads(WORKFLOW.read_text())
    cfg = json.loads(CONFIG.read_text())

    workdir = Path(tempfile.mkdtemp(prefix="wan_"))
    try:
        original = fetch_input_image(image_src, workdir / "original.png")
        anchor = workdir / "anchor_000.png"
        shutil.copy(original, anchor)

        size = (int(inp["width"]), int(inp["height"])) \
            if inp.get("width") and inp.get("height") else fit_dimensions(original, tier)

        started = time.time()
        raw: list[Path] = []
        timings: list[float] = []
        seg_prompts = spread_prompts(scenes, n)

        for i in range(n):
            t0 = time.time()
            seg, seg_anchor = run_segment(template, cfg, seg_prompts[i], anchor,
                                          seed + i, workdir, size, quality)
            timings.append(round(time.time() - t0, 1))
            raw.append(seg)
            if i < n - 1:
                anchor = seg_anchor or grab_anchor_fallback(
                    seg, workdir / f"anchor_{i + 1:03d}.png")
                if (i + 1) % REFRESH_EVERY == 0:
                    refresh_anchor(anchor, original)

        t_assemble = time.time()
        final = assemble(raw, workdir, upscale)
        t_assemble = round(time.time() - t_assemble, 1)

        gpu_sec = time.time() - started
        result = deliver(final)
        result.update({
            "segments": n,
            "duration_sec": round(seg_len + (n - 1) * step_len, 2),
            "resolution": f"{size[0]}x{size[1]}",
            "quality": {**quality, "split": quality.get("split") or moe_split(quality["steps"])},
            "gpu_seconds": round(gpu_sec, 1),
            "est_cost_usd": round(gpu_sec * 1.10 / 3600, 4),
            # Первый сегмент почти всегда дольше остальных: веса ещё не в
            # страничном кэше. Если и последующие держатся на том же уровне -
            # значит кэш не работает, смотрите раздел "Скорость" в README.
            "timings": {"segments_sec": timings, "assemble_sec": t_assemble},
            "prompts_used": seg_prompts,
        })
        return result

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        purge_comfy_dirs()


runpod.serverless.start({"handler": handler})
