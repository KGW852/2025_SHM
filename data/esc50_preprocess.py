# data/esc50_preprocess.py

import os
import random
import shutil

from configs.config import Config
from data.tools_preprocess import Segmenter

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def get_configs():
    config = Config.exp1()
    cfg = config.config_dict
    return cfg
    
def save_stft_image(save_path: str, chunk: np.ndarray, sr: int, n_fft: int, hop_length: int, window: str, figsize: tuple=(2.56, 2.56)):
    # STFT
    D = librosa.stft(chunk, n_fft=n_fft, hop_length=hop_length, window=window)
    # Magnitude
    S_abs = np.abs(D)
    # db transform
    S_db = librosa.amplitude_to_db(S_abs, ref=np.max)
    
    plt.figure(figsize=figsize)
    plt.axis('off')
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis=None, y_axis=None)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_wav_to_spectrograms(cfg: dict, input_dir: str, output_dir: str):
    """
    input_dir 내 .wav 파일 -> segment 나눔 -> STFT -> output_dir 에 .png 저장
    """
    os.makedirs(output_dir, exist_ok=True)
    sr_target = cfg["preprocess"].get("sr", 44100)
    segmenter = Segmenter(cfg)

    n_fft = cfg["preprocess"].get("n_fft", 128)
    hop_length = cfg["preprocess"].get("hop_length", 14)
    window_fn = cfg["preprocess"].get("window", 'hann')

    for file in sorted(os.listdir(input_dir)):
        if not file.endswith('.wav'):
            continue
        file_path = os.path.join(input_dir, file)
        file_name = os.path.splitext(file)[0]

        # load wav
        signal, sr_orig = sf.read(file_path)
        if len(signal.shape) > 1:
            # 모노 채널만 사용
            signal = signal[:, 0]

        # 리샘플링
        if sr_orig != sr_target:
            signal = librosa.resample(signal, orig_sr=sr_orig, target_sr=sr_target)
            sr_orig = sr_target

        signal = signal.astype(np.float32)

        # segment 분할
        segments = segmenter.data_overlap(len(signal))
        for i, (start_idx, end_idx) in enumerate(segments):
            chunk = signal[start_idx:end_idx]
            seg_name = f"{file_name}-seg{i+1}.png"
            save_path = os.path.join(output_dir, seg_name)

            # STFT 이미지 저장
            save_stft_image(
                save_path=save_path,
                chunk=chunk,
                sr=sr_orig,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window_fn
            )

def split_normal_and_copy(cfg, source_spg_dir):
    """
    source_spg_dir 내 정상 파일(.png)을
    train / eval 로 분리하여 복사
      - train_dir : cfg["train"]["src_dir"]
      - eval 목록은 반환
    """
    # 만약 seed이 preprocess가 아닌 config 최상단에 있다면 cfg["seed"]로 가져오도록
    seed = cfg.get("seed", 42)
    random.seed(seed)

    nsplit = cfg["preprocess"].get("nsplit", 50)
    train_dir = cfg["train"]["src_dir"]

    if not os.path.isdir(source_spg_dir):
        raise FileNotFoundError(f"[Error] No source_spg_dir: {source_spg_dir}")

    normal_files = sorted([f for f in os.listdir(source_spg_dir) if f.endswith('.png')])
    if len(normal_files) < nsplit:
        raise ValueError(f"Normal png files: {len(normal_files)}, need nsplit={nsplit} for eval split.")

    train_data, eval_data = train_test_split(
        normal_files,
        test_size=nsplit,
        random_state=seed,
        shuffle=True
    )

    # train data copy
    os.makedirs(train_dir, exist_ok=True)
    for f in train_data:
        src = os.path.join(source_spg_dir, f)
        base_name = os.path.splitext(f)[0]
        dst_filename = f"airplane_{base_name}.png"
        dst = os.path.join(train_dir, dst_filename)
        shutil.copy2(src, dst)

    return eval_data

def process_test40(cfg, target_spg_dir):
    """
    target_spg_dir 내 이름에 '-40-' 들어있는 파일(라벨40) -> test/test40로 복사
    """
    test40_dir = cfg["test"]["tgt_dir"]
    os.makedirs(test40_dir, exist_ok=True)

    test_files = [f for f in os.listdir(target_spg_dir) if f.endswith('.png')]
    for f in test_files:
        parts = os.path.splitext(f)[0].split('-')
        # 예: 1-172649-A-40-seg27 => parts[3] == '40'
        if len(parts) >= 4 and parts[3] == '40':
            src = os.path.join(target_spg_dir, f)
            dst_filename = f"helicopter_{f}"
            dst = os.path.join(test40_dir, dst_filename)
            shutil.copy2(src, dst)


def process_eval01(cfg, eval_normal_data, source_spg_dir):
    """
    eval_normal_data(정상) -> eval/eval01 로 복사
      - airplane_ prefix
    """
    eval01_dir = cfg["eval"]["src_dir"]
    os.makedirs(eval01_dir, exist_ok=True)

    for nf in eval_normal_data:
        src = os.path.join(source_spg_dir, nf)
        base_name = os.path.splitext(nf)[0]
        dst_filename = f"airplane_{base_name}.png"
        dst = os.path.join(eval01_dir, dst_filename)
        shutil.copy2(src, dst)


def process_eval02(cfg, eval_normal_data, s1_spg_dir, source_spg_dir):
    """
    s1_spg_dir( seawave_ ), + 동일 개수만큼 normal( airplane_ ) => eval/eval02 로 복사
    """
    eval02_dir = cfg["test"]["src_dir"]  # yaml상 eval02를 test: src_dir로 정의한 경우
    os.makedirs(eval02_dir, exist_ok=True)

    s1_files = [f for f in os.listdir(s1_spg_dir) if f.endswith('.png')]

    # 1) s1 -> eval02 (seawave_)
    for f in s1_files:
        src = os.path.join(s1_spg_dir, f)
        dst_filename = f"seawave_{f}"
        dst = os.path.join(eval02_dir, dst_filename)
        shutil.copy2(src, dst)

    # 2) s1 개수만큼 normal → eval02
    needed = len(s1_files)
    if needed > 0:
        if needed > len(eval_normal_data):
            raise ValueError(f"[Error] s1 count={needed}, but available normal={len(eval_normal_data)}")

        # 동일 seed으로 샘플
        seed = cfg.get("seed", 42)
        random.seed(seed)
        sampled_normals = random.sample(eval_normal_data, needed)

        for nf in sampled_normals:
            src = os.path.join(source_spg_dir, nf)
            base_name = os.path.splitext(nf)[0]
            dst_filename = f"airplane_{base_name}.png"
            dst = os.path.join(eval02_dir, dst_filename)
            shutil.copy2(src, dst)

def main():
    cfg = get_configs()
    
    # 1) original to STFT
    source_spg_dir = os.path.join(cfg["root_dir"], "source")
    target_spg_dir = os.path.join(cfg["root_dir"], "target")
    s1_spg_dir     = os.path.join(cfg["root_dir"], "s1")

    print("[Step1] Source -> STFT")
    process_wav_to_spectrograms(cfg=cfg, input_dir=cfg["orig_dir"]["src_dir"], output_dir=source_spg_dir)

    print("[Step2] Target -> STFT")
    process_wav_to_spectrograms(cfg=cfg, input_dir=cfg["orig_dir"]["tgt_dir"], output_dir=target_spg_dir)

    print("[Step3] s1 -> STFT")
    process_wav_to_spectrograms(cfg=cfg, input_dir=cfg["orig_dir"]["s1_dir"], output_dir=s1_spg_dir)

    # 2) normal(source) -> train/eval split
    print("[Step4] Train & Eval Split for Normal(Source)")
    eval_normal_data = split_normal_and_copy(cfg, source_spg_dir=source_spg_dir)

    # 3) target label=40 -> test40
    print("[Step5] Label=40 => test40")
    process_test40(cfg, target_spg_dir=target_spg_dir)

    # 4) eval01
    print("[Step6] Eval01")
    process_eval01(cfg, eval_normal_data, source_spg_dir=source_spg_dir)

    # 5) eval02 (source + s1)
    print("[Step7] Eval02 => source(same count) + s1")
    process_eval02(cfg, eval_normal_data, s1_spg_dir=s1_spg_dir, source_spg_dir=source_spg_dir)

    print("[Done] ESC50 Preprocessing completed.")


if __name__ == "__main__":
    main()