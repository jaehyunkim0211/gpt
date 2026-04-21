# 시계열 예측 & 분류 토이 프로젝트

두 종류의 시계열 데이터를 사용해 **예측(forecasting)** 과 **분류(classification)** 를
전통적 ML 베이스라인과 PyTorch GPU 딥러닝 모델로 비교하는 분석 환경입니다.

| 작업 | 데이터셋 | 베이스라인 | 딥러닝 |
|---|---|---|---|
| 예측 | Jena Climate (10분 간격, 14 변수) | Naïve · SARIMA · XGBoost | LSTM (multi-step 24h) |
| 분류 | UCI HAR (128 timestep × 9 채널) | RandomForest · XGBoost | 1D-CNN |

---

## 환경 요건

- **OS:** Windows (다른 OS도 가능, venv 활성화 명령만 다름)
- **Python:** 3.10 (3.11/3.12도 호환)
- **GPU:** NVIDIA GPU + CUDA 12.x 드라이버 권장 (CPU도 동작은 함)

확인:
```bash
py -3.10 --version
nvidia-smi      # CUDA Version 표시 확인
```

---

## 설치

### 1) 가상환경 생성 & 활성화
```bash
py -3.10 -m venv .venv
source .venv/Scripts/activate          # Windows Git Bash
# .venv\Scripts\Activate.ps1           # PowerShell
# .venv\Scripts\activate.bat           # cmd
python -m pip install --upgrade pip wheel setuptools
```

### 2) PyTorch GPU 설치 (별도 채널)
```bash
# CUDA 12.4 빌드 (RTX 30/40 시리즈 권장)
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8 환경이면
# pip install torch==2.5.1 torchvision==0.20.1 \
#     --index-url https://download.pytorch.org/whl/cu118
```

GPU 인식 확인:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True NVIDIA GeForce ...
```

### 3) 나머지 의존성
```bash
pip install -r requirements.txt
```

### 4) Jupyter 커널 등록
```bash
python -m ipykernel install --user --name=toy-ts --display-name "Python 3.10 (toy-ts)"
```
이후 노트북 우상단 커널 선택에서 **"Python 3.10 (toy-ts)"** 선택.

> 정확한 버전 잠금은 `requirements.lock.txt` 참고 (필요 시 `pip install -r requirements.lock.txt`).

---

## 데이터 다운로드

`src/data_loader.py`가 최초 실행 시 자동으로 받아 `data/raw/`에 캐시합니다.
미리 받아두려면:
```bash
python -c "from src.data_loader import fetch_all; fetch_all()"
```

| 파일 | 크기 | 출처 |
|---|---|---|
| `jena_climate_2009_2016.csv` | ~13MB | Max Planck Institute (TF mirror) |
| `UCI HAR Dataset/` | ~60MB | UCI Machine Learning Repository |

---

## 실행 순서

```bash
jupyter notebook notebooks/00_eda.ipynb                  # EDA + 무작위 샘플 뷰어
jupyter notebook notebooks/01_forecast_jena_climate.ipynb # 예측 모델 비교
jupyter notebook notebooks/02_classify_uci_har.ipynb     # 분류 모델 비교
```

각 노트북 마지막에서 `reports/figures/` 에 PNG 결과가 저장됩니다.

### 무작위 샘플 뷰어 (EDA 노트북)
- `seed=None`으로 호출하므로 **셀을 다시 실행할 때마다 다른 샘플**이 표시됩니다.
- 같은 결과를 원하면 셀에서 `seed=42` 등 정수를 지정하세요.

---

## 디렉터리 구조

```
.
├── README.md
├── requirements.txt           # 핵심 의존성 (PyTorch는 별도 설치)
├── requirements.lock.txt      # 전체 의존성 정확 버전
├── .gitignore
├── data/
│   ├── raw/                   # (gitignore) 다운로드 원본
│   └── processed/             # (gitignore) 가공 산출물
├── notebooks/
│   ├── 00_eda.ipynb
│   ├── 01_forecast_jena_climate.ipynb
│   └── 02_classify_uci_har.ipynb
├── src/
│   ├── data_loader.py         # 데이터셋 다운로드/로딩
│   ├── preprocessing.py       # 다운샘플, 스케일러, sliding window
│   ├── features.py            # lag / 시간 인코딩 feature
│   ├── models_baseline.py     # XGBoost / RandomForest
│   ├── models_dl.py           # LSTM forecaster, 1D-CNN classifier
│   ├── train.py               # 공통 학습 루프 (GPU 자동 사용)
│   ├── evaluate.py            # 회귀/분류 지표
│   └── visualize.py           # 모든 plot + 무작위 샘플 헬퍼
└── reports/
    └── figures/               # 결과 PNG 저장 위치
```

---

## 트러블슈팅

| 증상 | 해결 |
|---|---|
| `torch.cuda.is_available() == False` | nvidia-smi 결과의 CUDA Version 확인 → 일치하는 휠 채널(`cu118`/`cu121`/`cu124`)로 재설치 |
| Jupyter에서 `ModuleNotFoundError: src` | 노트북 첫 셀에서 `sys.path.insert(0, '..')` 실행 — 모든 노트북에 이미 포함됨 |
| 다른 가상환경 커널이 보임 | `jupyter kernelspec list`로 확인 후 `jupyter kernelspec uninstall <이름>` |
| ZIP 다운로드 실패 | 네트워크 확인 후 `data/raw/`의 부분 ZIP 삭제 → 재실행 |
