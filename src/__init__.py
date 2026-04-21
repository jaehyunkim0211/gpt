"""toy-ts 패키지 마커.

이 파일은 `src/` 디렉터리를 파이썬 패키지로 인식시키기 위한 빈 마커입니다.
노트북에서 `from src import preprocessing, train, ...` 형태로 임포트할 수 있습니다.

모듈 구성 개요:
    - data_loader        : Jena Climate / UCI HAR 다운로드·로딩
    - preprocessing      : 리샘플링·스케일링·슬라이딩 윈도우
    - features           : 베이스라인용 lag / 시간 인코딩 피처
    - models_baseline    : Naive, XGBoost, RandomForest 래퍼
    - models_dl          : PyTorch LSTM 예측기, 1D-CNN 분류기
    - train              : 공통 학습 루프 (GPU 자동 선택)
    - evaluate           : 회귀·분류 평가 지표
    - visualize          : 모든 plot 및 무작위 샘플 뷰어
"""
