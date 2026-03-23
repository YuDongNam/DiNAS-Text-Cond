# 신경망 구조 탐색을 위한 다중 조건부 그래프 확산 모델 (Editing Baseline)
Rohan Asthana, Joschua Conrad, Youssef Dawoud, Maurits Ortmanns, Vasileios Belagiannis

> **참고:** 이 리포지토리는 본래의 무에서 유를 창조하는 생성(Generation) 모델을 **그래프 수정(Graph Editing) 신경망 구조 탐색(NAS)** 베이스라인으로 작동할 수 있도록 수정된 브랜치입니다. `Parent Graph`(부모 그래프)와 `Text`(텍스트) 프롬프트를 조건으로 입력받아 일부분이 수정된 `Child Graph`(자식 그래프)를 생성하며, `NAD_triplet_dataset.jsonl` 데이터셋을 활용합니다.

이 리포지토리는 "Multi-conditioned Graph Diffusion for Neural Architecture Search" 논문의 코드를 포함하고 있습니다 [\[link\]](https://openreview.net/forum?id=5VotySkajV).

## 논문 초록
신경망 구조 탐색(NAS)은 거대하고 복잡한 탐색 공간을 조사하여 신경망 설계 과정을 자동화합니다. 탐색 능력을 향상시키기 위해, 우리는 이산 조건부 그래프 확산 프로세스(Discrete Conditional Graph Diffusion)를 사용하는 그래프 확산 기반 NAS 방식을 제시합니다. 우리는 높은 정확도 및 낮은 하드웨어 지연 시간과 같은 복합적인 제약 조건을 통제할 수 있도록 모델을 설계하였습니다. 

## 그래프 편집 파이프라인 (How it works)

본 베이스라인은 원본의 무작위 파이프라인을 다음과 같은 다중 조건(Multi-condition) 전략을 통해 **그래프 편집 프로세스**로 개조하여 작동합니다:

1. **데이터 전처리:** 확산(Diffusion)의 타겟이 되는 `Child Graph`는 희소 텐서(Sparse)로 로드되며, 기준이 되는 `Parent Graph`는 최대 노드 수 110에 맞춰 패딩된 밀집 텐서(Dense)로 고정 로드됩니다. 편집 지시문인 `Text Prompt`는 768차원 벡터로 임베딩됩니다.
2. **노이즈 스케줄링 (Forward Diffusion):** 정답인 `Child Graph`($z_t$)에만 이산 노이즈(Discrete noise)가 가해집니다. `Parent Graph`는 원본 구조를 안내하는 나침반 역할을 해야 하므로 노이즈가 섞이지 않고 깨끗하게 유지됩니다.
3. **조건 정보 병합 (Concatenation):** 노이즈 상태의 $z_t$를 신경망(Transformer)에 통과시키기 전, 온전한 `Parent Graph`의 노드 및 엣지 특징을 마지막 Feature 차원에 이어붙입니다 (`X_input = concat([X_t, X_parent])`).
4. **분류기 없는 안내 (Classifier-Free Guidance, CFG):** 편집 강도를 조절하기 위한 CFG 기법은 **텍스트 임베딩(Text condition)**에만 적용됩니다. 부모 그래프 정보는 원본 구조의 붕괴를 막기 위해 드롭아웃(Unconditioned) 패스에서도 항상 보존됩니다.
5. **추론 및 편집 (Inference):** 이산 확산 모델의 본래 수학적 근간을 유지하기 위해 $z_\tau$가 아닌 완전한 노이즈 상태 $z_T$에서부터 Denoising 샘플링을 시작합니다. 모델은 매 스텝마다 부모 그래프의 뼈대를 참조하며 텍스트가 지시한 대로 `Child Graph`를 깎아냅니다.

> **⚠️ 텍스트 임베딩(Text Embedding) 주의사항:** 텍스트 조건 제어를 위해서는 사전에 계산된 768차원 임베딩 텐서 캐시 파일이 필요합니다. 만약 설정(Config)에서 `embeddings_file` 경로가 비어있거나 파일이 없으면 **더미 임베딩(Dummy Embedding: 0과 1로 채워진 고정값)**으로 폴백(Fallback)되어 학습 및 생성이 진행됩니다. 실제 언어 지시 기반의 성공적인 생성 결과를 원하신다면 반드시 유효한 임베딩 파일을 연결해 주세요.

## 시작하기

수정된 DiNAS 편집 베이스라인을 사용하려면 다음 단계를 따르세요:

1. 리포지토리 클론: `git clone https://github.com/YuDongNam/DiNAS-Text-Cond.git`
2. `environment.yml`을 사용하여 기본 conda 환경을 로드하고, 데이터 연산을 위한 추가 의존성(`rdkit`, `pandas`, `seaborn`, `tensorflow-cpu`)을 설치하세요.
3. **NAD Triplet 편집 데이터셋으로 모델 학습 시작:**
   ```bash
   python main_reg_free.py dataset=nad
   ```
   이 명령어는 다중 조건(Multi-conditioned) 편집 환경 설정을 사용하여 확산 모델(Diffusion) 학습을 시작합니다.

4. **그래프 편집 태스크 테스트 (Dry-run):** 검증된 로컬 작동 확인용 스크립트입니다.
   ```bash
   python test_dryrun.py
   ```
   이 스크립트는 CPU와 호환되는 전체 순전파(Forward pass) 및 역전파(Backward pass)를 실행하며, 부모 그래프 구조(Dense)와 노이즈가 낀 자식 그래프 구조(Sparse)의 병합을 테스트합니다.

5. **평가 지표 (Evaluation Metrics):** 편집 태스크를 위한 평가 모듈을 제공합니다.
   ```bash
   python evaluate.py
   ```
   평가 스크립트는 다음과 같이 4가지 주요 지표를 계산합니다:
   - **Validity (유효성):** 논리적 그래프 제약을 통과하는 그래프의 비율.
   - **Uniqueness (고유성):** 생성된 샘플들 간의 구조적 다양성.
   - **Novelty (참신성):** 훈련 데이터셋(Training set)에 존재하지 않는 새로운 그래프의 비율.


## 논문 인용
```
@article{
asthana2024multiconditioned,
title={Multi-conditioned Graph Diffusion for Neural Architecture Search},
author={Rohan Asthana and Joschua Conrad and Youssef Dawoud and Maurits Ortmanns and Vasileios Belagiannis},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=5VotySkajV},
note={}
}
```

## 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE) 조건에 따라 배포됩니다.
