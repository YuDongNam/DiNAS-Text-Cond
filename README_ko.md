# 신경망 구조 탐색을 위한 다중 조건부 그래프 확산 모델 (Editing Baseline)
Rohan Asthana, Joschua Conrad, Youssef Dawoud, Maurits Ortmanns, Vasileios Belagiannis

> **참고:** 이 리포지토리는 본래의 무에서 유를 창조하는 생성(Generation) 모델을 **그래프 수정(Graph Editing) 신경망 구조 탐색(NAS)** 베이스라인으로 작동할 수 있도록 수정된 브랜치입니다. `Parent Graph`(부모 그래프)와 `Text`(텍스트) 프롬프트를 조건으로 입력받아 일부분이 수정된 `Child Graph`(자식 그래프)를 생성하며, `NAD_triplet_dataset.jsonl` 데이터셋을 활용합니다.

이 리포지토리는 "Multi-conditioned Graph Diffusion for Neural Architecture Search" 논문의 코드를 포함하고 있습니다 [\[link\]](https://openreview.net/forum?id=5VotySkajV).

## 논문 초록
신경망 구조 탐색(NAS)은 거대하고 복잡한 탐색 공간을 조사하여 신경망 설계 과정을 자동화합니다. 탐색 능력을 향상시키기 위해, 우리는 이산 조건부 그래프 확산 프로세스(Discrete Conditional Graph Diffusion)를 사용하는 그래프 확산 기반 NAS 방식을 제시합니다. 우리는 높은 정확도 및 낮은 하드웨어 지연 시간과 같은 복합적인 제약 조건을 통제할 수 있도록 모델을 설계하였습니다. 

## 시작하기

수정된 DiNAS 편집 베이스라인을 사용하려면 다음 단계를 따르세요:

1. 리포지토리 클론: `git clone https://github.com/YuDongNam/DiNAS-Text-Cond.git`
2. `environment.yml`을 사용하여 기본 conda 환경을 로드하고, 데이터 연산을 위한 추가 의존성(`rdkit`, `pandas`, `seaborn`, `tensorflow-cpu`)을 설치하세요.
3. **NAD Triplet 편집 데이터셋으로 모델 학습 시작:**
   ```bash
   python main_reg_free.py dataset=nad
   ```
   이 명령어는 다중 조건(Multi-conditioned) 편집 환경 설정을 사용하여 확산 모델(Diffusion) 학습을 시작합니다.

4. **평가 지표 (Evaluation Metrics):** 편집 태스크를 위한 평가 모듈을 제공합니다.
   ```bash
   python evaluate.py
   ```
   평가 스크립트는 다음과 같이 4가지 주요 지표를 계산합니다:
   - **Validity (유효성):** 논리적 그래프 제약을 통과하는 그래프의 비율.
   - **Uniqueness (고유성):** 생성된 샘플들 간의 구조적 다양성.
   - **Novelty (참신성):** 훈련 데이터셋(Training set)에 존재하지 않는 새로운 그래프의 비율.
   - **Latency (지연 시간):** 그래프 깊이 및 노드 수를 기반으로 한 시뮬레이션 지연 시간.

5. **그래프 편집 태스크 테스트 (Dry-run):** 검증된 로컬 작동 확인용 스크립트입니다.
   ```bash
   python test_dryrun.py
   ```
   이 스크립트는 CPU와 호환되는 전체 순전파(Forward pass) 및 역전파(Backward pass)를 실행하며, 부모 그래프 구조(Dense)와 노이즈가 낀 자식 그래프 구조(Sparse)의 병합을 테스트합니다.


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
