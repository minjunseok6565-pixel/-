# calibration2 (scheme combo balance harness)

목적: (공격 스킴 × 수비 스킴) 조합이 과도하게 높은 승률/넷레이팅을 만드는지 빠르게 탐지합니다.
로스터 영향(선수 스탯 분산)을 줄이기 위해, `n_rosters`개의 **균형형 더미 로스터**를 만들고
각 로스터에서 스킴 조합만 바꾼 팀들이 서로 경기하도록 구성합니다.

## 실행
프로젝트 루트에서(패키지 import 경로가 맞는 곳에서) 실행하세요.

예)
python -m main.calibration2.run --mode swiss --n_rosters 8 --k_opponents 12 --legs 2 --seed 1234 --out calib2.json

모드:
- swiss: 각 조합이 랜덤 상대 k개와만 경기(빠르게 과강 조합 탐지)
- vs_baseline: 모든 조합이 baseline 조합과만 경기(회귀 테스트에 좋음)
- full_matrix: 모든 조합이 모든 조합과 경기(상성 매트릭스까지)

knobs:
- pure: 스킴 knob(샤프니스/스트렝스)를 1.0 고정
- variation: 좁은 분산으로 knob을 샘플링(실전 변동성 가정)
