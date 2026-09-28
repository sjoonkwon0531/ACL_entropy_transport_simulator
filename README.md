# ACL 수소 수송 시뮬레이터 (acl-hsim)

비정질 탄소 하부막(ACL, amorphous carbon layer) 3차원 셀 안에서 **중성 수소 H⁰**와
**수소 이온 H⁺**의 침투·포획·방출을 실단위(초, nm)로 계산하고 시각화하는 도구입니다.
SK하이닉스 산학과제 "High NA EUV MOR/UL 멀티스케일 전산 설계" (성균관대 SPMDL)의
Track B 계산 결과를 트렌드 수준으로 반영합니다.

두 가지 구현이 들어 있습니다.

| 파일 | 형태 | 용도 |
|---|---|---|
| `streamlit_app.py` | Streamlit 앱 (Python/numpy) | 배포·공유용 메인. 파라미터별 계산 캐시 + 시간 스크러버 |
| `hsim.html` | 단일 HTML (브라우저 JS) | 서버 없이 파일 하나로 여는 실시간 애니메이션판 |

## 모델 요약

종별 2-장(이동상 u, 포획상 w) 반응-확산 모델:

```
∂u/∂t = ∇·(D(x,T)∇u) − v_d ∂u/∂z − k_t(x)·u·(1 − w/N_t) + k_d(T)·w
∂w/∂t = +k_t·u·(1 − w/N_t) − k_d·w
```

- **재료 노브**: sp²/sp³ 비(H⁰ 기저 확산·H⁺ π-사이트 밀도), 연결 복잡도 엔트로피 S_G
  (상관길이 ~1.5 nm 무질서장의 진폭 → 채널링), 트랩 사이트 밀도 N_t (Langmuir 포화)
- **공정 노브**: 온도(Arrhenius), 기준 D₀·E_a(조정 가능한 가정값), H⁺ 드리프트(스택 대전 대용)
- **소스**: point / Gaussian 임펄스, 또는 PEB 동안 지속 유입 후 차단(상부 밀폐/개방)
- **출력**: XZ·XY 단면 칼라맵, SIMS 비교용 깊이 프로파일 c̄(z), 침투 깊이·포획 분율·
  기판 도달 시계열, 질량 보존 오차(수치 신뢰 지표), 리포트/CSV 다운로드

수치: 명시적 유한체적, 면 확산계수 조화평균, 드리프트 상류차분, dt 자동 안정화.
**모든 기본값은 트렌드 수준 가정**이며 절대 보정은 SIMS 프로파일과 fab 계측 상관으로
수행하는 것을 전제로 합니다 (앱 내 가정표 참조).

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Community Cloud 배포

1. 이 저장소를 GitHub에 push (private 저장소 가능)
2. https://share.streamlit.io → "Create app" → 저장소/브랜치 선택,
   Main file path에 `streamlit_app.py`
3. 배포 후 사이드바에서 조건을 바꾸면 조건당 1회 계산 후 캐시됩니다
   (기본 32급 격자·30 s 기준 약 20–40초)

## 저장소 구조

```
acl-hsim/
├── streamlit_app.py   # 메인 앱
├── hsim.html          # 단일 파일 웹판 (브라우저에서 바로 열기)
├── requirements.txt
└── README.md
```

## 주의

연구용 트렌드 도구입니다. 특정 라인·소재에 대한 정량 예측에 쓰려면 D₀, E_a, N_t를
해당 라인의 SIMS/계측 데이터로 회귀한 뒤 사용하십시오.
