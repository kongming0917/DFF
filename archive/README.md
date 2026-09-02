# archive/

좌표가 고정된 fixed GT 데이터용 **1세대 코드**. 재현·참조용으로 보관하며, 활성 개발 대상이 아니다.

| 항목 | 설명 |
|---|---|
| `cnn_sim/` | CNN 회귀 1세대 |
| `filter_sim/` | 필터 휴리스틱 1세대 |
| `yolo_sim/` | YOLOv3-Tiny 검출 1세대 |
| `compare.py` | 위 세 방식의 fixed-GT 비교 스크립트 (절대 경로가 위 디렉토리를 참조) |

활성 코드는 brownian motion 데이터용 후속 버전(`cnn`, `yolo_brownian_sim`, `filter_brownian_sim`)과 공통 패키지 `dvslib/`를 사용한다.
