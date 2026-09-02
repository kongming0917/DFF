"""models layer — shared components and a model registry.

Responsibilities:
    - shared building blocks, reparameterization / QAT utils
    - model registry: build a model by name (experiment selects via config)

방식 고유 아키텍처(MobileOne, YOLOv3-Tiny 등)는 여기 등록만 하고 구현은 방식별로 둔다.
외부 공식 구현(mobileone_official.py 등)은 수정하지 않는다.
"""
