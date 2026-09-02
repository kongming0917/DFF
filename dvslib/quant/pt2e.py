#!/usr/bin/env python3
"""PT2E(Export 기반) QAT 유틸리티 — FPGA INT8 배포용. 모든 방식(CNN·YOLO·…)이 이 한 경로를 쓴다.

cnn/quantization.py에서 dvslib로 이동 (Phase 2). eager-mode QAT(모델별 QuantStub/fusion 수동 배치)를 **대체**한다. PT2E는 `torch.export`로
잡은 그래프 위에서 Quantizer가 observer/fake-quant를 자동 삽입하므로:
  - **모델 무관**: mobilenet_v2 / mobileone_s0 동일 경로 (eager는 MobileOne 전용이었음).
  - **Conv-BN fusion 자동**: MobileNetV2의 BatchNorm도 그래프에서 자동 fold (eager는 스킵됨).
  - FPGA 제약은 **Quantizer 한 곳**에만 기술 (현재 XNNPACK 대칭 INT8 베이스).

그래프 캡처 진입점은 torch 버전에 따라 다르다 (`_capture_graph`가 흡수):
  - torch 2.5+: `torch.export.export_for_training(...).module()` (권장)
  - torch 2.2~2.4: `torch._export.capture_pre_autograd_graph` (구 API, fallback)

  - get_fpga_quantizer : 대칭 INT8 Quantizer (FPGA 정수 누산 모델용)
  - prepare_qat        : 모델 → PT2E QAT 그래프 (fake-quant 삽입, 학습 가능)
  - convert            : QAT 학습 후 → INT8 추론 그래프
  - set_qat_mode       : exported 그래프의 train/eval 전환 (dvslib 루프의 set_mode 훅)
  - save_int8 / load_int8 : QAT 결과 저장·복원 (prepared state_dict → 재prepare·convert, 정확 왕복)

※ convert된 INT8 그래프의 단일 파일 저장(torch.save 통째 / torch.export.save .pt2)은 torch 2.5에서 실패한다
  (GraphModule pickle 재귀 오류, export 시 `train()` 미지원). FPGA weight 추출은 load_int8로 복원한 그래프를 순회한다.

표준 흐름:
    ex = (torch.randn(2, 5, 512, 512),)          # 그래프 캡처용 예시 입력 (batch≥2 권장, dynamic batch)
    m = get_model("mobileone_s0", input_channels=5)
    m.reparameterize()                            # MobileOne만: multi→single branch
    qat = prepare_qat(m, ex)                      # QAT 준비
    # ... QAT fine-tuning (qat를 일반 모델처럼 학습) ...
    int8 = convert(qat)                           # INT8 변환
"""

import torch
from torch.ao.quantization import (
    move_exported_model_to_train,
    move_exported_model_to_eval,
)
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)


def set_qat_mode(model, training: bool):
    """PT2E exported 모델의 train/eval 전환.

    PT2E exported 모델은 표준 `.train()/.eval()`을 지원하지 않으므로(NotImplementedError),
    전용 함수로 그래프 노드(BN·fake-quant)를 전환한다. dvslib의 trainer/evaluator에
    `set_mode` 훅으로 주입하면 공유 loop를 수정 없이 재사용할 수 있다 (공식 PT2E 패턴).
    """
    if training:
        move_exported_model_to_train(model)
    else:
        move_exported_model_to_eval(model)


def _capture_graph(model: torch.nn.Module, example_inputs):
    """torch 버전 무관 그래프 캡처 (export_for_training 우선, 구버전은 fallback).

    batch(dim 0)를 **dynamic**으로 선언한다. 안 하면 export가 example의 batch를 정적으로
    박아(specialization), `x.view(x.size(0), -1)`류 head(MobileOne)가 다른 batch에서
    shape mismatch로 깨진다. 모델 자체는 원래 모든 batch를 지원하므로, 이는 그 성질을
    export 후에도 유지시키는 표준 처리다. → example_inputs batch는 2 이상 권장.
    """
    from torch.export import Dim
    batch = Dim("batch")  # 모든 입력이 같은 batch를 공유
    dynamic_shapes = tuple({0: batch} for _ in example_inputs)
    try:
        from torch.export import export_for_training
        return export_for_training(model, example_inputs, dynamic_shapes=dynamic_shapes).module()
    except ImportError:
        from torch._export import capture_pre_autograd_graph
        return capture_pre_autograd_graph(model, example_inputs, dynamic_shapes=dynamic_shapes)


def get_fpga_quantizer(per_channel: bool = True) -> XNNPACKQuantizer:
    """대칭 INT8 Quantizer.

    - weight: per-channel(기본)/per-tensor **대칭** int8 (zero_point=0) — FPGA 정수 누산에 적합.
    - activation: per-tensor int8.
    - fbgemm의 `reduce_range`(x86 7비트 오버플로 회피책)는 **적용하지 않음** — FPGA엔 불필요.

    향후 FPGA 하드웨어 제약(PoT scale, 고정 bit-width 등)이 필요하면 이 함수만
    custom Quantizer로 교체하면 되고, 모델 코드는 건드릴 필요 없다.
    """
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(
        get_symmetric_quantization_config(is_per_channel=per_channel, is_qat=True)
    )
    return quantizer


def prepare_qat(model: torch.nn.Module, example_inputs, per_channel: bool = True):
    """모델을 PT2E QAT 그래프로 변환.

    Args:
        model: 원본 nn.Module. MobileOne은 `reparameterize()` 후 전달 권장
            (single-branch가 캡처되어 추론 구조와 일치).
        example_inputs: 그래프 캡처용 예시 입력 튜플. 예: `(torch.randn(1, 5, 512, 512),)`.
            shape의 H/W는 conv 양자화 파라미터와 무관하므로 작아도 무방하나, 모델이
            기대하는 채널 수는 맞춰야 한다.
        per_channel: weight per-channel 양자화 여부.

    Returns:
        QAT 준비된 GraphModule. 일반 모델처럼 forward/backward 가능 (fake-quant 삽입됨).
    """
    # QAT 캡처는 train 모드에서 — BatchNorm이 학습 형태로 잡혀야 prepare_qat_pt2e가 QAT BN을 처리.
    model = model.train()
    exported = _capture_graph(model, example_inputs)
    prepared = prepare_qat_pt2e(exported, get_fpga_quantizer(per_channel))
    return prepared


def convert(prepared):
    """QAT 학습이 끝난 PT2E 모델을 INT8 추론 그래프로 변환.

    반환된 GraphModule에는 `quantize_per_tensor`/`dequantize_per_tensor` 등의 양자화 노드가
    삽입돼 있다. FPGA용 weight/scale 추출은 이 그래프를 순회해 얻는다 (eager의
    `module._weight_bias()` 방식과 다름 — `export_mobileone_info.py` 참고).
    """
    return convert_pt2e(prepared)


def save_int8(prepared_model, path: str, config: dict = None) -> None:
    """QAT가 끝난 **prepared(fake-quant) 그래프**의 state_dict를 저장한다.

    convert된 INT8 그래프는 activation scale/zero_point가 그래프 노드의 **상수**로 박혀 state_dict에 없고,
    GraphModule 통째 pickle은 재귀 오류로 실패한다. 반면 prepared 그래프는 observer 통계·fake-quant 파라미터가
    모두 state_dict에 있으므로, 복원 시 같은 FP32 모델을 다시 prepare → state_dict 로드 → convert 하면
    **동일한 INT8 그래프**가 나온다 (검증: 7.1245px 왕복 일치).
    """
    torch.save({"model_state_dict": prepared_model.state_dict(), "config": config or {},
                "format": "pt2e_prepared"}, path)


def load_int8(path: str, build_fp32, example_inputs, per_channel: bool = True):
    """save_int8 산출물 → INT8 그래프 복원.

    build_fp32(): 학습 때와 같은 구조의 FP32 모델(가중치 무관, MobileOne은 reparameterize 후)을 만드는 callable.
    반환: (int8 GraphModule(eval), config)
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(ck, dict) and ck.get("format") == "pt2e_prepared"):
        raise ValueError(f"not a pt2e_prepared checkpoint (save_int8 산출물이 아님): {path}")
    prepared = prepare_qat(build_fp32().cpu(), example_inputs, per_channel=per_channel)
    prepared.load_state_dict(ck["model_state_dict"], strict=True)
    m = convert(prepared)
    set_qat_mode(m, False)
    return m, ck.get("config", {})
