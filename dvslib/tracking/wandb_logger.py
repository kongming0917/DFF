"""Thin wandb wrapper.

enabled=False면 모든 호출이 no-op (wandb 없이도 코드가 그대로 돈다).
모든 방식이 이 wrapper로 같은 프로젝트(dvs-laser)에 같은 방식으로 기록한다.
"""

from typing import Dict, Optional, Sequence


class WandbLogger:
    def __init__(
        self,
        project: str = "dvs-laser",
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        tags: Optional[Sequence[str]] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.run = None
        if enabled:
            import wandb

            self.wandb = wandb
            self.run = wandb.init(
                project=project, name=name, config=config or {}, tags=list(tags or [])
            )

    def log(self, data: Dict, step: Optional[int] = None) -> None:
        if self.run:
            self.wandb.log(data, step=step)

    def summary(self, data: Dict) -> None:
        if self.run:
            self.wandb.summary.update(data)

    def finish(self) -> None:
        if self.run:
            self.run.finish()
