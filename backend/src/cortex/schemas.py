"""Pydantic request / response models."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ROISeries(BaseModel):
    network: str
    activity: list[float] = Field(description="One value per second of stimulus.")


class PredictResponse(BaseModel):
    n_timesteps: int = Field(description="Seconds of stimulus (1 TR = 1s).")
    n_vertices: int = 20484
    networks: list[ROISeries]


class DiffMoment(BaseModel):
    t_seconds: int
    network: str
    delta: float = Field(description="Signed delta (B minus A) of network activity at t.")


class DiffResponse(BaseModel):
    n_timesteps: int
    networks: list[ROISeries]            # B minus A per network
    abs_delta_total: dict[str, float]    # how much each network differs overall
    top_moments: list[DiffMoment]
