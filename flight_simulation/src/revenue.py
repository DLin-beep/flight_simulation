from __future__ import annotations

from dataclasses import dataclass



@dataclass(frozen=True)

class RevenueModel:

    base_fare_usd: float

    ancillaries_usd_per_pax: float



def ticket_price_usd(distance_km: float, model: RevenueModel) -> float:



    return max(20.0, model.base_fare_usd + 0.08 * distance_km ** 0.85)



def revenue_usd(pax: int, price_usd: float, model: RevenueModel) -> float:

    return max(0.0, pax) * (max(0.0, price_usd) + max(0.0, model.ancillaries_usd_per_pax))



def default_revenue_model() -> RevenueModel:

    return RevenueModel(base_fare_usd=85.0, ancillaries_usd_per_pax=22.0)

