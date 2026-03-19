from typing import Optional

from pydantic import BaseModel


class FraudPredictionRequest(BaseModel):
    TransactionDT: float
    TransactionAmt: float
    ProductCD: Optional[str] = None
    card4: Optional[str] = None
    card6: Optional[str] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = None
    DeviceInfo: Optional[str] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    C1: Optional[float] = None
    C4: Optional[float] = None
    C8: Optional[float] = None
    C10: Optional[float] = None
    C14: Optional[float] = None
    D1: Optional[float] = None
    D4: Optional[float] = None
    id_30: Optional[str] = None
    id_31: Optional[str] = None
    id_35: Optional[str] = None


class FraudPredictionResponse(BaseModel):
    fraud_probability: float
    fraud_prediction: int
    threshold_used: float
    decision: str
