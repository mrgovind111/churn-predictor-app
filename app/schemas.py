from pydantic import BaseModel, Field

class ChurnInput(BaseModel):
    tenure: int = Field(..., ge=0, le=100, description="Months as customer")
    monthly_charges: float = Field(..., ge=0, le=500, description="Monthly bill")
    total_charges: float = Field(..., ge=0, description="Total billed so far")
    gender: str = Field(..., pattern="^(Male|Female)$", description="Male or Female")

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 12,
                "monthly_charges": 70.5,
                "total_charges": 850.0,
                "gender": "Male"
            }
        }
    }

class PredictionOutput(BaseModel):
    prediction: int
    label: str
    probability: float
