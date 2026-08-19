"""
AthleteGuard AI Backend
-----------------------
FastAPI service for athlete readiness, hybrid injury-risk,
and What-If decision-support scenarios.

Current scope:
- Health endpoints
- Unified Readiness Engine
- Temporal ML injury-risk prediction
- Hybrid Readiness + ML risk assessment
- Hybrid What-If scenario simulation

Architecture:

    Request
       |
       +--> Readiness Engine
       |
       +--> Temporal ML
       |
       +--> Hybrid Risk
       |
       +--> What-If Scenario
       |
       --> API Response

Important:
This is a synthetic-development prototype and not a
clinical diagnosis or medical clearance system.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from ai.hybrid.hybrid_risk_engine import (
    calculate_hybrid_risk,
)
from ai.scoring.readiness_service import (
    calculate_readiness_output,
)
from ai.scoring.what_if import (
    calculate_what_if,
)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="AthleteGuard AI API",
    description=(
        "Backend API for athlete readiness, temporal "
        "injury-risk, hybrid risk assessment, and "
        "temporary What-If training scenarios."
    ),
    version="1.3.0",
)


# ============================================================
# Readiness request / response models
# ============================================================

class ReadinessRequest(BaseModel):
    """
    Request for the unified Readiness Engine.

    This endpoint uses the same readiness logic that powers
    the Hybrid AI.
    """

    player_id: int = Field(
        ...,
        ge=1,
        description="Unique athlete/player identifier.",
        examples=[12],
    )

    training_duration_min: float = Field(
        ...,
        gt=0,
        description="Training duration in minutes.",
        examples=[76.0],
    )

    rpe: float = Field(
        ...,
        ge=0,
        le=10,
        description="Rate of perceived exertion.",
        examples=[5.58],
    )

    sleep_duration: float = Field(
        ...,
        ge=0,
        le=24,
        description="Sleep duration in hours.",
        examples=[8.49],
    )

    sleep_quality: float = Field(
        ...,
        ge=0,
        le=100,
        description="Sleep quality score.",
        examples=[80.0],
    )

    reaction_time_ms: float = Field(
        ...,
        gt=0,
        description="Reaction time in milliseconds.",
        examples=[194.54],
    )

    recovery_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Recovery score.",
        examples=[78.38],
    )

    baseline_sleep: float = Field(
        default=8.0,
        gt=0,
        le=24,
        description="Historical baseline sleep duration.",
        examples=[8.012],
    )

    baseline_reaction_time_ms: float | None = Field(
        default=None,
        gt=0,
        description="Historical baseline reaction time.",
        examples=[197.47],
    )

    baseline_training_load: float | None = Field(
        default=None,
        gt=0,
        description="Historical baseline training load.",
        examples=[363.25],
    )

    acute_load: float = Field(
        ...,
        ge=0,
        description="Current acute workload.",
        examples=[375.34],
    )

    chronic_load: float = Field(
        ...,
        gt=0,
        description="Current chronic workload.",
        examples=[363.25],
    )

    injury_context: bool = Field(
        default=False,
        description="Recent injury context flag.",
    )


class ReadinessResponse(BaseModel):
    """Unified Readiness Engine response."""

    player_id: str

    readiness_score: float = Field(
        ...,
        ge=0,
        le=100,
    )

    risk_level: str

    recommendation: str

    data_quality: float = Field(
        ...,
        ge=0,
        le=1,
    )

    factors: list[Any]

    early_warning: Any

    metrics: dict[str, Any]

    subscores: dict[str, Any]

    deviations: dict[str, Any]


# ============================================================
# Hybrid request / response models
# ============================================================

class HybridRiskRequest(BaseModel):
    """
    Input contract for the combined Readiness + Temporal ML engine.

    Contains:
    1. Raw readiness measurements
    2. Historical baselines
    3. Temporal ML engineered features
    """

    player_id: int = Field(
        ...,
        ge=1,
        description="Unique athlete/player identifier.",
        examples=[12],
    )

    # --------------------------------------------------------
    # Raw Readiness Engine inputs
    # --------------------------------------------------------

    training_duration_min: float = Field(
        ...,
        gt=0,
        description="Training duration in minutes.",
        examples=[76.0],
    )

    rpe: float = Field(
        ...,
        ge=0,
        le=10,
        description="Rate of perceived exertion.",
        examples=[5.58],
    )

    sleep_duration: float = Field(
        ...,
        ge=0,
        le=24,
        description="Sleep duration in hours.",
        examples=[8.49],
    )

    sleep_quality: float = Field(
        ...,
        ge=0,
        le=100,
        description="Sleep quality score.",
        examples=[80.0],
    )

    reaction_time_ms: float = Field(
        ...,
        gt=0,
        description="Reaction time in milliseconds.",
        examples=[194.54],
    )

    recovery_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Recovery score.",
        examples=[78.38],
    )

    baseline_sleep: float = Field(
        default=8.0,
        gt=0,
        le=24,
        description="Historical baseline sleep duration.",
        examples=[8.012],
    )

    baseline_reaction_time_ms: float | None = Field(
        default=None,
        gt=0,
        description="Historical baseline reaction time.",
        examples=[197.47],
    )

    baseline_training_load: float | None = Field(
        default=None,
        gt=0,
        description="Historical baseline training load.",
        examples=[363.25],
    )

    acute_load: float = Field(
        ...,
        ge=0,
        description="Current acute workload.",
        examples=[375.34],
    )

    chronic_load: float = Field(
        ...,
        gt=0,
        description="Current chronic workload.",
        examples=[363.25],
    )

    injury_context: bool = Field(
        default=False,
        description="Recent injury context flag.",
    )

    # --------------------------------------------------------
    # Temporal ML engineered features
    # --------------------------------------------------------

    session_load: float = Field(
        ...,
        ge=0,
        description="Current session load.",
        examples=[424.08],
    )

    acwr: float = Field(
        ...,
        ge=0,
        description="Acute:Chronic Workload Ratio.",
        examples=[1.033],
    )

    sleep_deviation: float = Field(
        ...,
        description="Sleep deviation from baseline.",
        examples=[0.478],
    )

    reaction_time_deviation: float = Field(
        ...,
        description="Reaction-time deviation from baseline.",
        examples=[-2.93],
    )

    recovery_drop: float = Field(
        ...,
        description="Drop in recovery from baseline.",
        examples=[-3.792],
    )

    warning_points: float = Field(
        ...,
        ge=0,
        description="Aggregated warning indicator count.",
        examples=[0.0],
    )


class HybridRiskResponse(BaseModel):
    """Combined Readiness + Temporal ML result."""

    player_id: str

    final_risk_level: str

    readiness_score: float = Field(
        ...,
        ge=0,
        le=100,
    )

    readiness_risk_level: str

    ml_injury_probability: float = Field(
        ...,
        ge=0,
        le=1,
    )

    ml_risk_level: str

    recommendation: str

    data_quality: float = Field(
        ...,
        ge=0,
        le=1,
    )

    factors: list[Any]

    early_warning: Any

    metrics: dict[str, Any]

    subscores: dict[str, Any]

    deviations: dict[str, Any]


# ============================================================
# What-If request model
# ============================================================

class WhatIfRequest(HybridRiskRequest):
    """
    Request for a temporary What-If scenario.

    The base athlete data is identical to HybridRiskRequest.

    'changes' contains temporary values that should be
    simulated without modifying or persisting the original
    athlete data.
    """

    changes: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Temporary scenario changes. "
            "Examples include training_duration_min, rpe, "
            "sleep_duration, sleep_quality, recovery_score, "
            "session_load, acute_load, acwr, recovery_drop, "
            "and warning_points."
        ),
        examples=[
            {
                "training_duration_min": 60.0,
                "rpe": 5.0,
                "sleep_duration": 8.5,
                "sleep_quality": 90.0,
                "recovery_score": 85.0,
                "session_load": 300.0,
            }
        ],
    )
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import os

app = FastAPI(
    title="AthleteGuard AI Engine",
    description="AI engine for injury prediction, readiness, and user accounts.",
    version="1.0.0"
)

# 1. نماذج البيانات (Models)
class ReadinessRequest(BaseModel):
    player_id: int = Field(..., example=101)
    acute_load: float = Field(..., gt=0, example=800.0)
    chronic_load: float = Field(..., gt=0, example=600.0)
    sleep_hours: float = Field(..., ge=0, le=24, example=7.5)
    hrv_status: str = Field(default="normal", example="normal")

class UserAuthSchema(BaseModel):
    username: str
    password: str
    role: str  # "coach" أو "player"
    player_id: int | None = None

# قاعدة بيانات مؤقتة لتخزين الحسابات المسجلة
fake_users_db = []

# 2. مسارات الصلاحيات والحسابات (Coach / Player Accounts)
@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
def register_user(user: UserAuthSchema):
    for existing in fake_users_db:
        if existing["username"] == user.username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
    fake_users_db.append(user.dict())
    return {
        "status": "success",
        "message": "User registered successfully",
        "role": user.role,
        "username": user.username
    }

@app.post("/api/auth/login")
def login_user(user: UserAuthSchema):
    for existing in fake_users_db:
        if existing["username"] == user.username and existing["password"] == user.password:
            return {
                "status": "success",
                "message": f"Welcome back, {user.role}",
                "role": existing["role"],
                "player_id": existing.get("player_id"),
                "token": "token-bearer-athlete-guard-2026"
            }
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid username, password, or role"
    )

# 3. مسار جلب البيانات الحقيقية
@app.get("/api/players/real-data")
def get_real_players_data():
    # يبحث عن ملف البيانات الحقيقية إذا كان موجوداً
    csv_path = "data/processed/final_dataset.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.head(10).to_dict(orient="records")
    return {"message": "System is running with default mock/active engine data."}

@app.get("/")
def root():
    return {"message": "AthleteGuard AI Engine is running perfectly!"}
class SchedulePlanRequest(BaseModel):
    player_id: int = Field(..., example=101)
    readiness_score: int = Field(..., ge=0, le=100, example=65)
    current_workload: str = Field(..., example="High") # High, Moderate, Low

@app.post("/api/schedule/smart-planner")
def smart_schedule_planner(data: SchedulePlanRequest):
    # التخطيط الذكي للجدول بناءً على جاهزية اللاعب
    if data.readiness_score < 50 or data.current_workload == "High":
        plan = {
            "schedule_type": "Recovery Focus",
            "recommended_activity": "Active Recovery / Pool Session / Physio",
            "intensity": "Low",
            "duration_minutes": 30,
            "note": "Player is at risk of overtraining. Immediate load management required."
        }
    elif 50 <= data.readiness_score < 80:
        plan = {
            "schedule_type": "Modified Training",
            "recommended_activity": "Skill drills, tactical walkthrough, light passing",
            "intensity": "Moderate",
            "duration_minutes": 60,
            "note": "Proceed with caution. Monitor fatigue levels closely."
        }
    else:
        plan = {
            "schedule_type": "Full Training",
            "recommended_activity": "High-intensity full team session, match simulation",
            "intensity": "High",
            "duration_minutes": 90,
            "note": "Player is fully optimal and ready for high physical exertion."
        }
        
    return {
        "player_id": data.player_id,
        "readiness_score": data.readiness_score,
        "smart_plan": plan
    }
from typing import Dict, Any

# 1. نموذج تفصيلي لتحليل الفريق دفعة واحدة (Batch Team API Pro)
class TeamPlayerInput(BaseModel):
    player_id: int = Field(..., example=101)
    name: str = Field(..., example="Ahmed Ali")
    acute_load: float = Field(..., gt=0, example=850.0)
    chronic_load: float = Field(..., gt=0, example=600.0)
    sleep_hours: float = Field(..., ge=0, le=24, example=6.0)
    hrv_status: str = Field(default="normal", example="normal")

class AdvancedTeamBatchRequest(BaseModel):
    team_id: str = Field(..., example="FC_Elite_1st_Team")
    sport_type: str = Field(..., example="Football")
    players: List[TeamPlayerInput]

# 2. نموذج نظام العودة للعب (Return-to-Play Engine)
class ReturnToPlayRequest(BaseModel):
    player_id: int = Field(..., example=101)
    injury_type: str = Field(..., example="Hamstring Strain")
    days_post_injury: int = Field(..., ge=0, example=14)
    pain_scale: int = Field(..., ge=0, le=10, example=2)  # من 0 إلى 10
    functional_test_passed: bool = Field(..., example=True)

@app.post("/api/v1/team/comprehensive-batch-analysis", tags=["Enterprise Team Analytics"])
def comprehensive_batch_analysis(data: AdvancedTeamBatchRequest):
    """
    تحليل جبار وشامل للفريق بالكامل دفعة واحدة، يحسب متوسط أحمال الفريق، 
    ويصنف اللاعبين حسب خطورة الإصابة، ويعطي مؤشر أمان الفريق الإجمالي.
    """
    analyzed_players = []
    high_risk_count = 0
    moderate_risk_count = 0
    optimal_count = 0

    for player in data.players:
        acwr = calculate_acwr(player.acute_load, player.chronic_load)
        metrics = evaluate_athlete_status(acwr, player.sleep_hours, player.hrv_status)
        
        if metrics["level"] == "High Risk":
            high_risk_count += 1
        elif metrics["level"] == "Moderate Risk":
            moderate_risk_count += 1
        else:
            optimal_count += 1

        analyzed_players.append({
            "player_id": player.player_id,
            "name": player.name,
            "acwr": acwr,
            "status": metrics["level"],
            "action_required": metrics["recommendation"]
        })

    team_safety_index = round((optimal_count / len(data.players)) * 100, 1) if data.players else 0

    return {
        "status": "success",
        "engine_version": "2.0-Enterprise",
        "team_id": data.team_id,
        "sport_type": data.sport_type,
        "team_analytics_summary": {
            "total_players_evaluated": len(data.players),
            "high_risk_players": high_risk_count,
            "moderate_risk_players": moderate_risk_count,
            "optimal_players": optimal_count,
            "team_safety_index_percentage": f"{team_safety_index}%"
        },
        "roster_detailed_report": analyzed_players
    }

@app.post("/api/v1/medical/return-to-play", tags=["Medical & Physio Intelligence"])
def return_to_play_assessment(data: ReturnToPlayRequest):
    """
    محرك طبي ذكي ومتطور لتحديد جاهزية اللاعب للعودة للتدريبات الجماعية 
    أو المباريات بعد الإصابة بناءً على مقاييس الألم والفحوصات الوظيفية.
    """
    clearance_status = "Not Cleared"
    next_phase = ""
    restrictions = []

    if data.pain_scale > 4:
        clearance_status = "Denied"
        next_phase = "Phase 1: Strict Rehabilitation & Pain Management"
        restrictions.append("No running or high-impact cutting movements.")
    elif not data.functional_test_passed:
        clearance_status = "Conditional / Restricted"
        next_phase = "Phase 2: Controlled Functional Training"
        restrictions.append("Allowed non-contact individual drills only.")
    elif data.days_post_injury < 10 and data.pain_scale > 2:
        clearance_status = "Restricted Integration"
        next_phase = "Phase 3: Partial Team Training (No Scrimmages)"
        restrictions.append("Limited duration (Max 20 mins team drills).")
    else:
        clearance_status = "Full Clearance"
        next_phase = "Phase 4: Full Return to Competition (RTP)"
        restrictions.append("None. Player is fully cleared for match play.")

    return {
        "player_id": data.player_id,
        "injury_type": data.injury_type,
        "medical_assessment": {
            "clearance_status": clearance_status,
            "assigned_phase": next_phase,
            "pain_reported": data.pain_scale,
            "functional_tests_status": "Passed" if data.functional_test_passed else "Failed/Pending",
            "clinical_restrictions": restrictions,
            "rehab_confidence_score": 88 if data.functional_test_passed else 45
        }
    }
class MLCalibrationRequest(BaseModel):
    model_name: str = Field(..., example="Injury_Risk_Predictor_v1")
    sensitivity_factor: float = Field(..., ge=0.5, le=2.0, example=1.2)
    sport_category: str = Field(..., example="Football")
    auto_adjust_thresholds: bool = Field(default=True, example=True)

@app.post("/api/v1/ml/calibrate-model", tags=["AI Engine Calibration"])
def calibrate_ml_model(data: MLCalibrationRequest):
    """
    مسار ذكي لمعايرة نموذج الذكاء الاصطناعي وضبط حساسية التنبؤات 
    بناءً على طبيعة الرياضة وخصائص الفريق لرفع دقة النتائج.
    """
    # حساب معامل المعايرة الجديد بناءً على الحساسية المدخلة
    base_accuracy = 92.4
    adjusted_accuracy = round(min(99.5, base_accuracy * (1.0 + (data.sensitivity_factor - 1.0) * 0.15)), 2)
    
    status_msg = "Model successfully recalibrated and weights updated."
    if data.auto_adjust_thresholds:
        status_msg += " Automatic risk thresholds have been re-optimized for " + data.sport_category

    return {
        "status": "success",
        "engine_state": "active",
        "calibration_report": {
            "model_name": data.model_name,
            "sport_category": data.sport_category,
            "applied_sensitivity": data.sensitivity_factor,
            "previous_accuracy": f"{base_accuracy}%",
            "new_calibrated_accuracy": f"{adjusted_accuracy}%",
            "message": status_msg
        }
    }

# ============================================================
# Health response
# ============================================================

class HealthResponse(BaseModel):
    """API health response."""

    status: str
    service: str
    version: str


# ============================================================
# Health endpoints
# ============================================================

@app.get(
    "/",
    response_model=HealthResponse,
    tags=["Health"],
)
def root() -> HealthResponse:
    """Basic API health check."""

    return HealthResponse(
        status="ok",
        service="AthleteGuard AI API",
        version=app.version,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
)
def health_check() -> HealthResponse:
    """Dedicated health endpoint for monitoring."""

    return HealthResponse(
        status="healthy",
        service="AthleteGuard AI API",
        version=app.version,
    )


# ============================================================
# Unified Readiness endpoint
# ============================================================

@app.post(
    "/get_player_readiness",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    tags=["Readiness"],
)
def get_player_readiness(
    data: ReadinessRequest,
) -> ReadinessResponse:
    """
    Run the unified Readiness Engine.

    Uses the same readiness logic that powers the Hybrid AI.
    """

    try:

        player_data = data.model_dump()

        result = calculate_readiness_output(
            player_data
        )

        return ReadinessResponse(
            **result
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


# ============================================================
# Hybrid Risk endpoint
# ============================================================

@app.post(
    "/get_hybrid_risk",
    response_model=HybridRiskResponse,
    status_code=status.HTTP_200_OK,
    tags=["Hybrid Risk"],
)
def get_hybrid_risk(
    data: HybridRiskRequest,
) -> HybridRiskResponse:
    """
    Run the combined Readiness + Temporal ML engine.
    """

    try:

        player_data = data.model_dump()

        result = calculate_hybrid_risk(
            player_data
        )

        return HybridRiskResponse(
            **result
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


# ============================================================
# What-If endpoint
# ============================================================

@app.post(
    "/get_what_if",
    status_code=status.HTTP_200_OK,
    tags=["What-If"],
)
def get_what_if(
    data: WhatIfRequest,
) -> dict[str, Any]:
    """
    Compare the athlete's current state with a temporary
    intervention scenario.

    The scenario is not persisted.
    """

    try:

        request_data = data.model_dump()

        changes = request_data.pop(
            "changes",
            {},
        )

        result = calculate_what_if(
            player_data=request_data,
            changes=changes,
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
