"""
Data models for the Emergency Dispatch Optimizer.

Strictly typed schema defining the Agent's interface for a real-world 
911 Operations Research environment. Features rigorous Service Level Agreement (SLA) 
tracking, multi-resource batching, and strict bounds validation.
"""

from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field, StrictStr, StrictInt
from openenv.core.env_server.types import Action, Observation

# Define globally permitted unit types to prevent hallucinated vehicles
UnitType = Literal["fire_truck", "ambulance", "police_car"]
ActionType = Literal["dispatch", "wait"]


class Incident(BaseModel):
    """Represents an active emergency with strict time-to-live constraints."""
    
    id: StrictStr = Field(..., description="Unique identifier for the incident.")
    description: StrictStr = Field(..., description="Context of the emergency.")
    
    # Severity is mathematically locked between 1 and 5
    severity: StrictInt = Field(..., ge=1, le=5, description="Severity scale from 1 (low) to 5 (high).")
    
    required_units: Dict[UnitType, StrictInt] = Field(
        ..., 
        description="Map of unit types to exact quantities needed (e.g., {'ambulance': 2, 'fire_truck': 1})."
    )
    assigned_units: Dict[UnitType, StrictInt] = Field(
        default_factory=dict, 
        description="Units currently responding to or on-scene at this incident."
    )
    
    # Time and SLA Tracking
    step_arrived: StrictInt = Field(..., ge=0, description="The timeline step when this incident was reported.")
    expires_at_step: StrictInt = Field(..., ge=1, description="Absolute step timeline limit. If unresolved by this step, the SLA is breached.")
    
    # Resolution States
    is_resolved: bool = Field(default=False, description="True if all required resources have been successfully allocated.")
    is_failed: bool = Field(default=False, description="True if the incident expired before required resources arrived.")


class Unit(BaseModel):
    """Represents a dispatchable emergency response vehicle."""
    
    id: StrictStr = Field(..., description="Unique identifier for the dispatch unit.")
    type: UnitType = Field(..., description="Classification of the emergency vehicle.")
    is_available: bool = Field(default=True, description="True if the unit is idle at the station and ready for deployment.")
    busy_until_step: StrictInt = Field(default=0, ge=0, description="The exact timeline step when this unit will finish its current job and return.")


class EmergencyDispatchOptimizerAction(Action):
    """
    Action space for the Agent.
    Strictly accepts either a batch dispatch command or a wait command to advance time.
    """
    
    action_type: ActionType = Field(
        ..., 
        description="Strategic action to take. 'dispatch' allocates resources. 'wait' advances the simulation clock."
    )
    incident_id: Optional[StrictStr] = Field(
        default=None, 
        description="Target emergency ID. Required if action_type is 'dispatch'."
    )
    unit_ids: Optional[List[StrictStr]] = Field(
        default=None, 
        description="List of target unit IDs to deploy simultaneously. Required if action_type is 'dispatch'."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"action_type": "dispatch", "incident_id": "inc_123456", "unit_ids": ["amb_1", "fire_2"]},
                {"action_type": "wait", "incident_id": None, "unit_ids": None}
            ]
        }
    }


class EmergencyDispatchOptimizerObservation(Observation):
    """
    Observation space detailing the current state of the dispatch operations board.
    Inherits standard OpenEnv tracking (reward, done, metadata).
    """
    
    active_incidents: List[Incident] = Field(
        default_factory=list, 
        description="List of unresolved and unexpired emergencies requiring attention."
    )
    available_units: List[Unit] = Field(
        default_factory=list, 
        description="List of idle units currently available for immediate dispatch."
    )
    feedback: StrictStr = Field(
        default="", 
        description="System telematics feedback from the previous action (e.g., success confirmations or validation errors)."
    )
    current_step: StrictInt = Field(
        default=0, 
        ge=0,
        description="The current operational time step of the simulation."
    )
    task_name: StrictStr = Field(
        default="medium", 
        description="The current active difficulty curriculum."
    )