"""
Data models for the Emergency Dispatch Optimizer Environment.

Defines the strictly typed Action and Observation spaces for the 911 Dispatch task.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation

# --- Sub-models for Environment State ---

class Incident(BaseModel):
    id: str = Field(..., description="Unique identifier for the incident")
    type: Literal["fire", "medical", "police"] = Field(..., description="Type of emergency")
    severity: int = Field(..., description="Severity scale from 1 (low) to 5 (high)")
    is_resolved: bool = Field(default=False, description="Whether the incident has been handled")
    step_arrived: int = Field(default=0, description="The step number when this incident appears")

class Unit(BaseModel):
    id: str = Field(..., description="Unique identifier for the dispatch unit")
    type: Literal["fire_truck", "ambulance", "police_car"] = Field(..., description="Type of vehicle")
    is_available: bool = Field(default=True, description="Whether the unit is ready to be dispatched")


# --- Main OpenEnv Models ---

class EmergencyDispatchOptimizerAction(Action):
    """Action for the Emergency Dispatch Optimizer environment."""

    action_type: Literal["dispatch", "wait"] = Field(
        ..., 
        description="Type of action to take. 'wait' holds units for future incidents."
    )
    incident_id: Optional[str] = Field(
        default=None, 
        description="ID of the incident to handle (required if action_type is 'dispatch')"
    )
    unit_id: Optional[str] = Field(
        default=None, 
        description="ID of the unit to send (required if action_type is 'dispatch')"
    )


class EmergencyDispatchOptimizerObservation(Observation):
    """Observation from the Emergency Dispatch Optimizer environment."""

    active_incidents: List[Incident] = Field(
        default_factory=list, 
        description="List of currently visible, unresolved incidents"
    )
    available_units: List[Unit] = Field(
        default_factory=list, 
        description="List of units currently available at the station"
    )
    feedback: str = Field(
        default="", 
        description="Feedback message from the previous action (e.g. success or error)"
    )
    current_step: int = Field(
        default=0, 
        description="Current step number in the episode"
    )
    task_name: str = Field(
        default="easy", 
        description="The current difficulty level being evaluated"
    )