"""
Emergency Dispatch Optimizer Environment Implementation.

An RL environment where an agent acts as a 911 dispatcher.
The agent must pair appropriate response units with incoming incidents, 
maximizing efficiency and handling dynamic arrivals.
"""

import os
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    # This works when run from the root folder (e.g., Docker & Hugging Face)
    from Emergency_Dispatch_Optimizer.models import (
        EmergencyDispatchOptimizerAction, 
        EmergencyDispatchOptimizerObservation,
        Incident,
        Unit
    )
except ImportError:
    # This works when run directly from inside the server/ folder
    from Emergency_Dispatch_Optimizer.models import (
        EmergencyDispatchOptimizerAction, 
        EmergencyDispatchOptimizerObservation,
        Incident,
        Unit
    )


class EmergencyDispatchOptimizerEnvironment(Environment):
    """
    Emergency Dispatch Environment.
    
    Tasks:
    - Easy: 1 Incident, 1 Unit (Basic matching)
    - Medium: Multiple Incidents, exact units (Prioritization)
    - Hard: Incidents arriving over time (Requires 'wait' action planning)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the Emergency_Dispatch_Optimizer environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.incidents = []
        self.units = []
        self.total_severity = 0.0
        self.max_steps = 10
        self.task_name = "easy"

    def reset(self) -> EmergencyDispatchOptimizerObservation:
        """
        Reset the environment and load the requested task.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Read task from environment variable (default to easy)
        # This allows inference.py to easily loop through tasks
        self.task_name = os.environ.get("DISPATCH_TASK", "easy").lower()

        # --- Task 1: Easy (Basic mapping) ---
        if self.task_name == "easy":
            self.incidents = [Incident(id="inc_1", type="medical", severity=3, step_arrived=0)]
            self.units = [Unit(id="u_1", type="ambulance")]
            self.max_steps = 3

        # --- Task 2: Medium (Multiple mapping & Prioritization) ---
        elif self.task_name == "medium":
            self.incidents = [
                Incident(id="inc_1", type="fire", severity=5, step_arrived=0),
                Incident(id="inc_2", type="medical", severity=2, step_arrived=0)
            ]
            self.units = [
                Unit(id="u_1", type="fire_truck"),
                Unit(id="u_2", type="ambulance"),
                Unit(id="u_3", type="police_car") # Decoy unit
            ]
            self.max_steps = 5

# --- Task 3: Hard (Long-Running dynamic arrivals & Resource Management) ---
        else: 
            self.incidents = [
                Incident(id="inc_1", type="police", severity=2, step_arrived=0),
                Incident(id="inc_2", type="medical", severity=3, step_arrived=2),
                Incident(id="inc_3", type="fire", severity=5, step_arrived=4),
                Incident(id="inc_4", type="police", severity=4, step_arrived=6),
                Incident(id="inc_5", type="medical", severity=5, step_arrived=8)
            ]
            self.units = [
                Unit(id="u_1", type="police_car"),
                Unit(id="u_2", type="ambulance"),
                Unit(id="u_3", type="fire_truck")
            ]
            # Extended steps to allow units time to handle calls and return
            self.max_steps = 15

        # Pre-calculate total severity for 0.0 to 1.0 Reward Normalization
        self.total_severity = max(1.0, sum(i.severity for i in self.incidents))

        return self._get_observation(
            feedback=f"Environment reset to {self.task_name} task. Ready for dispatch.",
            reward=0.0,
            done=False
        )

    def step(self, action: EmergencyDispatchOptimizerAction) -> EmergencyDispatchOptimizerObservation:  # type: ignore[override]
        """Execute a step in the environment."""
        self._state.step_count += 1
        reward = 0.0
        feedback = ""

        for unit in self.units:
            if not unit.is_available and self._state.step_count >= unit.busy_until_step:
                unit.is_available = True
                feedback += f"[{unit.id} returned to station] "


        # Only show incidents that have "arrived" by the current step
        active_incidents = [
            i for i in self.incidents 
            if not i.is_resolved and i.step_arrived <= self._state.step_count
        ]

        # 1. Logic for "wait" action
        if action.action_type == "wait":
            feedback = "Agent chose to wait for future incidents."
            
        # 2. Logic for "dispatch" action
        elif action.action_type == "dispatch":
            if not action.incident_id or not action.unit_id:
                feedback = "Error: Dispatch action requires both incident_id and unit_id."
                reward = -0.05 # Penalty for bad formatting
            else:
                incident = next((i for i in active_incidents if i.id == action.incident_id), None)
                unit = next((u for u in self.units if u.is_available and u.id == action.unit_id), None)

                if not incident:
                    feedback = f"Error: Invalid or inactive incident_id: '{action.incident_id}'."
                    reward = -0.05
                elif not unit:
                    feedback = f"Error: Invalid or unavailable unit_id: '{action.unit_id}'."
                    reward = -0.05
                else:
                    # Validate Unit-to-Incident match
                    match_map = {
                        "fire": "fire_truck",
                        "medical": "ambulance",
                        "police": "police_car"
                    }
                    
                    if match_map[incident.type] == unit.type:
                        incident.is_resolved = True
                        unit.is_available = False
                        # Unit takes 3 steps to complete the job and return
                        unit.busy_until_step = self._state.step_count + 3 
                        
                        reward = incident.severity / self.total_severity
                        feedback += f"Success! Dispatched {unit.id} to {incident.id}. It will return at step {unit.busy_until_step}."
                    else:
                        feedback = f"Failure: {incident.type} needs {match_map[incident.type]}, got {unit.type}."
                        reward = -0.1 # Strong penalty for sending wrong unit

        # 3. Check episode termination conditions
        all_resolved = all(i.is_resolved for i in self.incidents)
        out_of_steps = self._state.step_count >= self.max_steps
        done = all_resolved or out_of_steps

        if done:
            if all_resolved:
                feedback += " Episode Complete: All incidents successfully resolved!"
            else:
                feedback += " Episode Ended: Maximum steps reached before resolving all incidents."

        return self._get_observation(feedback=feedback, reward=reward, done=done)

    def _get_observation(self, feedback: str, reward: float, done: bool) -> EmergencyDispatchOptimizerObservation:
        """Helper to construct the observation state cleanly."""
        active_incidents = [
            i for i in self.incidents 
            if not i.is_resolved and i.step_arrived <= self._state.step_count
        ]
        available_units = [u for u in self.units if u.is_available]

        return EmergencyDispatchOptimizerObservation(
            active_incidents=active_incidents,
            available_units=available_units,
            feedback=feedback,
            current_step=self._state.step_count,
            task_name=self.task_name,
            reward=reward,
            done=done,
            metadata={"step": self._state.step_count}
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state