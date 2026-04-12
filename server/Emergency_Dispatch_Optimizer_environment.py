"""
Emergency Dispatch Optimizer Environment.

An enterprise-grade Operations Research simulation. Models real-world 911 dispatch
mechanics including Service Level Agreements (SLAs), multi-resource batching, 
partial fulfillment, and temporal resource constraints.
"""

import os
import random
from uuid import uuid4
from typing import List, Dict, Any, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Bulletproof single-dot relative imports for Docker compatibility
from .models import (
    EmergencyDispatchOptimizerAction, 
    EmergencyDispatchOptimizerObservation,
    Incident,
    Unit,
    UnitType
)


class EmergencyDispatchOptimizerEnvironment(Environment):
    """
    Simulates a high-stakes dispatch center. The RL Agent must optimize resource
    allocation against strict time-to-live (SLA) constraints.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Core Simulation State
        self.incidents: List[Incident] = []
        self.units: List[Unit] = []
        self.task_name: str = "easy"
        self.max_steps: int = 20
        
        # Mathematical Grader Metrics
        self.total_required_deployments: int = 1
        self.successful_deployments: int = 0
        self.penalties_accrued: float = 0.0

    def _generate_incident(self, step: int, i_type: str, severity: int) -> Incident:
        """Procedurally factory for generating rigorous incident constraints."""
        idx = str(uuid4())[:6]
        
        # Dynamically determine resource requirements based on type and severity
        reqs: Dict[UnitType, int] = {}
        if i_type == "fire":
            reqs["fire_truck"] = min(severity, 3)
            reqs["ambulance"] = 1 if severity >= 3 else 0
            desc = f"Structure fire reported. Severity {severity}."
        elif i_type == "medical":
            reqs["ambulance"] = 2 if severity >= 4 else 1
            desc = f"Medical emergency. Severity {severity}."
        else:
            reqs["police_car"] = min(severity, 3)
            desc = f"Law enforcement required. Severity {severity}."
            
        # Higher severity = less time to respond
        sla_window = 12 - severity
            
        return Incident(
            id=f"inc_{idx}",
            description=desc,
            severity=severity,
            required_units=reqs,
            assigned_units={"fire_truck": 0, "ambulance": 0, "police_car": 0},
            step_arrived=step,
            expires_at_step=step + sla_window,
            is_resolved=False,
            is_failed=False
        )

    def reset(self) -> EmergencyDispatchOptimizerObservation:
        """Initializes the environment based on curriculum difficulty."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task_name = os.environ.get("DISPATCH_TASK", "easy").lower()
        
        # Reset Metrics
        self.successful_deployments = 0
        self.penalties_accrued = 0.0
        
        # Procedural Fleet Generation
        self.units = [
            Unit(id="amb_1", type="ambulance"), Unit(id="amb_2", type="ambulance"), Unit(id="amb_3", type="ambulance"),
            Unit(id="fire_1", type="fire_truck"), Unit(id="fire_2", type="fire_truck"), Unit(id="fire_3", type="fire_truck"),
            Unit(id="pol_1", type="police_car"), Unit(id="pol_2", type="police_car"), Unit(id="pol_3", type="police_car")
        ]
        
        self.incidents = []
        
        # Task Curriculum Generation
        if self.task_name == "easy":
            self.max_steps = 10
            self.incidents.append(self._generate_incident(0, "medical", 2))
            self.incidents.append(self._generate_incident(2, "police", 1))
            
        elif self.task_name == "medium":
            self.max_steps = 15
            self.incidents.append(self._generate_incident(0, "fire", 2)) 
            self.incidents.append(self._generate_incident(3, "medical", 3))
            self.incidents.append(self._generate_incident(5, "police", 2))
            
        else: # hard
            self.max_steps = 25
            types = ["fire", "medical", "police"]
            # Streaming incidents force the agent to plan resource returns
            for step in [0, 2, 4, 7, 10]:
                self.incidents.append(
                    self._generate_incident(step, random.choice(types), random.randint(3, 5))
                )

        # Pre-calculate perfect score denominator to guarantee [0.0 - 1.0] bounds
        total_reqs = sum(sum(inc.required_units.values()) for inc in self.incidents)
        self.total_required_deployments = max(1, total_reqs)

        return self._get_observation("Simulation initialized. Standing by for dispatch orders.", 0.0, False)

    def step(self, action: EmergencyDispatchOptimizerAction) -> EmergencyDispatchOptimizerObservation:  # type: ignore[override]
        """Core simulation loop. Executes in discrete deterministic phases."""
        
        # PHASE 1: Temporal Advance
        self._state.step_count += 1
        step_reward = 0.0
        feedback_log = []

        # PHASE 2: Fleet Telematics (Process returning units)
        for unit in self.units:
            if not unit.is_available and self._state.step_count >= unit.busy_until_step:
                unit.is_available = True
                feedback_log.append(f"[{unit.id} returned to station]")

        # PHASE 3: SLA Evaluation (Process Expirations)
        for inc in self.incidents:
            if not inc.is_resolved and not inc.is_failed and inc.step_arrived <= self._state.step_count:
                if self._state.step_count > inc.expires_at_step:
                    inc.is_failed = True
                    # Severe penalty for SLA breach proportional to incident severity
                    penalty = (inc.severity * 0.1)
                    self.penalties_accrued += penalty
                    step_reward -= penalty
                    feedback_log.append(f"[CRITICAL: {inc.id} SLA expired!]")

        # Evaluate currently visible incidents
        active_incidents = [
            i for i in self.incidents 
            if not i.is_resolved and not i.is_failed and i.step_arrived <= self._state.step_count
        ]

        # PHASE 4: Action Execution
        if action.action_type == "wait":
            feedback_log.append("Advanced simulation time.")
            
        elif action.action_type == "dispatch":
            step_reward, dispatch_msg = self._process_dispatch(action, active_incidents)
            if dispatch_msg:
                feedback_log.append(dispatch_msg)

        # PHASE 5: Termination & Grader Calculation
        all_finished = all((i.is_resolved or i.is_failed) for i in self.incidents)
        out_of_time = self._state.step_count >= self.max_steps
        done = all_finished or out_of_time

        if done:
            feedback_log.append("SIMULATION TERMINATED.")

        feedback_str = " | ".join(feedback_log) if feedback_log else "No events."
        return self._get_observation(feedback_str, step_reward, done)

    def _process_dispatch(self, action: EmergencyDispatchOptimizerAction, active_incidents: List[Incident]) -> Tuple[float, str]:
        """Handles the complex logic of multi-unit dispatching and partial fulfillment."""
        reward = 0.0
        
        if not action.incident_id or not action.unit_ids:
            self.penalties_accrued += 0.05
            return -0.05, "Error: Missing incident_id or unit_ids payload."
            
        incident = next((i for i in active_incidents if i.id == action.incident_id), None)
        if not incident:
            self.penalties_accrued += 0.05
            return -0.05, f"Error: Target {action.incident_id} is invalid or inactive."

        # Bulletproof: Deduplicate IDs to prevent LLM hallucination crashes
        safe_unit_ids = list(set(action.unit_ids))
        deployed_count = 0
        
        for uid in safe_unit_ids:
            unit = next((u for u in self.units if u.is_available and u.id == uid), None)
            if not unit:
                reward -= 0.02
                self.penalties_accrued += 0.02
                continue
                
            # Logic Gate: Is this unit type actually needed at the scene?
            needed = incident.required_units.get(unit.type, 0)
            assigned = incident.assigned_units.get(unit.type, 0)
            
            if assigned < needed:
                # Deploy the unit
                incident.assigned_units[unit.type] += 1
                unit.is_available = False
                unit.busy_until_step = self._state.step_count + 4  # 4 step operational turnaround
                
                # Dense Fractional Reward for correct allocation
                self.successful_deployments += 1
                reward_chunk = 1.0 / self.total_required_deployments
                reward += reward_chunk
                deployed_count += 1
            else:
                # Penalty for wasting resources
                reward -= 0.05
                self.penalties_accrued += 0.05

        # Check for full resolution
        resolved = all(incident.assigned_units.get(t, 0) >= c for t, c in incident.required_units.items())
        if resolved:
            incident.is_resolved = True
            return reward, f"Deployed {deployed_count} units. [{incident.id} SECURED!]"
            
        return reward, f"Deployed {deployed_count} units. {incident.id} requires additional resources."

    def _get_observation(self, feedback: str, reward: float, done: bool) -> EmergencyDispatchOptimizerObservation:
        """Constructs the heavily typed observation payload for the Agent."""
        
        active = [
            i for i in self.incidents 
            if not i.is_resolved and not i.is_failed and i.step_arrived <= self._state.step_count
        ]
        avail = [u for u in self.units if u.is_available]
        
        # Strict [0.0 - 1.0] mathematical bound for the final Hackathon Grader
        raw_score = (self.successful_deployments / self.total_required_deployments) - self.penalties_accrued
        normalized_score = max(0.01, min(0.99, raw_score))

        return EmergencyDispatchOptimizerObservation(
            active_incidents=active, 
            available_units=avail, 
            feedback=feedback.strip(),
            current_step=self._state.step_count, 
            task_name=self.task_name,
            reward=reward, 
            done=done, 
            metadata={"score": normalized_score}
        )

    @property
    def state(self) -> State:
        return self._state