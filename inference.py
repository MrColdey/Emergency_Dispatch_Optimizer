"""
Inference Script for the Emergency Dispatch Optimizer.
======================================================
MANDATORY HACKATHON COMPLIANCE:
- Named `inference.py` in the root directory.
- Uses OpenAI Client with HF_TOKEN and API_BASE_URL.
- Strictly emits [START], [STEP], and [END] logs.
- Evaluates 3 distinct difficulty tasks.
"""

import os
import json
import re
import textwrap
from typing import Optional
from openai import OpenAI

# Bulletproof imports to handle both direct runs and Docker/Uvicorn runs
try:
    from server.models import EmergencyDispatchOptimizerAction
    from server.Emergency_Dispatch_Optimizer_environment import EmergencyDispatchOptimizerEnvironment
except ImportError:
    from models import EmergencyDispatchOptimizerAction
    from server.Emergency_Dispatch_Optimizer_environment import EmergencyDispatchOptimizerEnvironment

# --- Hackathon Required Environment Variables ---
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "emergency_dispatch_optimizer"

# Max steps scaled up to 30 to comfortably allow the "hard" long-running scenario (25 steps)
MAX_STEPS = 30

# --- Advanced System Prompt ---
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert 911 Operations Research AI.
    Your objective is to optimize emergency dispatch logistics against strict SLA constraints.
    
    RULES:
    1. Review 'active_incidents' and their 'required_units' (e.g., {"ambulance": 2, "fire_truck": 1}).
    2. Review 'assigned_units' to see what is already at the scene.
    3. You must batch-dispatch ALL currently needed units to an incident in a single action using the "unit_ids" array.
    4. Prioritize high-severity incidents and incidents close to their 'expires_at_step' limit.
    5. Dispatched units take 4 steps to complete a job and return.
    6. If you do not have the required units available, or you are waiting for units to return, output "wait".
    
    Output ONLY valid JSON matching this exact schema. No markdown, no explanations.
    {
        "action_type": "dispatch" or "wait",
        "incident_id": "inc_abc123" (or null if waiting),
        "unit_ids": ["amb_1", "amb_2", "fire_1"] (or null if waiting)
    }
    """
).strip()

# --- Strict Logging Helpers (Do not modify formatting) ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    # Action string strictly sanitized for single-line compliance
    action_clean = action.replace('\n', '').replace('\r', '').strip()
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# --- Utility to parse LLM JSON ---
def parse_llm_action(response_text: str) -> EmergencyDispatchOptimizerAction:
    """Safely extracts JSON from LLM output and converts to the strict Pydantic Action."""
    try:
        # Strip potential markdown code blocks (e.g., ```json ... ```)
        cleaned = re.sub(r'```(?:json)?\n?(.*?)\n?```', r'\1', response_text, flags=re.DOTALL).strip()
        parsed = json.loads(cleaned)
        return EmergencyDispatchOptimizerAction(**parsed)
    except Exception:
        # Fallback to 'wait' if the LLM hallucinates non-JSON, preventing environment crashes
        return EmergencyDispatchOptimizerAction(action_type="wait", incident_id=None, unit_ids=None)

# --- Main Inference Loop ---
def run_inference(task_name: str, client: OpenAI) -> None:
    """Runs a single curriculum episode."""
    
    # Instruct the environment which task to load
    os.environ["DISPATCH_TASK"] = task_name
    
    env = EmergencyDispatchOptimizerEnvironment()
    obs = env.reset()
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    rewards = []
    done = False
    step_count = 0
    total_score = 0.0
    
    while not done and step_count < MAX_STEPS:
        step_count += 1
        error_msg = None
        action_json_str = ""
        
        # Format the Observation payload
        prompt = f"Current Observation:\n{obs.model_dump_json(indent=2)}"
        
        try:
            # Call the LLM using OpenAI client via HF Router
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0, # Temperature 0.0 forces highest determinism for structured logic
                max_tokens=150
            )
            
            action_json_str = response.choices[0].message.content
            action = parse_llm_action(action_json_str)
            
        except Exception as e:
            error_msg = f"LLM Call Failed: {str(e)}"
            action = EmergencyDispatchOptimizerAction(action_type="wait")
            action_json_str = '{"action_type": "wait", "error": "fallback"}'

        # Step the Environment
        try:
            obs = env.step(action)
            reward = float(obs.reward)
            done = bool(obs.done)
        except Exception as e:
            error_msg = f"Env Step Failed: {str(e)}"
            reward = 0.0
            done = True
            
        total_score += reward
        rewards.append(reward)
        
        # Log the step exactly as requested
        log_step(
            step=step_count,
            action=action_json_str,
            reward=reward,
            done=done,
            error=error_msg
        )

    # Retrieve the mathematically rigorous normalized score calculated by the environment
    # Fallback to mathematically clamping the cumulative reward if metadata is missing
    final_score = float(obs.metadata.get("score", max(0.01, min(0.99, total_score))))
    
    # Define a reasonable success threshold for the final output log
    # Complex RL environments consider 70%+ an operational success
    success = final_score >= 0.7 
    
    log_end(success=success, steps=step_count, score=final_score, rewards=rewards)


if __name__ == "__main__":
    if not API_KEY:
        print("WARNING: HF_TOKEN environment variable is missing. LLM calls may fail.")
        
    client = OpenAI(
        api_key=API_KEY or "empty", 
        base_url=API_BASE_URL
    )
    
    # Hackathon requirement: Test 3 distinct tasks (Easy, Medium, Hard)
    tasks = ["easy", "medium", "hard"]
    
    for task in tasks:
        print(f"\n--- Running Task: {task.upper()} ---")
        run_inference(task, client)