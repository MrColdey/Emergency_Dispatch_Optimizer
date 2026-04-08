"""
Inference Script for the Emergency Dispatch Optimizer Environment.
================================================================
MANDATORY HACKATHON COMPLIANCE:
- Named `inference.py` in the root directory.
- Uses OpenAI Client with HF_TOKEN and API_BASE_URL.
- Strictly emits [START], [STEP], and [END] logs.
- Tests all 3 difficulty tasks.
"""

import os
import json
import re
import textwrap
from typing import Optional
from openai import OpenAI

try:
    from Emergency_Dispatch_Optimizer.models import EmergencyDispatchOptimizerAction
    from server.Emergency_Dispatch_Optimizer_environment import EmergencyDispatchOptimizerEnvironment
except ImportError:
    from Emergency_Dispatch_Optimizer.models import EmergencyDispatchOptimizerAction
    from server.Emergency_Dispatch_Optimizer_environment import EmergencyDispatchOptimizerEnvironment

# --- Hackathon Required Environment Variables ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("API_KEY", os.getenv("HF_TOKEN"))
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "emergency_dispatch_optimizer"
MAX_STEPS = 8

# --- System Prompt ---
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert 911 emergency dispatcher AI. 
    You will receive observations containing active incidents and available units.
    
    Rules:
    - Match 'medical' incidents with 'ambulance' units.
    - Match 'fire' incidents with 'fire_truck' units.
    - Match 'police' incidents with 'police_car' units.
    - Prioritize high-severity incidents.
    - If no suitable unit is available, or you want to wait for future incidents, choose "wait".
    
    You must respond with ONLY valid JSON matching this schema:
    {
        "action_type": "dispatch" or "wait",
        "incident_id": "string id of the incident" (or null if waiting),
        "unit_id": "string id of the unit" (or null if waiting)
    }
    Do not include markdown blocks, explanations, or any text other than the JSON object.
    """
).strip()

# --- Strict Logging Helpers (Do not modify formatting) ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    # Action string sanitized to remove newlines for clean single-line logs
    action_clean = action.replace('\n', '').replace('\r', '')
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# --- Utility to parse LLM JSON ---
def parse_llm_action(response_text: str) -> EmergencyDispatchOptimizerAction:
    """Safely extracts JSON from LLM output and converts to Pydantic Action."""
    try:
        # Strip potential markdown code blocks (e.g., ```json ... ```)
        cleaned = re.sub(r'```(?:json)?\n?(.*?)\n?```', r'\1', response_text, flags=re.DOTALL).strip()
        parsed = json.loads(cleaned)
        return EmergencyDispatchOptimizerAction(**parsed)
    except Exception as e:
        # Fallback to 'wait' if the LLM hallucinates non-JSON
        return EmergencyDispatchOptimizerAction(action_type="wait", incident_id=None, unit_id=None)

# --- Main Inference Loop ---
def run_inference(task_name: str, client: OpenAI) -> None:
    """Runs a single episode for a specific task difficulty."""
    
    # Instruct the environment which task to load during reset
    os.environ["DISPATCH_TASK"] = task_name
    
    # Initialize Environment
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
        
        # 1. Format the LLM Prompt with the current observation
        prompt = f"Current Observation:\n{obs.model_dump_json(indent=2)}"
        
        try:
            # 2. Call the LLM using OpenAI client via HF Router
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1, # Low temp for structured JSON output
                max_tokens=150
            )
            
            action_json_str = response.choices[0].message.content
            action = parse_llm_action(action_json_str)
            
        except Exception as e:
            error_msg = f"LLM Call or Parsing Failed: {str(e)}"
            action = EmergencyDispatchOptimizerAction(action_type="wait")
            action_json_str = '{"action_type": "wait", "error": "fallback"}'

        # 3. Step the Environment
        try:
            obs = env.step(action)
            # The OpenEnv step() returns an Observation object. 
            # We fetch reward and done from its base attributes.
            reward = float(obs.reward)
            done = bool(obs.done)
        except Exception as e:
            error_msg = f"Env Step Failed: {str(e)}"
            reward = 0.0
            done = True
            
        total_score += reward
        rewards.append(reward)
        
        # 4. Log the step exactly as requested
        log_step(
            step=step_count,
            action=action_json_str,
            reward=reward,
            done=done,
            error=error_msg
        )

    # 5. Determine success threshold
    # Since total_severity maps accurately to 1.0, resolving all correctly yields ~1.0
    # Allow a small floating-point margin (0.95)
    success = total_score >= 0.95 
    
    # Ensure score bounds [0.0, 1.0] just in case of penalties
    final_score = max(0.01, min(0.99, total_score))
    
    log_end(success=success, steps=step_count, score=final_score, rewards=rewards)


if __name__ == "__main__":
    if not API_KEY:
        print("WARNING: HF_TOKEN environment variable is missing. LLM calls may fail.")
        
    client = OpenAI(
        api_key=API_KEY or "empty", 
        base_url=API_BASE_URL
    )
    
    # The Hackathon requirement: Test at least 3 tasks (Easy, Medium, Hard)
    tasks = ["easy", "medium", "hard"]
    
    for task in tasks:
        print(f"\n--- Running Task: {task.upper()} ---")
        run_inference(task, client)