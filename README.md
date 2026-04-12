# 🚨 Emergency Dispatch Optimizer (OpenEnv)

A high-fidelity Operations Research & Logistics environment built on the **OpenEnv** framework. 

This environment simulates a high-stakes 911 emergency dispatch center. An AI/RL agent must triage procedurally generated emergencies, allocate specific combinations of emergency response vehicles, and optimize fleet logistics against strict **Service Level Agreement (SLA)** time-to-live constraints.

---

## 📖 Motivation & Real-World Utility

Unlike static puzzle games, this environment models dynamic, real-world temporal resource management—a critical challenge in logistics, ride-sharing, and emergency services. 

* **Multi-Resource Batching:** Complex emergencies (e.g., Structure Fires) require multiple unit types simultaneously (Fire Trucks + Ambulances).
* **Temporal Planning:** Dispatched units become "busy" for a set number of time steps. The agent must strategically use the `wait` action to advance time and allow the fleet to return to the station.
* **Strict SLAs:** Incidents have a strict `expires_at_step` limit. Ignoring high-severity incidents results in critical SLA breaches and heavy score penalties.

---

## 📂 Project Structure

```text
emergency-dispatch-optimizer/
├── Dockerfile                # Root-level Dockerfile (Enables Web UI)
├── inference.py              # Strict LLM evaluation script (Tests all 3 tasks)
├── openenv.yaml              # OpenEnv specification config
├── .dockerignore             # Protects env/venv from corrupting Linux builds
└── server/
    ├── app.py                # FastAPI & OpenEnv Router wrapper
    ├── environment.py        # Core RL procedural generation & SLA simulation
    └── models.py             # Heavily constrained Pydantic schemas
```

---

## 🔬 Environment Spaces

The environment utilizes heavily constrained Pydantic models (using `Literal`, `StrictInt`, and mathematical bounds) to prevent LLM hallucinations from corrupting the simulation state.

### Action Space
The agent outputs a single JSON object per step:

| Field | Type | Description |
| :--- | :--- | :--- |
| **`action_type`** | `Literal["dispatch", "wait"]` | Core action. `dispatch` assigns units; `wait` advances the simulation clock. |
| **`incident_id`** | `Optional[str]` | The ID of the targeted emergency. |
| **`unit_ids`** | `Optional[List[str]]` | Array of unit IDs to deploy simultaneously (e.g., `["amb_1", "fire_2"]`). |

### Observation Space
The agent receives a rich state payload detailing the operational board:

| Field | Type | Description |
| :--- | :--- | :--- |
| **`active_incidents`** | `List[Incident]` | Unresolved emergencies with `severity`, `required_units`, and `expires_at_step`. |
| **`available_units`** | `List[Unit]` | Idle fleet vehicles ready for immediate deployment. |
| **`feedback`** | `str` | Surgical telematics feedback from the previous action. |
| **`current_step`** | `int` | The current operational timeline step. |
| **`task_name`** | `str` | Current curriculum difficulty level. |

---

## 📈 Tasks & Difficulty Curriculum

The environment features procedural generation driven by the `DISPATCH_TASK` environment variable, enabling seamless curriculum learning.

1. **🟢 Easy (`task=easy`)**
   * **Objective:** Basic 1:1 resource mapping. 
   * **Mechanics:** Low severity incidents, simple fleet constraints. Introduces the agent to the observation schema.

2. **🟡 Medium (`task=medium`)**
   * **Objective:** Multi-resource batching and prioritization.
   * **Mechanics:** Introduces moderate severity incidents requiring combinations of units (e.g., Fire + Medical). The agent must prioritize targets before they expire.

3. **🔴 Hard (`task=hard`)**
   * **Objective:** Long-running temporal resource management.
   * **Mechanics:** Incidents stream in over 25 steps. The fleet is insufficient to handle all incidents simultaneously. The agent **must** execute the `wait` action to advance time, allow units to complete jobs, and return to the station before SLA limits are breached.

---

## 🏆 Reward Function & Grader

The environment utilizes a **Dense Fractional Reward** system that mathematically clamps the final episode score strictly between `0.0` and `1.0`.

* **Partial Fulfillment (+):** Assigning a correct unit to an incident yields a fractional reward: `1.0 / total_required_deployments`.
* **Resource Waste (-):** Dispatching unnecessary units or hallucinated IDs yields immediate step penalties (`-0.05`).
* **SLA Breach (-):** If an incident's `expires_at_step` is surpassed, a heavy penalty proportional to the incident's severity is applied.
* **Final Grader:** The environment's metadata returns the normalized cumulative score bounded to `[0.0, 1.0]`.

---

## 📊 Baseline Scores

Evaluated using `Qwen/Qwen2.5-72B-Instruct` via the Hugging Face Router at `Temperature = 0.0`. 
*The model successfully learned to batch-dispatch resources and utilize the `wait` command to allow fleet returns.*

* **Easy Task:** `0.99` (Success)
* **Medium Task:** `0.99` (Success)
* **Hard Task:** `0.85+` (Success - Successfully managed complex SLA streams and fleet returns).

---

## 🚀 Setup & Usage Instructions

### 1. Local Testing
Install `uv` and OpenEnv core:
```bash
pip install openenv-core uv
```
Create a `.env` file in the root directory:
```env
HF_TOKEN=hf_your_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
```
Run the baseline inference script:
```bash
uv run --env-file .env inference.py
```

### 2. Docker & Web UI (Visual Interactive Mode)
The environment includes a built-in Gradio Web UI for manual testing.
```bash
# Build the Docker image
docker build -t emergency_env .

# Run the container
docker run -p 8000:8000 emergency_env
```
Open your browser and navigate to **[http://localhost:8000/web](http://localhost:8000/web)** to play the environment manually.

### 3. Submission Validation
To ensure full OpenEnv spec compliance:
```bash
curl -fsSL https://raw.githubusercontent.com/huggingface/openenv/main/scripts/validate-submission.sh | bash -s -- http://localhost:8000 .
```
