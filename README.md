# Emergency Dispatch Optimizer

## 1. Environment Description and Motivation
The Emergency Dispatch Optimizer is a real-world simulation of a 911 dispatch center. The motivation is to train RL agents to efficiently allocate limited emergency response units (ambulances, fire trucks, police cars) to incoming incidents based on varying severities and types. This models a genuine, high-stakes municipal logistics and routing problem, replacing toy games with a highly practical utility task.

## 2. Action and Observation Spaces
* **Observation Space**: The agent observes `active_incidents` (type, severity, arrival step) and `available_units` (type, availability status), along with the current step and environment feedback.
* **Action Space**: The agent can output a `dispatch` command (requires `incident_id` and `unit_id`) to accurately assign a unit to an emergency, or a `wait` command to strategically hold units in reserve for future, potentially higher-severity incidents.

## 3. Task Descriptions
* **Easy**: 1 Incident, 1 Unit. Tests basic type-matching logic (e.g., medical -> ambulance).
* **Medium**: Multiple incidents and mixed units including decoys. Tests prioritization of high-severity incidents over low-severity ones.
* **Hard**: Dynamic time arrivals. Incidents arrive over varying steps, forcing the agent to strategically use the 'wait' action instead of immediately depleting resources on low-severity calls.

## 4. Setup and Usage Instructions
**Local Setup:**
1. Install dependencies: `pip install openenv-core openai pydantic`
2. Run locally via the OpenEnv CLI: `openenv run`
3. Access the web interface at `http://localhost:8000/web`

**Docker Setup:**
1. Build the image: `docker build -t emergency_dispatch .`
2. Run the container: `docker run -p 8000:8000 emergency_dispatch`

## 5. Baseline Scores
Tested using `Qwen/Qwen2.5-72B-Instruct` via the Hugging Face Router:
* **Easy Score**: 1.00
* **Medium Score**: 1.00
* **Hard Score**: 1.00