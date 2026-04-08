# 🏥 Hospital Triage OpenEnv
**Meta × Scaler OpenEnv Hackathon — Round 1 Submission**

A complete OpenEnv environment simulating an **Emergency Department Triage System**.
An AI agent triages patients by assigning them to treatment rooms, calling specialists,
and responding to surge events through the standard `step()` / `reset()` / `state()` API.

---

## 🌍 Real-World Task
Emergency departments face life-or-death resource allocation every minute. This environment models:
- Patient prioritisation by severity (LOW → CRITICAL)
- Room assignment under capacity constraints
- Specialist routing for complex cases
- Dynamic events: patient surges and staff shortages

---

## ⚡ Quick Start
```python
from env.hospital_triage_env import HospitalTriageEnv
env = HospitalTriageEnv(difficulty="easy", seed=42)
obs = env.reset()
done = False
while not done:
    action = 0
    obs, reward, done, info = env.step(action)
    print(f"Step {obs['step']} | Reward: {reward:.2f} | Queue: {obs['queue_length']}")
print(f"Final score: {env._normalised_score():.3f}")
```

---

## 🎮 Action Space
| Action | Description | Available |
|--------|-------------|-----------|
| 0 | Assign patient to Room 1 | All |
| 1 | Assign patient to Room 2 | All |
| 2 | Assign patient to Room 3 | medium/hard |
| 3 | Assign with PRIORITY flag | medium/hard |
| 4 | Assign to Room 5 | hard only |
| 5 | Send patient back to waiting | All |
| 6 | Discharge treated patient | All |
| 7 | Call specialist | hard only |

---

## 👁️ Observation Space
```python
obs = env.state()
obs["obs_flat"]         # 45-float vector for RL agents
obs["queue"]            # waiting patients with severity, urgency, condition
obs["rooms"]            # room occupancy and status
obs["surge_active"]     # bool — surge event active
obs["stats"]            # recovered / died / deteriorated counts
obs["normalised_score"] # float in [0, 1]
```

---

## 🏆 Reward Function
| Event | Reward |
|-------|--------|
| CRITICAL treated on time | +1.00 |
| HIGH treated on time | +0.70 |
| MEDIUM treated on time | +0.40 |
| LOW treated on time | +0.20 |
| Treated late | x0.50 |
| Correct specialist | x1.20 |
| Patient deteriorates | -0.50 |
| Patient dies | -1.00 |
| Wrong action | -0.05 to -0.30 |

---

## 📊 Three Tasks
### Task 1 — Quiet Shift (easy)
2 rooms, 3 patients max, 50 steps. No surge. Pass = 0 deaths.

### Task 2 — Evening Rush (medium)
3 rooms, 6 patients, 100 steps. Specialists present. Pass = 80% recovery.

### Task 3 — Mass Casualty Incident (hard)
5 rooms, 10 patients, 200 steps. Surges every 40 steps. Staff shortages every 60 steps.

---

## 📈 Baseline Scores (Greedy Agent, 5 Seeds)
```bash
python baseline_inference.py --agent greedy --seeds 5
```

| Task | Avg Score | Avg Grade | Avg Deaths | Pass Rate |
|------|-----------|-----------|------------|-----------|
| EASY | 0.810 | 1.000 | 0.0 | 100% |
| MEDIUM | 0.419 | 0.300 | 3.0 | 60% |
| HARD | 0.287 | 0.281 | 3.4 | 40% |

> Scores are fully reproducible across seeds.

---

## 🖼️ Demo

### ✅ 11/11 Tests Passing
![Tests Passing](asets/hospital_opencv.png)

### 📊 Baseline Inference Output
![Baseline Scores](asets/hospital_openenv.png)

---

## 🧪 Tests
```bash
python tests/test_environment.py
# 11/11 tests passed
```

**Test coverage:**
- `test_reset_returns_valid_state`
- `test_step_returns_correct_types`
- `test_flat_obs_length`
- `test_episode_runs_to_completion`
- `test_reward_bounded`
- `test_all_difficulties`
- `test_invalid_action_raises`
- `test_easy_task_grade`
- `test_medium_task_grade`
- `test_hard_task_grade`
- `test_get_task_factory`

---

## 🚀 Deployment
```bash
docker build -t hospital-triage-openenv .
docker run -p 7860:7860 hospital-triage-openenv
```

---

## ✅ OpenEnv Spec Compliance
| Requirement | Status |
|-------------|--------|
| Real-world task | ✅ Emergency Department |
| step/reset/state API | ✅ |
| openenv.yaml | ✅ |
| 3 tasks with graders | ✅ |
| Reward in 0.0–1.0 | ✅ |
| Partial progress signals | ✅ |
| Baseline inference script | ✅ |
| Reproducible scores | ✅ |
| HuggingFace + Dockerfile | ✅ |
| README | ✅ |