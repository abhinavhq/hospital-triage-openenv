import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from env.hospital_triage_env import HospitalTriageEnv

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "greedy")
HF_TOKEN = os.environ.get("HF_TOKEN")

_env = None

def reset(difficulty="easy", seed=42):
    global _env
    _env = HospitalTriageEnv(difficulty=difficulty, seed=seed)
    obs = _env.reset()
    return obs

def step(action: int):
    global _env
    if _env is None:
        reset()
    obs, reward, done, info = _env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }

def state():
    global _env
    if _env is None:
        reset()
    return _env.state()

if __name__ == "__main__":
    obs = reset(difficulty="easy", seed=42)
    print("Reset OK:", obs)
    result = step(0)
    print("Step OK:", result)
    s = state()
    print("State OK:", s)import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from env.hospital_triage_env import HospitalTriageEnv

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "greedy")
HF_TOKEN = os.environ.get("HF_TOKEN")

_env = None

def reset(difficulty="easy", seed=42):
    global _env
    _env = HospitalTriageEnv(difficulty=difficulty, seed=seed)
    obs = _env.reset()
    return obs

def step(action: int):
    global _env
    if _env is None:
        reset()
    obs, reward, done, info = _env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }

def state():
    global _env
    if _env is None:
        reset()
    return _env.state()

if __name__ == "__main__":
    obs = reset(difficulty="easy", seed=42)
    print("Reset OK:", obs)
    result = step(0)
    print("Step OK:", result)
    s = state()
    print("State OK:", s)
