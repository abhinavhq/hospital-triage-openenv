import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from env.hospital_triage_env import HospitalTriageEnv
from tasks.tasks import get_task, EasyTriageTask, MediumTriageTask, HardTriageTask

def test_reset_returns_valid_state():
    env = HospitalTriageEnv(difficulty="easy", seed=0)
    obs = env.reset()
    assert "step" in obs
    assert "queue" in obs
    assert "rooms" in obs
    assert obs["step"] == 0
    assert obs["done"] is False

def test_step_returns_correct_types():
    env = HospitalTriageEnv(difficulty="easy", seed=0)
    env.reset()
    obs, reward, done, info = env.step(0)
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

def test_flat_obs_length():
    for diff in ["easy", "medium", "hard"]:
        env = HospitalTriageEnv(difficulty=diff, seed=0)
        env.reset()
        obs = env.state()
        assert len(obs["obs_flat"]) == 45

def test_episode_runs_to_completion():
    env = HospitalTriageEnv(difficulty="easy", seed=1)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(0)
        steps += 1
        assert steps < 1000
    assert done is True

def test_reward_bounded():
    env = HospitalTriageEnv(difficulty="hard", seed=3)
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(1)
    score = env._normalised_score()
    assert 0.0 <= score <= 1.0

def test_all_difficulties():
    for diff in ["easy", "medium", "hard"]:
        env = HospitalTriageEnv(difficulty=diff, seed=42)
        obs = env.reset()
        assert obs["difficulty"] == diff

def test_invalid_action_raises():
    env = HospitalTriageEnv(difficulty="easy", seed=0)
    env.reset()
    try:
        env.step(99)
        assert False
    except AssertionError:
        pass

def test_easy_task_grade():
    task = EasyTriageTask(seed=0)
    task.reset()
    done = False
    while not done:
        obs = task.state()
        action = 0 if obs["queue"] else 5
        _, _, done, _ = task.step(action)
    result = task.grade()
    assert "grade" in result
    assert 0.0 <= result["grade"] <= 1.0

def test_medium_task_grade():
    task = MediumTriageTask(seed=7)
    task.reset()
    done = False
    while not done:
        obs = task.state()
        action = 1 if obs["free_rooms"] > 0 else 5
        _, _, done, _ = task.step(action)
    result = task.grade()
    assert 0.0 <= result["grade"] <= 1.0

def test_hard_task_grade():
    task = HardTriageTask(seed=99)
    task.reset()
    done = False
    while not done:
        _, _, done, _ = task.step(0)
    result = task.grade()
    assert 0.0 <= result["grade"] <= 1.0

def test_get_task_factory():
    for diff in ["easy", "medium", "hard"]:
        t = get_task(diff, seed=0)
        obs = t.reset()
        assert obs["difficulty"] == diff

if __name__ == "__main__":
    tests = [
        test_reset_returns_valid_state,
        test_step_returns_correct_types,
        test_flat_obs_length,
        test_episode_runs_to_completion,
        test_reward_bounded,
        test_all_difficulties,
        test_invalid_action_raises,
        test_easy_task_grade,
        test_medium_task_grade,
        test_hard_task_grade,
        test_get_task_factory,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")