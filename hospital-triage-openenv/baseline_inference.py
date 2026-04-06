import argparse, sys, os, random, time
sys.path.insert(0, os.path.dirname(__file__))
from tasks.tasks import get_task

class RandomAgent:
    name = "random"
    def __init__(self, difficulty):
        self._nactions = {"easy": 5, "medium": 7, "hard": 8}[difficulty]
        self._rng = random.Random(0)
    def act(self, obs):
        return self._rng.randint(0, self._nactions - 1)

class GreedyTriageAgent:
    name = "greedy"
    def act(self, obs):
        queue = obs.get("queue", [])
        rooms = obs.get("rooms", [])
        free_rooms = [r for r in rooms if not r["occupied"]]
        difficulty = obs.get("difficulty", "easy")
        if not queue:
            return 5
        top = queue[0]
        if top.get("is_treated"):
            return 6
        if not free_rooms:
            return 5
        room_idx = free_rooms[0]["room_id"]
        if difficulty == "hard" and top.get("needs_specialist"):
            return 7
        if difficulty in ("medium", "hard") and top["severity"] >= 3:
            return 3
        action_map = {0: 0, 1: 1, 2: 2, 3: 4}
        return action_map.get(room_idx, 0)

class PriorityQueueAgent:
    name = "priority_queue"
    def act(self, obs):
        queue = obs.get("queue", [])
        rooms = obs.get("rooms", [])
        free_rooms = [r for r in rooms if not r["occupied"]]
        difficulty = obs.get("difficulty", "easy")
        if not queue:
            return 5
        urgent = sorted(queue, key=lambda p: -p["urgency_pct"])
        top = urgent[0]
        if top.get("is_treated"):
            return 6
        if not free_rooms:
            return 5
        room_idx = free_rooms[0]["room_id"]
        if difficulty == "hard" and top.get("needs_specialist"):
            return 7
        if difficulty in ("medium", "hard") and top["severity"] >= 3:
            return 3
        action_map = {0: 0, 1: 1, 2: 2, 3: 4}
        return action_map.get(room_idx, 0)

AGENTS = {
    "random": RandomAgent,
    "greedy": GreedyTriageAgent,
    "priority_queue": PriorityQueueAgent,
}

def run_episode(agent, difficulty, seed):
    task = get_task(difficulty, seed=seed)
    obs = task.reset()
    done = False
    total_reward = 0.0
    steps = 0
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = task.step(action)
        total_reward += reward
        steps += 1
    result = task.grade() if hasattr(task, "grade") else {}
    result.update({
        "seed": seed,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "norm_score": task.score(),
    })
    return result

def run_all(agent_name="greedy", num_seeds=5):
    AgentClass = AGENTS[agent_name]
    print(f"\n{'='*60}")
    print(f"  Hospital Triage OpenEnv — Baseline Inference")
    print(f"  Agent: {agent_name.upper()}   |   Seeds: 0-{num_seeds-1}")
    print(f"{'='*60}\n")
    for difficulty in ["easy", "medium", "hard"]:
        agent = AgentClass(difficulty) if agent_name == "random" else AgentClass()
        scores, grades, deaths_list = [], [], []
        t0 = time.time()
        for seed in range(num_seeds):
            r = run_episode(agent, difficulty, seed)
            scores.append(r["norm_score"])
            grades.append(r.get("grade", r["norm_score"]))
            deaths_list.append(r.get("deaths", 0))
        elapsed = time.time() - t0
        avg_score = sum(scores) / len(scores)
        avg_grade = sum(grades) / len(grades)
        avg_deaths = sum(deaths_list) / len(deaths_list)
        pass_rate = sum(1 for g in grades if g >= 0.4) / len(grades)
        print(f"  [{difficulty.upper():6s}]  "
              f"avg_score={avg_score:.3f}  "
              f"avg_grade={avg_grade:.3f}  "
              f"avg_deaths={avg_deaths:.1f}  "
              f"pass_rate={pass_rate*100:.0f}%  "
              f"({elapsed:.1f}s)")
    print(f"\n{'='*60}")
    print("  Baseline inference complete. Scores are reproducible.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="greedy", choices=list(AGENTS.keys()))
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    run_all(agent_name=args.agent, num_seeds=args.seeds)