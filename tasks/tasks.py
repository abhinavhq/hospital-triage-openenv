from __future__ import annotations
from typing import Any, Dict, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from env.hospital_triage_env import HospitalTriageEnv

class BaseTriageTask:
    difficulty: str = "easy"
    task_name: str = "base"
    description: str = ""

    def __init__(self, seed: int = 42):
        self.env = HospitalTriageEnv(difficulty=self.difficulty, seed=seed)
        self.seed = seed

    def reset(self):
        return self.env.reset()

    def step(self, action: int):
        return self.env.step(action)

    def state(self):
        return self.env.state()

    def score(self):
        return self.env._normalised_score()


class EasyTriageTask(BaseTriageTask):
    difficulty = "easy"
    task_name = "quiet_shift"
    description = "Manage a quiet ED shift. 2 rooms, small patient load."

    def grade(self):
        stats = self.env.stats
        deaths = stats["died"]
        deteriorated = stats["deteriorated"]
        norm_score = self.env._normalised_score()
        if deaths == 0 and deteriorated == 0:
            grade = 1.0
        elif deaths == 0 and deteriorated <= 1:
            grade = 0.7
        elif deaths <= 1:
            grade = 0.4
        else:
            grade = max(0.0, norm_score - 0.2)
        return {
            "task": self.task_name, "difficulty": self.difficulty,
            "grade": round(grade, 4), "norm_score": norm_score,
            "deaths": deaths, "deteriorated": deteriorated,
            "recovered": stats["recovered"], "passed": grade >= 0.4,
        }


class MediumTriageTask(BaseTriageTask):
    difficulty = "medium"
    task_name = "evening_rush"
    description = "Handle an evening rush with specialist patients."

    def grade(self):
        stats = self.env.stats
        total = max(1, sum(stats[k] for k in ("recovered","died","deteriorated","discharged")))
        recovery_rate = stats["recovered"] / total
        deaths = stats["died"]
        norm_score = self.env._normalised_score()
        if recovery_rate >= 0.80 and deaths <= 1:
            grade = 1.0
        elif recovery_rate >= 0.60 and deaths <= 2:
            grade = 0.7
        elif recovery_rate >= 0.40 and deaths <= 4:
            grade = 0.4
        else:
            grade = max(0.0, norm_score - 0.3)
        return {
            "task": self.task_name, "difficulty": self.difficulty,
            "grade": round(grade, 4), "norm_score": norm_score,
            "recovery_rate": round(recovery_rate, 3), "deaths": deaths,
            "deteriorated": stats["deteriorated"], "recovered": stats["recovered"],
            "passed": grade >= 0.4,
        }


class HardTriageTask(BaseTriageTask):
    difficulty = "hard"
    task_name = "mass_casualty"
    description = "Survive a mass casualty event. Surges, staff shortages."

    def grade(self):
        stats = self.env.stats
        deaths = stats["died"]
        norm_score = self.env._normalised_score()
        if norm_score >= 0.75 and deaths <= 2:
            grade = 1.0
        elif norm_score >= 0.55 and deaths <= 5:
            grade = 0.7
        elif norm_score >= 0.35:
            grade = 0.4
        else:
            grade = max(0.0, norm_score)
        return {
            "task": self.task_name, "difficulty": self.difficulty,
            "grade": round(grade, 4), "norm_score": norm_score,
            "deaths": deaths, "deteriorated": stats["deteriorated"],
            "recovered": stats["recovered"], "wasted": stats["wasted_actions"],
            "passed": grade >= 0.4,
        }


TASKS = {
    "easy": EasyTriageTask,
    "medium": MediumTriageTask,
    "hard": HardTriageTask,
}

def get_task(difficulty: str, seed: int = 42):
    assert difficulty in TASKS, f"Unknown difficulty: {difficulty}"
    return TASKS[difficulty](seed=seed)