from __future__ import annotations
import random
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import IntEnum

class Severity(IntEnum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4

MAX_WAIT = {1: 120, 2: 30, 3: 10, 4: 3}
BASE_REWARD = {1: 0.20, 2: 0.40, 3: 0.70, 4: 1.00}
TREAT_DURATION = {1: 3, 2: 5, 3: 8, 4: 12}
SEVERITY_LABEL = {1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}
CONDITION_CODES = [
    "chest_pain", "fracture", "head_trauma", "respiratory_distress",
    "abdominal_pain", "allergic_reaction", "stroke", "sepsis",
    "minor_laceration", "cardiac_arrest"
]

@dataclass
class Patient:
    pid: int
    severity: int
    arrival_step: int
    age: int
    condition: str
    needs_specialist: bool
    wait_time: int = 0
    room_id: Optional[int] = None
    is_treated: bool = False
    outcome: Optional[str] = None

    def obs_vector(self) -> List[float]:
        return [
            self.pid / 200.0,
            self.severity / 4.0,
            min(self.wait_time / 120.0, 1.0),
            self.age / 100.0,
            CONDITION_CODES.index(self.condition) / 9.0,
            float(self.needs_specialist),
            float(self.room_id is not None),
            float(self.is_treated),
        ]

@dataclass
class Room:
    rid: int
    occupied: bool = False
    patient_id: Optional[int] = None
    steps_remaining: int = 0
    has_specialist: bool = False

class HospitalTriageEnv:
    CONFIGS = {
        "easy":   dict(num_rooms=2, max_queue=3,  max_steps=50,  spawn_prob=0.30, specialist=False, surges=False),
        "medium": dict(num_rooms=3, max_queue=6,  max_steps=100, spawn_prob=0.50, specialist=True,  surges=False),
        "hard":   dict(num_rooms=5, max_queue=10, max_steps=200, spawn_prob=0.70, specialist=True,  surges=True),
    }

    def __init__(self, difficulty="easy", seed=None):
        assert difficulty in self.CONFIGS
        self.difficulty = difficulty
        self.cfg = self.CONFIGS[difficulty]
        self.seed = seed
        self.reset()

    def reset(self):
        self._rng = random.Random(self.seed)
        self.rooms = [Room(rid=i) for i in range(self.cfg["num_rooms"])]
        self.patients = {}
        self.queue = []
        self.step_n = 0
        self._pid_counter = 0
        self._done = False
        self._surge = False
        self._shortage = False
        self._cumulative_reward = 0.0
        self._max_possible_reward = 0.0
        self.stats = dict(recovered=0, deteriorated=0, died=0, discharged=0, wasted_actions=0)
        for _ in range(max(1, self.cfg["max_queue"] // 2)):
            self._spawn_patient()
        return self.state()

    def step(self, action):
        assert not self._done, "Episode finished. Call reset()."
        assert 0 <= action <= 7, f"Action must be 0-7, got {action}"
        reward = 0.0
        info = {"step": self.step_n, "events": [], "action": action}
        if self.queue:
            reward += self._apply_action(action, info)
        reward += self._tick_rooms(info)
        reward += self._tick_patients(info)
        if len(self.queue) < self.cfg["max_queue"]:
            if self._rng.random() < self.cfg["spawn_prob"]:
                self._spawn_patient()
        if self.cfg["surges"]:
            self._handle_events(info)
        self.step_n += 1
        self._cumulative_reward += reward
        self._done = (self.step_n >= self.cfg["max_steps"] or self._all_dead_or_treated())
        obs = self.state()
        info["cumulative_reward"] = round(self._cumulative_reward, 4)
        info["normalised_score"] = self._normalised_score()
        return obs, round(reward, 4), self._done, info

    def state(self):
        waiting = []
        for pid in self.queue[:self.cfg["max_queue"]]:
            p = self.patients[pid]
            waiting.append({
                "patient_id": p.pid, "severity": p.severity,
                "severity_label": SEVERITY_LABEL[p.severity],
                "condition": p.condition, "age": p.age,
                "wait_time": p.wait_time, "max_wait": MAX_WAIT[p.severity],
                "urgency_pct": round(min(p.wait_time / MAX_WAIT[p.severity], 1.0), 3),
                "needs_specialist": p.needs_specialist,
                "is_treated": p.is_treated, "obs_vector": p.obs_vector(),
            })
        rooms_obs = [{"room_id": r.rid, "occupied": r.occupied,
                      "patient_id": r.patient_id, "steps_remaining": r.steps_remaining,
                      "has_specialist": r.has_specialist} for r in self.rooms]
        return {
            "step": self.step_n, "max_steps": self.cfg["max_steps"],
            "difficulty": self.difficulty, "queue": waiting,
            "queue_length": len(self.queue), "rooms": rooms_obs,
            "free_rooms": sum(1 for r in self.rooms if not r.occupied),
            "surge_active": self._surge, "staff_shortage": self._shortage,
            "stats": dict(self.stats),
            "normalised_score": self._normalised_score(),
            "obs_flat": self._flat_obs(), "done": self._done,
        }

    def _apply_action(self, action, info):
        reward = 0.0
        pid = self.queue[0]
        p = self.patients[pid]
        room_map = {0: 0, 1: 1, 2: 2, 3: 2, 4: 4}
        if action in room_map:
            ridx = room_map[action]
            if ridx >= len(self.rooms):
                reward -= 0.05; self.stats["wasted_actions"] += 1; return reward
            room = self.rooms[ridx]
            if room.occupied:
                reward -= 0.05; self.stats["wasted_actions"] += 1
                info["events"].append(f"Room {ridx+1} occupied"); return reward
            self.queue.pop(0)
            p.room_id = ridx
            room.occupied = True; room.patient_id = pid
            dur = TREAT_DURATION[p.severity]
            if self._shortage: dur = int(dur * 1.5)
            room.steps_remaining = dur
            if action == 3 and p.severity >= Severity.HIGH:
                reward += 0.05
                info["events"].append(f"Priority triage Patient {pid}")
            self._max_possible_reward += BASE_REWARD[p.severity]
            info["events"].append(f"Patient {pid} [{SEVERITY_LABEL[p.severity]}] -> Room {ridx+1}")
        elif action == 5:
            if p.severity > Severity.LOW:
                reward -= 0.1 * p.severity; self.stats["wasted_actions"] += 1
        elif action == 6:
            if p.is_treated:
                self.queue.pop(0); p.outcome = "recovered"
                self.stats["discharged"] += 1; reward += 0.10
            else:
                reward -= 0.30; self.stats["wasted_actions"] += 1
        elif action == 7:
            if not self.cfg["specialist"]:
                reward -= 0.05; return reward
            if not p.needs_specialist:
                reward -= 0.05; return reward
            free = next((r for r in self.rooms if not r.occupied), None)
            if free:
                self.queue.pop(0); p.room_id = free.rid
                free.occupied = True; free.patient_id = pid
                free.has_specialist = True; free.steps_remaining = 15
                reward += 0.10
                self._max_possible_reward += BASE_REWARD[p.severity] * 1.2
        return reward

    def _tick_rooms(self, info):
        reward = 0.0
        for room in self.rooms:
            if not room.occupied: continue
            room.steps_remaining -= 1
            if room.steps_remaining <= 0:
                pid = room.patient_id; p = self.patients[pid]
                p.is_treated = True; p.outcome = "recovered"
                self.stats["recovered"] += 1
                r = BASE_REWARD[p.severity]
                if p.wait_time > MAX_WAIT[p.severity]: r *= 0.5
                if room.has_specialist and p.needs_specialist: r *= 1.2
                r = min(r, 1.0); reward += r
                room.occupied = False; room.patient_id = None; room.has_specialist = False
                self.queue.append(pid)
                info["events"].append(f"Treatment done Patient {pid} +{r:.2f}")
        return reward

    def _tick_patients(self, info):
        reward = 0.0
        for pid in list(self.queue):
            p = self.patients[pid]
            if p.is_treated or p.room_id is not None: continue
            p.wait_time += 1
            limit = MAX_WAIT[p.severity]
            if p.wait_time == int(limit * 1.5) and p.outcome is None:
                p.outcome = "deteriorated"; p.severity = min(p.severity + 1, 4)
                self.stats["deteriorated"] += 1; reward -= 0.50
            if p.wait_time >= limit * 3 and p.outcome != "died":
                p.outcome = "died"; self.queue.remove(pid)
                self.stats["died"] += 1; reward -= 1.00
        return reward

    def _handle_events(self, info):
        if self.step_n % 40 == 0 and self.step_n > 0:
            self._surge = True
            for _ in range(3):
                if len(self.queue) < self.cfg["max_queue"]:
                    self._spawn_patient(force_severity=Severity.CRITICAL)
            info["events"].append("SURGE - 3 critical patients!")
        if self.step_n % 60 == 0 and self.step_n > 0:
            self._shortage = not self._shortage
        if self.step_n % 45 != 0: self._surge = False

    def _spawn_patient(self, force_severity=None):
        sev = force_severity or self._rng.choices([1,2,3,4], weights=[0.35,0.30,0.20,0.15])[0]
        self._pid_counter += 1
        p = Patient(pid=self._pid_counter, severity=sev, arrival_step=self.step_n,
                    age=self._rng.randint(1,100), condition=self._rng.choice(CONDITION_CODES),
                    needs_specialist=(self.cfg["specialist"] and self._rng.random() < 0.25))
        self.patients[p.pid] = p
        inserted = False
        for i, qpid in enumerate(self.queue):
            if self.patients[qpid].severity < p.severity:
                self.queue.insert(i, p.pid); inserted = True; break
        if not inserted: self.queue.append(p.pid)

    def _all_dead_or_treated(self):
        if not self.patients: return False
        return all(p.outcome in ("recovered","died","deteriorated") or p.is_treated
                   for p in self.patients.values())

    def _normalised_score(self):
        if self._max_possible_reward == 0: return 0.0
        return round(max(0.0, min(1.0, self._cumulative_reward / max(self._max_possible_reward,1))), 4)

    def _flat_obs(self):
        obs = [self.step_n/self.cfg["max_steps"],
               sum(1 for r in self.rooms if not r.occupied)/self.cfg["num_rooms"],
               len(self.queue)/self.cfg["max_queue"],
               float(self._surge), float(self._shortage)]
        for i in range(5):
            if i < len(self.queue):
                obs.extend(self.patients[self.queue[i]].obs_vector())
            else:
                obs.extend([0.0]*8)
        return obs