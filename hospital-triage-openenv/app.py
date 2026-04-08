from flask import Flask, request, jsonify
from flask_cors import CORS
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from env.hospital_triage_env import HospitalTriageEnv

app = Flask(__name__)
CORS(app)

_env = HospitalTriageEnv(difficulty="easy", seed=42)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "env": "HospitalTriageEnv"})


@app.route("/reset", methods=["POST"])
def reset():
    global _env
    data = request.get_json(force=True) or {}
    difficulty = data.get("difficulty", "easy")
    seed = int(data.get("seed", 42))
    _env = HospitalTriageEnv(difficulty=difficulty, seed=seed)
    obs = _env.reset()
    return jsonify(obs)


@app.route("/step", methods=["POST"])
def step():
    data = request.get_json(force=True) or {}
    action = int(data.get("action", 0))
    obs, reward, done, info = _env.step(action)
    return jsonify({
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    })


@app.route("/state", methods=["GET"])
def state():
    return jsonify(_env.state())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)