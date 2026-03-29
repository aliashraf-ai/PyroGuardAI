import uvicorn
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import numpy as np
import cv2
import json
import random
import sys
import os

# --- CONFIGURATION ---
GRID_W, GRID_H = 100, 80

# --- 1. STRICT MODULE LOADER ---
# This section forces the use of YOUR files. It does not use mocks.
SYSTEM = {}

print("\n" + "=" * 50)
print("🔥 PYROGUARD AI SERVER: BOOTING...")
print("=" * 50)

# LOAD SIMULATION
try:
    import fire_simulation

    SYSTEM["SIM"] = fire_simulation.FireSim(GRID_W, GRID_H)
    print("✅ [PHYSICS] FireSim loaded.")
except ImportError:
    print("❌ [PHYSICS] 'fire_simulation.py' NOT FOUND. Falling back to internal physics.")
    SYSTEM["SIM"] = None

# LOAD PSO (SWARM)
try:
    import drone_swarm_pso

    SYSTEM["PSO"] = drone_swarm_pso
    print("✅ [SWARM] PSO Optimizer loaded.")
except ImportError:
    print("❌ [SWARM] 'drone_swarm_pso.py' NOT FOUND. Drones will be manual.")
    SYSTEM["PSO"] = None

# LOAD DNN (RISK)
try:
    import fire_predictor_dnn

    SYSTEM["DNN"] = fire_predictor_dnn
    print("✅ [RISK] DNN Predictor loaded.")
except ImportError:
    print("❌ [RISK] 'fire_predictor_dnn.py' NOT FOUND.")
    SYSTEM["DNN"] = None

# LOAD RL (RESOURCE)
try:
    import resource_allocator_rl
    import pickle

    if os.path.exists('models/rl_agent.pkl'):
        with open('models/rl_agent.pkl', 'rb') as f:
            SYSTEM["RL"] = pickle.load(f)
        print("✅ [RL] Agent Loaded (Trained Model).")
    else:
        SYSTEM["RL"] = resource_allocator_rl.QLearningAgent()
        print("✅ [RL] Agent Loaded (New Instance).")
except Exception as e:
    print(f"❌ [RL] Failed to load RL Agent: {e}")
    SYSTEM["RL"] = None

# LOAD CNN (VISION)
try:
    import satellite_analyzer_cnn

    # Attempt to initialize the class directly
    if hasattr(satellite_analyzer_cnn, 'FireDetectorCNN'):
        SYSTEM["CNN"] = satellite_analyzer_cnn.FireDetectorCNN()
        print("✅ [VISION] CNN Model loaded.")
    else:
        print("❌ [VISION] 'FireDetectorCNN' class not found in satellite_analyzer_cnn.py")
        SYSTEM["CNN"] = None
except ImportError:
    print("❌ [VISION] 'satellite_analyzer_cnn.py' NOT FOUND.")
    SYSTEM["CNN"] = None


# --- 2. GAME STATE ---
class GameState:
    def __init__(self):
        # Fire Grid
        if SYSTEM["SIM"]:
            self.grid = SYSTEM["SIM"].grid
        else:
            self.grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)

        # Drone State
        self.drones = []
        for i in range(8):
            self.drones.append({
                "id": i,
                "x": random.randint(10, 90), "y": random.randint(10, 70),
                "tx": random.randint(10, 90), "ty": random.randint(10, 70),
                "action": "HOVER"
            })

        self.rl_decision = "WAITING..."

    def update_physics(self):
        # 1. Fire Physics
        if SYSTEM["SIM"]:
            SYSTEM["SIM"].update()
        else:
            # Internal Circular Expansion Logic
            if np.sum(self.grid) > 0:
                # Diffuse heat outwards
                blurred = cv2.GaussianBlur(self.grid, (5, 5), 0)
                mask = self.grid > 2
                # Grow edges
                self.grid[mask] = np.maximum(self.grid[mask], blurred[mask] * 1.05)
                self.grid = np.clip(self.grid, 0, 100)

        # 2. Drone Interaction
        for d in self.drones:
            # Simple check: Is drone over fire?
            gx, gy = int(d['x']), int(d['y'])
            if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                if self.grid[gy, gx] > 10:
                    d['action'] = "EXTINGUISH"
                    # Extinguish: Reduce fire intensity in radius
                    r = 3
                    self.grid[max(0, gy - r):min(GRID_H, gy + r), max(0, gx - r):min(GRID_W, gx + r)] *= 0.8
                else:
                    d['action'] = "HOVER"


state = GameState()

# --- 3. SERVER API ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Run Physics
        state.update_physics()

        # Run RL (Periodically)
        if SYSTEM["RL"] and random.random() < 0.05:
            # Construct state vector for RL
            s = (int(np.sum(state.grid)), 5, 5, 8)
            action_idx = SYSTEM["RL"].choose_action(s, training=False)
            names = getattr(SYSTEM["RL"], 'action_names', ["DEPLOY", "HOVER", "EVACUATE", "SUPPRESS", "IDLE"])
            state.rl_decision = names[action_idx] if action_idx < len(names) else "UNKNOWN"

            if "Drones" in state.rl_decision or "DEPLOY" in state.rl_decision:
                # Trigger PSO automatically
                trigger_pso_logic()

        # Send Data to Frontend
        # Optimize: Only send non-zero fire pixels
        fire_data = []
        ys, xs = np.where(state.grid > 5)
        for y, x in zip(ys, xs):
            fire_data.append([int(x), int(y), int(state.grid[y, x])])

        await websocket.send_json({
            "drones": state.drones,
            "fire": fire_data,
            "rl": state.rl_decision
        })
        await asyncio.sleep(0.05)  # 20 Hz Update Rate


# --- ENDPOINTS ---

@app.post("/api/predict_risk")
async def predict_risk(data: dict):
    """Bridge to your DNN"""
    if not SYSTEM["DNN"]: return {"error": "DNN Module Missing"}
    try:
        # Calls your actual function
        risk, probs = SYSTEM["DNN"].predict_fire_risk(data)
        labels = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        return {"level": labels[risk], "risk_id": int(risk)}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """Bridge to your CNN"""
    if not SYSTEM["CNN"]: return {"label": "CNN MISSING", "conf": 0}

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize to standard CNN input
        img_resized = cv2.resize(img, (224, 224))

        # Call YOUR model
        if hasattr(SYSTEM["CNN"], 'predict'):
            # Handle different return types from user models
            res = SYSTEM["CNN"].predict(img_resized)
            if isinstance(res, tuple):
                label, conf = res
            else:
                label = res; conf = 1.0

            # Ensure JSON serializable
            if isinstance(conf, (np.float32, np.float64)):
                conf = float(conf)
            elif isinstance(conf, (list, np.ndarray)):
                conf = float(conf[0])

            return {"label": str(label), "conf": conf}
        else:
            return {"label": "Model has no predict()", "conf": 0}

    except Exception as e:
        return {"label": f"Error: {str(e)}", "conf": 0}


@app.post("/api/pso")
async def trigger_pso():
    trigger_pso_logic()
    return {"status": "ok"}


@app.post("/api/ignite")
async def ignite(coords: dict):
    gx, gy = coords['x'], coords['y']
    # Create fire blob
    state.grid[max(0, gy - 3):min(GRID_H, gy + 4), max(0, gx - 3):min(GRID_W, gx + 4)] = 100
    return {"status": "ignited"}


def trigger_pso_logic():
    """Runs your PSO code to update drone targets"""
    ys, xs = np.where(state.grid > 20)
    if len(xs) == 0: return  # No fire

    targets = []
    if SYSTEM["PSO"]:
        try:
            fires = [{'x': float(c), 'y': float(r), 'intensity': 1.0, 'id': i} for i, (r, c) in enumerate(zip(ys, xs))]
            swarm = SYSTEM["PSO"].WildfireDroneSwarm(n_drones=8, map_size=(GRID_W, GRID_H))
            swarm.fire_locations = fires
            _, best_pos = swarm.optimize(n_iterations=15)

            for i in range(8):
                targets.append((best_pos[i * 2], best_pos[i * 2 + 1]))
        except Exception as e:
            print(f"PSO Error: {e}")
            # Fallback if PSO crashes
            for i in range(8): targets.append((xs[i % len(xs)], ys[i % len(ys)]))
    else:
        # Fallback if no PSO module
        for i in range(8): targets.append((xs[i % len(xs)], ys[i % len(ys)]))

    # Update Drone Targets
    for i, d in enumerate(state.drones):
        if i < len(targets):
            d['tx'] = targets[i][0]
            d['ty'] = targets[i][1]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)