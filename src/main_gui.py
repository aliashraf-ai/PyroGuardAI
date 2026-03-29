import math
import os
import random
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import torch
from PIL import Image, ImageTk

# --- 1. STRICT BACKEND IMPORTS ---
try:
    from satellite_analyzer_cnn import FireDetectorCNN
    from resource_allocator_rl import QLearningAgent
    from drone_swarm_pso import WildfireDroneSwarm
    from fire_predictor_dnn import FireRiskDNN, predict_fire_risk

    print("✅ AI Backend Modules Loaded Successfully.")
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")

# --- 2. VISUAL THEME SETTINGS ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

COLOR_BG = "#0b0f19"  # Deep Navy Background
COLOR_PANEL = "#151b2b"  # Panel Background
COLOR_ACCENT = "#3b82f6"  # Bright Blue
COLOR_FIRE = "#ef4444"  # Fire Red
COLOR_SUCCESS = "#10b981"  # Success Green
COLOR_WARN = "#f59e0b"  # Warning Orange
COLOR_TEXT_DIM = "#94a3b8"  # Dim Text

# Global Simulation Variables
GLOBAL_RISK_FACTOR = 0.05
MAX_FIRE_OBJECTS = 150  # [IMPROVEMENT] Prevents System Crash


# ==================== 3. ADVANCED SIMULATION CLASSES ====================

class Particle:
    """Visual effects for Fire Embers and Water Spray"""

    def __init__(self, x, y, type="EMBER"):
        self.x = x
        self.y = y
        self.type = type
        self.life = 1.0  # 1.0 to 0.0

        if type == "EMBER":
            self.vx = random.uniform(-0.5, 0.5)
            self.vy = random.uniform(-2.0, -0.5)  # Float upwards
            self.color = random.choice(["#fca5a5", "#fbbf24", "#ef4444"])
            self.decay = 0.03
        else:  # WATER
            self.vx = random.uniform(-2.0, 2.0)
            self.vy = random.uniform(-2.0, 2.0)
            self.color = "#60a5fa"
            self.decay = 0.1

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= self.decay


class SimFire:
    """Dynamic Fire Entity that grows based on environmental risk"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.intensity = 0.1
        self.health = 20.0
        self.max_health = 100.0
        self.active = True
        self.id = random.randint(1000, 9999)
        self.particles = []
        self.pulse_phase = random.uniform(0, 6.28)
        self.display_radius = 15

    def update(self):
        # 1. Dynamic Growth Logic (Controlled by DNN Risk)
        if self.active and self.health < self.max_health:
            # Growth scales with global risk
            growth = 0.05 * (1 + GLOBAL_RISK_FACTOR * 4)
            self.health += growth
            self.intensity = self.health / 100.0

        # 2. Visual Pulsing Effect
        self.pulse_phase += 0.15
        self.display_radius = (12 + (self.intensity * 35)) + math.sin(self.pulse_phase) * 2

        # 3. Particle System (Embers)
        if random.random() < (0.4 * self.intensity):
            self.particles.append(Particle(self.x, self.y, "EMBER"))

        # Update particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles: p.update()


class SimDrone:
    """Autonomous Quadcopter with Swarm Physics"""

    def __init__(self, uid, base_x, base_y, role="STANDBY"):
        self.id = uid
        self.base_x = base_x
        self.base_y = base_y
        self.x = base_x
        self.y = base_y
        self.role = role  # PATROL or STANDBY

        # Physics Vectors
        self.vx = 0
        self.vy = 0
        self.angle = 0

        # State Machine
        self.state = "PATROLLING" if role == "PATROL" else "IDLE"
        self.target_fire = None
        self.target_x = None
        self.target_y = None

        # Stats
        self.max_speed = 4.5 if role == "PATROL" else 7.5
        self.water = 100.0
        self.trail = []
        self.water_particles = []

        # Visuals (Rotor animation)
        self.rotor_phase = 0
        self.patrol_angle = random.uniform(0, 6.28)
        self.patrol_radius = 120
        self.scan_angle = 0

    def update(self, all_drones, all_fires):
        self.rotor_phase += 1.0  # Spin rotors
        target_x, target_y = self.base_x, self.base_y  # Default target

        # --- 1. INTELLIGENT TARGETING ---
        if self.target_fire and not self.target_fire.active:
            self.target_fire = None
            # Auto-assign to nearest active fire
            active_fires = [f for f in all_fires if f.active]
            if active_fires:
                self.target_fire = min(active_fires, key=lambda f: math.hypot(f.x - self.x, f.y - self.y))
                self.state = "DEPLOYED"
            else:
                self.state = "RETURNING"

        # --- 2. STATE MACHINE ---
        if self.state == "PATROLLING":
            self.patrol_angle += 0.02
            target_x = self.base_x + math.cos(self.patrol_angle) * self.patrol_radius
            target_y = self.base_y + math.sin(self.patrol_angle) * self.patrol_radius
            self.scan_angle += 0.25

        elif self.state == "IDLE":
            target_x, target_y = self.base_x, self.base_y

        elif self.state == "DEPLOYED":
            if self.target_fire and self.target_fire.active:
                target_x, target_y = self.target_fire.x, self.target_fire.y
                if math.hypot(self.x - target_x, self.y - target_y) < 60:
                    self.state = "EXTINGUISHING"
            else:
                self.state = "RETURNING"

        elif self.state == "EXTINGUISHING":
            if self.target_fire and self.target_fire.active:
                # Orbit Logic
                orbit_time = time.time() * 3 + self.id
                target_x = self.target_fire.x + math.cos(orbit_time) * 45
                target_y = self.target_fire.y + math.sin(orbit_time) * 45

                # Extinguish Action
                self.water -= 0.3
                self.target_fire.health -= 0.8

                # Water Particles
                for _ in range(2):
                    p = Particle(self.x, self.y, "WATER")
                    # Shoot towards fire center
                    aim_angle = math.atan2(self.target_fire.y - self.y, self.target_fire.x - self.x)
                    p.vx = math.cos(aim_angle) * 8 + random.uniform(-1, 1)
                    p.vy = math.sin(aim_angle) * 8 + random.uniform(-1, 1)
                    self.water_particles.append(p)

                if self.target_fire.health <= 0:
                    self.target_fire.active = False
                    self.state = "RETURNING"
                if self.water <= 0:
                    self.state = "RETURNING"
            else:
                self.state = "RETURNING"

        elif self.state == "RETURNING":
            target_x, target_y = self.base_x, self.base_y
            if math.hypot(self.x - target_x, self.y - target_y) < 20:
                self.water = 100
                self.state = "PATROLLING" if self.role == "PATROL" else "IDLE"
                self.vx, self.vy = 0, 0  # Stop

        # --- 3. PHYSICS & SEPARATION ---
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.hypot(dx, dy)

        # Separation (Boids)
        sep_x, sep_y = 0, 0
        separation_radius = 40

        for other in all_drones:
            if other != self:
                d = math.hypot(self.x - other.x, self.y - other.y)
                if d < separation_radius:
                    push_force = (separation_radius - d) / separation_radius
                    sep_x += (self.x - other.x) * push_force * 3.0
                    sep_y += (self.y - other.y) * push_force * 3.0

        if dist > 0:
            desired_vx = (dx / dist) * self.max_speed
            desired_vy = (dy / dist) * self.max_speed

            desired_vx += sep_x
            desired_vy += sep_y

            steer_factor = 0.15
            self.vx += (desired_vx - self.vx) * steer_factor
            self.vy += (desired_vy - self.vy) * steer_factor

        self.x += self.vx
        self.y += self.vy

        # Update Particles & Trail
        self.water_particles = [p for p in self.water_particles if p.life > 0]
        for p in self.water_particles: p.update()

        self.trail.append((self.x, self.y))
        if len(self.trail) > 12: self.trail.pop(0)


# ==================== 4. GUI APPLICATION CLASS ====================

class PyroGuardApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🔥 PYROGUARD AI - Integrated Command Center")
        self.geometry("1400x900")
        self.configure(fg_color=COLOR_BG)

        # Application State
        self.fires = []
        self.drones = []
        self.extinguished_count = 0
        self.webcam_active = False
        self.cap = None

        # AI Models
        self.cnn = None
        self.dnn = None
        self.pso = None
        self.rl = None

        # Start Loading AI in Background
        threading.Thread(target=self.load_ai_modules, daemon=True).start()

        # Build UI
        self.setup_ui()
        self.init_drone_fleet()

        # Start Animation
        self.animate_loop()

    def load_ai_modules(self):
        """Initializes all AI components"""
        try:
            # CNN
            self.cnn = FireDetectorCNN()

            # DNN
            self.dnn = FireRiskDNN()
            if os.path.exists('models/fire_risk_dnn.pth'):
                self.dnn.load_state_dict(torch.load('models/fire_risk_dnn.pth'))
                self.dnn.eval()  # Set to eval mode

            # PSO
            self.pso = WildfireDroneSwarm(n_drones=8, map_size=(800, 600))

            # RL
            if os.path.exists('models/rl_agent.pkl'):
                import pickle
                with open('models/rl_agent.pkl', 'rb') as f:
                    self.rl = pickle.load(f)
            else:
                self.rl = QLearningAgent()

            self.lbl_status.configure(text="SYSTEM ONLINE", fg_color=COLOR_SUCCESS)
            self.update_dnn_risk()  # Initial risk calculation
        except Exception as e:
            print(f"AI Load Error: {e}")

    def init_drone_fleet(self):
        """Spawns 8 drones across 2 bases"""
        # Base 1 (Top Left)
        self.drones.append(SimDrone(0, 100, 100, role="PATROL"))
        for i in range(1, 4):
            self.drones.append(SimDrone(i, 100, 100, role="STANDBY"))

        # Base 2 (Bottom Right)
        bx, by = 700, 500
        self.drones.append(SimDrone(4, bx, by, role="PATROL"))
        for i in range(5, 8):
            self.drones.append(SimDrone(i, bx, by, role="STANDBY"))

    # ==================== UI SETUP ====================

    def setup_ui(self):
        # 3-Column Grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- HEADER ---
        header = ctk.CTkFrame(self, height=60, fg_color=COLOR_PANEL)
        header.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=(10, 0))

        ctk.CTkLabel(header, text="🔥 PYROGUARD AI", font=("Orbitron", 26, "bold"), text_color=COLOR_ACCENT).pack(
            side="left", padx=20)
        self.lbl_status = ctk.CTkButton(header, text="INITIALIZING...", fg_color=COLOR_WARN, width=120, hover=False,
                                        corner_radius=20)
        self.lbl_status.pack(side="right", padx=20)

        # --- LEFT PANEL (DNN & Controls) ---
        left = ctk.CTkFrame(self, width=280, fg_color=COLOR_PANEL)
        left.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(left, text="ENV PARAMETERS (DNN)", font=("Arial", 12, "bold"), text_color=COLOR_ACCENT).pack(
            pady=15)

        self.var_wind = tk.DoubleVar(value=20)
        self.var_temp = tk.DoubleVar(value=30)
        self.var_humid = tk.DoubleVar(value=40)

        self.add_slider(left, "Wind Speed (km/h)", self.var_wind, 0, 100)
        self.add_slider(left, "Temperature (°C)", self.var_temp, 0, 50)
        self.add_slider(left, "Humidity (%)", self.var_humid, 0, 100)

        self.risk_lbl = ctk.CTkLabel(left, text="RISK: LOW", font=("Arial", 18, "bold"), text_color=COLOR_SUCCESS)
        self.risk_lbl.pack(pady=10)

        ctk.CTkLabel(left, text="COMMANDS", font=("Arial", 12, "bold"), text_color=COLOR_ACCENT).pack(pady=(20, 5))

        ctk.CTkButton(left, text="🔥 SPAWN FIRE", fg_color=COLOR_FIRE, command=self.spawn_fire).pack(fill="x", padx=15,
                                                                                                    pady=5)
        ctk.CTkButton(left, text="🚁 OPTIMIZE (PSO)", fg_color=COLOR_ACCENT, command=self.run_pso_logic).pack(fill="x",
                                                                                                             padx=15,
                                                                                                             pady=5)
        ctk.CTkButton(left, text="📤 ANALYZE IMAGE (CNN)", fg_color=COLOR_PANEL, border_width=1, border_color="#334155",
                      command=self.upload_image).pack(fill="x", padx=15, pady=5)
        ctk.CTkButton(left, text="↺ RESET", fg_color="#475569", command=self.reset_sim).pack(fill="x", padx=15, pady=5)

        # --- CENTER (Map) ---
        center = ctk.CTkFrame(self, fg_color="transparent")
        center.grid(row=1, column=1, sticky="nsew", padx=0, pady=10)

        self.canvas = tk.Canvas(center, width=800, height=600, bg="#05080f", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", lambda e: self.spawn_fire(e.x, e.y))

        # --- RIGHT PANEL (Webcam & Metrics) ---
        right = ctk.CTkFrame(self, width=320, fg_color=COLOR_PANEL)
        right.grid(row=1, column=2, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(right, text="SURVEILLANCE (CNN)", font=("Arial", 12, "bold"), text_color=COLOR_ACCENT).pack(
            pady=15)

        self.cam_frame = ctk.CTkFrame(right, fg_color="black", height=240)
        self.cam_frame.pack(fill="x", padx=15)
        self.cam_frame.pack_propagate(False)
        self.cam_lbl = tk.Label(self.cam_frame, bg="black", text="CAM OFF", fg="white")
        self.cam_lbl.pack(expand=True, fill="both")

        self.btn_cam = ctk.CTkButton(right, text="▶ START CAMERA", command=self.toggle_cam)
        self.btn_cam.pack(fill="x", padx=15, pady=10)

        self.cnn_res = ctk.CTkLabel(right, text="Status: Waiting...", text_color=COLOR_TEXT_DIM)
        self.cnn_res.pack()

        ctk.CTkLabel(right, text="LIVE METRICS", font=("Arial", 12, "bold"), text_color=COLOR_ACCENT).pack(
            pady=(20, 10))
        self.met_fire = self.add_metric(right, "Active Fires", "0", COLOR_FIRE)
        self.met_drone = self.add_metric(right, "Drones Active", "0/8", COLOR_ACCENT)
        self.met_success = self.add_metric(right, "Extinguished", "0", COLOR_SUCCESS)

    # --- Slider Helper ---
    def add_slider(self, parent, text, var, vmin, vmax):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(f, text=text, font=("Arial", 10)).pack(anchor="w")
        ctk.CTkSlider(f, from_=vmin, to=vmax, variable=var, command=lambda x: self.update_dnn_risk()).pack(fill="x")

    def add_metric(self, parent, text, val, color):
        f = ctk.CTkFrame(parent, fg_color=COLOR_BG)
        f.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(f, text=text, font=("Arial", 11)).pack(side="left", padx=10, pady=5)
        l = ctk.CTkLabel(f, text=val, font=("Orbitron", 14, "bold"), text_color=color)
        l.pack(side="right", padx=10)
        return l

    # ==================== LOGIC MODULES ====================

    def update_dnn_risk(self):
        """Calculates Fire Risk using DNN with Manual Fallback [IMPROVEMENT: Fixed Crash]"""
        global GLOBAL_RISK_FACTOR

        # inputs: [Wind, Humidity, Temp, Vegetation(0.6), DaysNoRain(15), RegionCode(12), Elevation(500)]
        inputs_list = [
            float(self.var_wind.get()),
            float(self.var_humid.get()),
            float(self.var_temp.get()),
            0.6, 15, 12, 500
        ]

        risk_idx = 0
        ai_success = False

        # 1. Try AI Inference (with Tensor conversion fix)
        if self.dnn:
            try:
                # [FIX] Convert list to Tensor AND add batch dimension (1, 7)
                input_tensor = torch.tensor([inputs_list], dtype=torch.float32)

                with torch.no_grad():
                    output = self.dnn(input_tensor)
                    _, predicted = torch.max(output, 1)
                    risk_idx = predicted.item()

                print(f"✅ DNN PREDICTION: Index {risk_idx}")
                ai_success = True
            except Exception as e:
                print(f"⚠️ DNN Inference Failed ({e}) - Switching to Heuristic Mode")

        # 2. Manual Fallback (Heuristic Logic) if AI fails
        if not ai_success:
            # Logic: High Temp + High Wind - Humidity = Danger
            score = (self.var_temp.get() * 1.5) + (self.var_wind.get() * 1.2) - (self.var_humid.get() * 0.8)
            if score < 20:
                risk_idx = 0  # Low
            elif score < 50:
                risk_idx = 1  # Moderate
            elif score < 80:
                risk_idx = 2  # High
            else:
                risk_idx = 3  # Critical

        # 3. Update GUI & Physics based on Risk
        levels = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        colors = [COLOR_SUCCESS, "#facc15", "#f97316", "#ef4444"]
        self.risk_lbl.configure(text=f"RISK: {levels[risk_idx]}", text_color=colors[risk_idx])

        # [IMPROVEMENT] Scale spread speed significantly based on risk
        GLOBAL_RISK_FACTOR = [0.05, 0.2, 0.5, 0.8][risk_idx]

    def spawn_fire(self, x=None, y=None):
        if len(self.fires) >= MAX_FIRE_OBJECTS: return  # Safety check
        if x is None: x, y = random.randint(200, 600), random.randint(200, 400)
        self.fires.append(SimFire(x, y))
        self.trigger_rl_response()

    def trigger_rl_response(self):
        if not self.rl: return
        state = (len(self.fires) * 10, 10, len(self.fires), 8)
        action = self.rl.choose_action(state, training=False)
        action_name = self.rl.action_names[action]
        if "Drones" in action_name:
            self.run_pso_logic()

    def run_pso_logic(self):
        self.lbl_status.configure(text="SWARM OPTIMIZING", fg_color=COLOR_ACCENT)
        standby = [d for d in self.drones if d.role == "STANDBY"]
        if not standby or not self.fires: return

        if self.pso:
            fire_data = [{'x': f.x, 'y': f.y, 'intensity': f.intensity, 'id': f.id} for f in self.fires]
            self.pso.fire_locations = fire_data
            cost, positions = self.pso.optimize(n_iterations=10)

            for i, drone in enumerate(standby):
                if i * 2 + 1 < len(positions):
                    tx, ty = positions[i * 2], positions[i * 2 + 1]
                    closest = min(self.fires, key=lambda f: math.hypot(f.x - tx, f.y - ty))
                    if closest.active:
                        drone.target_fire = closest
                        drone.state = "DEPLOYED"

    def reset_sim(self):
        self.fires = []
        self.extinguished_count = 0
        for d in self.drones:
            d.state = "PATROLLING" if d.role == "PATROL" else "IDLE"
            d.water = 100
            d.vx, d.vy = 0, 0
        self.canvas.delete("all")

    # ==================== WEBCAM & IMAGE MODULE ====================

    def toggle_cam(self):
        if self.webcam_active:
            self.webcam_active = False
            self.btn_cam.configure(text="▶ START CAMERA", fg_color=COLOR_ACCENT)
            if self.cap: self.cap.release()
            self.cam_lbl.configure(image="", text="CAM OFF")
        else:
            self.webcam_active = True
            self.cap = cv2.VideoCapture(0)
            self.btn_cam.configure(text="⏹ STOP CAMERA", fg_color=COLOR_FIRE)
            threading.Thread(target=self.cam_loop, daemon=True).start()

    def cam_loop(self):
        frame_cnt = 0
        while self.webcam_active:
            ret, frame = self.cap.read()
            if not ret: break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((280, 200))
            imgtk = ImageTk.PhotoImage(image=img)
            try:
                self.cam_lbl.configure(image=imgtk, text="")
                self.cam_lbl.image = imgtk
            except:
                pass

            frame_cnt += 1
            if frame_cnt % 30 == 0 and self.cnn:
                cv2.imwrite("temp_cam.jpg", frame)
                label, conf = self.cnn.predict("temp_cam.jpg")
                color = COLOR_FIRE if label == "Fire" else COLOR_SUCCESS
                try:
                    self.cnn_res.configure(text=f"CNN: {label} ({conf:.1f}%)", text_color=color)
                    if label == "Fire" and conf > 75:
                        if not self.fires: self.after(0, self.spawn_fire)
                except:
                    pass

    def upload_image(self):
        path = filedialog.askopenfilename()
        if path and self.cnn:
            label, conf = self.cnn.predict(path)
            tk.messagebox.showinfo("CNN Result", f"Analysis Complete:\n\nLabel: {label}\nConfidence: {conf:.2f}%")
            if label == "Fire":
                if tk.messagebox.askyesno("Action Required", "Fire detected! Deploy swarm?"):
                    self.spawn_fire()

    # ==================== VISUALIZATION LOOP ====================

    def animate_loop(self):
        """Main Loop [IMPROVEMENT: Added Spread Logic & Safety]"""
        self.canvas.delete("all")

        # 1. Background Grid
        for i in range(0, 2000, 50):
            self.canvas.create_line(i, 0, i, 2000, fill="#1e293b")
            self.canvas.create_line(0, i, 2000, i, fill="#1e293b")

        # 2. Draw Bases
        self.canvas.create_oval(90, 90, 110, 110, outline=COLOR_ACCENT, width=2)
        self.canvas.create_text(100, 120, text="BASE ALPHA", fill=COLOR_TEXT_DIM, font=("Arial", 8))

        self.canvas.create_oval(690, 490, 710, 510, outline=COLOR_ACCENT, width=2)
        self.canvas.create_text(700, 520, text="BASE BRAVO", fill=COLOR_TEXT_DIM, font=("Arial", 8))

        # 3. Update Simulation Logic
        dead_fires = [f for f in self.fires if not f.active]
        if dead_fires: self.extinguished_count += len(dead_fires)

        self.fires = [f for f in self.fires if f.active]

        # [IMPROVEMENT] Fire Spread Mechanism
        new_fires = []
        current_fire_count = len(self.fires)

        for f in self.fires:
            f.update()
            # Only spread if intense enough, and system limit not reached
            if f.intensity > 0.5 and current_fire_count < MAX_FIRE_OBJECTS:
                # Spread probability based on Risk Factor
                # Critical Risk (0.8) -> ~5% spread chance per frame
                spread_chance = 0.01 + (GLOBAL_RISK_FACTOR * 0.05)

                if random.random() < spread_chance:
                    angle = random.uniform(0, 6.28)
                    dist = random.randint(25, 45)
                    nx = f.x + math.cos(angle) * dist
                    ny = f.y + math.sin(angle) * dist

                    if 50 < nx < 750 and 50 < ny < 550:
                        new_fires.append(SimFire(nx, ny))
                        current_fire_count += 1

        self.fires.extend(new_fires)

        for d in self.drones: d.update(self.drones, self.fires)

        # 4. Draw Swarm Network
        active_drones = [d for d in self.drones if d.state != "IDLE"]
        for i in range(len(active_drones)):
            for j in range(i + 1, len(active_drones)):
                d1, d2 = active_drones[i], active_drones[j]
                if math.hypot(d1.x - d2.x, d1.y - d2.y) < 100:
                    self.canvas.create_line(d1.x, d1.y, d2.x, d2.y, fill="#1e40af", width=1)

        # 5. Draw Fires
        for f in self.fires:
            for p in f.particles:
                self.canvas.create_oval(p.x, p.y, p.x + 2, p.y + 2, fill=p.color, outline="")
            r = f.display_radius
            self.canvas.create_oval(f.x - r, f.y - r, f.x + r, f.y + r, fill=COLOR_FIRE, outline="#ef4444")
            self.canvas.create_oval(f.x - r * 0.6, f.y - r * 0.6, f.x + r * 0.6, f.y + r * 0.6, fill="#fca5a5",
                                    outline="")

        # 6. Draw Drones
        deployed_count = 0
        for d in self.drones:
            if d.state in ["DEPLOYED", "EXTINGUISHING"]: deployed_count += 1
            if d.target_fire and d.state != "RETURNING":
                self.canvas.create_line(d.x, d.y, d.target_fire.x, d.target_fire.y, fill=COLOR_WARN, dash=(2, 2))

            for p in d.water_particles:
                self.canvas.create_oval(p.x, p.y, p.x + 3, p.y + 3, fill="#60a5fa", outline="")

            if len(d.trail) > 1:
                self.canvas.create_line(d.trail, fill="#3b82f6", width=2, smooth=True)

            color = COLOR_ACCENT if d.role == "STANDBY" else COLOR_SUCCESS
            if d.state == "EXTINGUISHING": color = "#f472b6"

            self.canvas.create_line(d.x - 8, d.y - 8, d.x + 8, d.y + 8, fill="white", width=2)
            self.canvas.create_line(d.x + 8, d.y - 8, d.x - 8, d.y + 8, fill="white", width=2)
            self.canvas.create_oval(d.x - 3, d.y - 3, d.x + 3, d.y + 3, fill=color, outline="")

            if d.state == "PATROLLING":
                ex = d.x + math.cos(d.scan_angle) * 20
                ey = d.y + math.sin(d.scan_angle) * 20
                self.canvas.create_line(d.x, d.y, ex, ey, fill="#10b981", width=1)

            self.canvas.create_text(d.x, d.y - 15, text=f"D{d.id}", fill="white", font=("Arial", 8))

        # 7. Update Metrics
        self.met_fire.configure(text=str(len(self.fires)))
        self.met_drone.configure(text=f"{deployed_count}/8")
        self.met_success.configure(text=str(self.extinguished_count))

        self.after(50, self.animate_loop)


if __name__ == "__main__":
    app = PyroGuardApp()
    app.mainloop()