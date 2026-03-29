"""
PSO Drone Swarm Coordinator
Multi-objective optimization for wildfire response drone fleet
Uses Particle Swarm Optimization with 4 competing objectives
"""

import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import pickle
import os

class WildfireDroneSwarm:
    """
    Multi-objective PSO for coordinating fire-fighting drone fleet

    Objectives:
    1. Minimize response time to fire incidents
    2. Maximize area coverage
    3. Minimize fuel consumption (distance traveled)
    4. Balance workload across drones
    """

    def __init__(self, n_drones=8, map_size=(1000, 1000)):
        self.n_drones = n_drones
        self.map_size = map_size
        self.drone_positions = None
        self.fire_locations = []
        self.base_stations = [(200, 200), (800, 800), (500, 200)]
        self.cost_history = []

        print(f"🚁 Initializing Drone Swarm with {n_drones} drones")
        print(f"   Map size: {map_size[0]}x{map_size[1]} meters")
        print(f"   Base stations: {len(self.base_stations)}")

    def generate_fire_incidents(self, n_fires=5):
        """Generate random fire locations on map"""

        self.fire_locations = []
        for i in range(n_fires):
            x = np.random.randint(50, self.map_size[0] - 50)
            y = np.random.randint(50, self.map_size[1] - 50)
            intensity = np.random.uniform(0.3, 1.0)
            self.fire_locations.append({
                'x': x,
                'y': y,
                'intensity': intensity,
                'id': i
            })

        print(f"\n🔥 Generated {n_fires} fire incidents:")
        for fire in self.fire_locations:
            print(f"   Fire {fire['id']}: ({fire['x']}, {fire['y']}) "
                  f"Intensity: {fire['intensity']:.2f}")

        return self.fire_locations

    def calculate_distance(self, pos1, pos2):
        """Euclidean distance between two points"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def objective_response_time(self, positions):
        """Objective 1: Minimize total response time"""
        if len(self.fire_locations) == 0:
            return 0

        total_response = 0
        for i in range(self.n_drones):
            drone_pos = (positions[i*2], positions[i*2 + 1])
            min_dist = float('inf')
            for fire in self.fire_locations:
                fire_pos = (fire['x'], fire['y'])
                dist = self.calculate_distance(drone_pos, fire_pos)
                weighted_dist = dist / fire['intensity']
                min_dist = min(min_dist, weighted_dist)
            total_response += min_dist

        return total_response

    def objective_coverage(self, positions):
        """Objective 2: Maximize coverage"""
        drone_positions = [(positions[i*2], positions[i*2 + 1])
                          for i in range(self.n_drones)]

        centroid_x = np.mean([p[0] for p in drone_positions])
        centroid_y = np.mean([p[1] for p in drone_positions])

        variance = sum((p[0] - centroid_x)**2 + (p[1] - centroid_y)**2
                      for p in drone_positions) / self.n_drones

        return variance

    def objective_fuel(self, positions):
        """Objective 3: Minimize fuel consumption"""
        total_fuel = 0
        for i in range(self.n_drones):
            drone_pos = (positions[i*2], positions[i*2 + 1])
            min_base_dist = min(self.calculate_distance(drone_pos, base)
                               for base in self.base_stations)
            total_fuel += min_base_dist

        return total_fuel

    def objective_workload_balance(self, positions):
        """Objective 4: Balance workload"""
        if len(self.fire_locations) == 0:
            return 0

        drone_assignments = [0] * self.n_drones

        for fire in self.fire_locations:
            fire_pos = (fire['x'], fire['y'])
            min_dist = float('inf')
            nearest_drone = 0

            for i in range(self.n_drones):
                drone_pos = (positions[i*2], positions[i*2 + 1])
                dist = self.calculate_distance(drone_pos, fire_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_drone = i

            drone_assignments[nearest_drone] += 1

        mean_workload = len(self.fire_locations) / self.n_drones
        variance = sum((w - mean_workload)**2 for w in drone_assignments) / self.n_drones

        return variance

    def multi_objective_fitness(self, positions):
        """Combined fitness function with weighted objectives"""
        w1, w2, w3, w4 = 0.40, 0.20, 0.25, 0.15

        obj1 = self.objective_response_time(positions)
        obj2 = self.objective_coverage(positions)
        obj3 = self.objective_fuel(positions)
        obj4 = self.objective_workload_balance(positions)

        obj1_norm = obj1 / 1000
        obj2_norm = obj2 / 100000
        obj3_norm = obj3 / 1000
        obj4_norm = obj4 / 5

        total_cost = (w1 * obj1_norm + w2 * obj2_norm +
                     w3 * obj3_norm + w4 * obj4_norm)

        return total_cost

    def fitness_function(self, X):
        """Fitness function for PySwarms"""
        n_particles = X.shape[0]
        fitness = np.zeros(n_particles)

        for i in range(n_particles):
            fitness[i] = self.multi_objective_fitness(X[i])

        return fitness

    def optimize(self, n_iterations=50):
        """Run PSO optimization"""

        print(f"\n🔥 RUNNING PSO OPTIMIZATION...")
        print(f"   Particles: {self.n_drones * 3}")
        print(f"   Iterations: {n_iterations}")

        options = {'c1': 2.0, 'c2': 2.0, 'w': 0.9}

        lower_bounds = np.array([0, 0] * self.n_drones)
        upper_bounds = np.array([self.map_size[0], self.map_size[1]] * self.n_drones)
        bounds = (lower_bounds, upper_bounds)

        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_drones * 3,
            dimensions=self.n_drones * 2,
            options=options,
            bounds=bounds
        )

        print("\n📊 Optimization Progress:")
        print("-" * 60)

        best_cost, best_pos = optimizer.optimize(
            self.fitness_function,
            iters=n_iterations,
            verbose=True
        )

        self.cost_history = optimizer.cost_history
        self.drone_positions = best_pos

        print("\n" + "="*60)
        print(f"✅ OPTIMIZATION COMPLETE!")
        print(f"   Best fitness: {best_cost:.4f}")
        print("="*60)

        print(f"\n🚁 Optimal Drone Positions:")
        for i in range(self.n_drones):
            x = best_pos[i*2]
            y = best_pos[i*2 + 1]
            print(f"   Drone {i+1}: ({x:.0f}, {y:.0f})")

        return best_cost, best_pos

    def evaluate_solution(self):
        """Evaluate solution quality"""

        if self.drone_positions is None:
            print("❌ No solution found!")
            return

        print(f"\n📈 SOLUTION EVALUATION:")
        print("-" * 60)

        obj1 = self.objective_response_time(self.drone_positions)
        obj2 = self.objective_coverage(self.drone_positions)
        obj3 = self.objective_fuel(self.drone_positions)
        obj4 = self.objective_workload_balance(self.drone_positions)

        print(f"   Response Time Score: {obj1:.2f}")
        print(f"   Coverage Variance: {obj2:.2f}")
        print(f"   Fuel Consumption: {obj3:.2f} meters")
        print(f"   Workload Balance: {obj4:.2f}")

        fire_assignments = {}
        for fire in self.fire_locations:
            fire_pos = (fire['x'], fire['y'])
            min_dist = float('inf')
            assigned_drone = 0

            for i in range(self.n_drones):
                drone_pos = (self.drone_positions[i*2], self.drone_positions[i*2 + 1])
                dist = self.calculate_distance(drone_pos, fire_pos)
                if dist < min_dist:
                    min_dist = dist
                    assigned_drone = i

            fire_assignments[fire['id']] = {
                'drone': assigned_drone + 1,
                'distance': min_dist
            }

        print(f"\n🔥 Fire Assignments:")
        for fire_id, assignment in fire_assignments.items():
            print(f"   Fire {fire_id} → Drone {assignment['drone']} "
                  f"(distance: {assignment['distance']:.0f}m)")

        return fire_assignments

    def visualize_solution(self):
        """CREATE CRYSTAL CLEAR VISUALIZATION WITH LEGENDS AND LABELS"""

        if self.drone_positions is None:
            print("❌ No solution to visualize!")
            return

        # Create figure
        fig = plt.figure(figsize=(18, 9))
        fig.suptitle('🔥 PYROGUARD AI: WILDFIRE DRONE SWARM OPTIMIZATION (PSO)',
                    fontsize=16, fontweight='bold', y=0.98)

        # LEFT: Map with CLEAR LEGEND
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_xlim(-50, self.map_size[0] + 50)
        ax1.set_ylim(-50, self.map_size[1] + 50)
        ax1.set_aspect('equal')
        ax1.set_facecolor('#0a0a0a')
        ax1.grid(True, alpha=0.15, color='white', linestyle='--')

        # Title box
        title_box = FancyBboxPatch((200, 1020), 600, 60,
                                  boxstyle="round,pad=10",
                                  facecolor='#1a1a1a',
                                  edgecolor='cyan',
                                  linewidth=3)
        ax1.add_patch(title_box)
        ax1.text(500, 1050, '📍 DRONE DEPLOYMENT MAP',
                ha='center', va='center', fontsize=14,
                color='cyan', fontweight='bold')

        # Plot base stations with labels
        for idx, base in enumerate(self.base_stations):
            # Base circle
            circle = plt.Circle(base, 80, color='cyan', alpha=0.15, zorder=1)
            ax1.add_patch(circle)
            # Base marker
            ax1.plot(base[0], base[1], 's', color='cyan', markersize=18,
                    markeredgecolor='white', markeredgewidth=2, zorder=5)
            # Base label with box
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor='cyan',
                            edgecolor='white', linewidth=1.5, alpha=0.9)
            ax1.text(base[0], base[1] - 100, f'BASE-{idx+1}',
                    ha='center', color='black', fontsize=9,
                    fontweight='bold', bbox=bbox_props, zorder=6)

        # Calculate fire assignments
        fire_assignments = {}
        for fire in self.fire_locations:
            fire_pos = (fire['x'], fire['y'])
            min_dist = float('inf')
            assigned_drone = 0

            for i in range(self.n_drones):
                drone_pos = (self.drone_positions[i*2], self.drone_positions[i*2 + 1])
                dist = self.calculate_distance(drone_pos, fire_pos)
                if dist < min_dist:
                    min_dist = dist
                    assigned_drone = i

            if assigned_drone not in fire_assignments:
                fire_assignments[assigned_drone] = []
            fire_assignments[assigned_drone].append((fire, min_dist))

        # Colors for drones
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_drones))

        # Plot fires with intensity rings
        for fire in self.fire_locations:
            size = 150 + fire['intensity'] * 400
            # Outer glow
            ax1.scatter(fire['x'], fire['y'], s=size*1.5, c='red',
                       alpha=0.15, zorder=2)
            # Main fire
            ax1.scatter(fire['x'], fire['y'], s=size, c='red',
                       alpha=0.5 + fire['intensity'] * 0.3,
                       edgecolors='orange', linewidths=3, zorder=3)
            # Fire center
            ax1.scatter(fire['x'], fire['y'], s=80, c='yellow',
                       marker='*', edgecolors='red', linewidths=2, zorder=4)
            # Fire label with box
            bbox_props = dict(boxstyle='round,pad=0.4', facecolor='red',
                            edgecolor='yellow', linewidth=2, alpha=0.95)
            ax1.text(fire['x'], fire['y'] - 60,
                    f"🔥 FIRE-{fire['id']}\nIntensity: {fire['intensity']:.1f}",
                    ha='center', va='top', color='white', fontsize=8,
                    fontweight='bold', bbox=bbox_props, zorder=7)

        # Plot drones with assignments
        for i in range(self.n_drones):
            x = self.drone_positions[i*2]
            y = self.drone_positions[i*2 + 1]

            # Draw assignment lines
            if i in fire_assignments:
                for fire, dist in fire_assignments[i]:
                    ax1.plot([x, fire['x']], [y, fire['y']],
                            color=colors[i], alpha=0.7, linewidth=2.5,
                            linestyle='--', zorder=8)
                    # Distance label on line
                    mid_x = (x + fire['x']) / 2
                    mid_y = (y + fire['y']) / 2
                    ax1.text(mid_x, mid_y, f"{dist:.0f}m",
                            fontsize=7, color='white',
                            bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor=colors[i], alpha=0.8),
                            ha='center', zorder=9)

            # Drone marker
            ax1.scatter(x, y, s=600, c=[colors[i]], edgecolors='white',
                       linewidths=4, marker='^', zorder=10, alpha=0.95)
            # Drone label with box
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor=colors[i],
                            edgecolor='white', linewidth=2, alpha=0.95)
            label_text = f'DRONE-{i+1}'
            if i in fire_assignments:
                label_text += f'\n({len(fire_assignments[i])} fires)'
            ax1.text(x, y + 60, label_text,
                    ha='center', va='bottom', color='black', fontsize=9,
                    fontweight='bold', bbox=bbox_props, zorder=11)

        # LEGEND BOX
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='w', label='Drone (optimized position)',
                      markerfacecolor='gray', markersize=12, markeredgecolor='white',
                      markeredgewidth=2, linestyle='None'),
            plt.Line2D([0], [0], marker='*', color='w', label='Fire incident (intensity shown)',
                      markerfacecolor='red', markersize=15, markeredgecolor='yellow',
                      markeredgewidth=2, linestyle='None'),
            plt.Line2D([0], [0], marker='s', color='w', label='Base station (refuel/reload)',
                      markerfacecolor='cyan', markersize=10, markeredgecolor='white',
                      markeredgewidth=2, linestyle='None'),
            plt.Line2D([0], [0], color='white', linewidth=2, linestyle='--',
                      label='Drone → Fire assignment path')
        ]
        legend = ax1.legend(handles=legend_elements, loc='upper left',
                          fontsize=9, framealpha=0.95, edgecolor='white',
                          facecolor='#1a1a1a', labelcolor='white',
                          title='LEGEND', title_fontsize=11)
        legend.get_title().set_color('cyan')
        legend.get_title().set_weight('bold')

        ax1.set_xlabel('X Position (meters)', fontsize=12, color='white', fontweight='bold')
        ax1.set_ylabel('Y Position (meters)', fontsize=12, color='white', fontweight='bold')
        ax1.tick_params(colors='white', labelsize=10)

        # RIGHT: Objectives + Convergence
        ax2 = plt.subplot(2, 2, 2)

        # Calculate objectives
        obj1 = self.objective_response_time(self.drone_positions)
        obj2 = self.objective_coverage(self.drone_positions)
        obj3 = self.objective_fuel(self.drone_positions)
        obj4 = self.objective_workload_balance(self.drone_positions)

        objectives = ['Response\nTime', 'Area\nCoverage', 'Fuel\nEfficiency', 'Workload\nBalance']
        values = [obj1/10, obj2/1000, obj3/10, obj4*10]
        colors_bar = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24']

        bars = ax2.barh(objectives, values, color=colors_bar, alpha=0.9,
                       edgecolor='white', linewidth=2.5)
        ax2.set_xlabel('Normalized Score (Lower = Better)', fontsize=11, fontweight='bold')
        ax2.set_title('📊 Multi-Objective Performance Breakdown',
                     fontsize=12, fontweight='bold', pad=15)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        ax2.set_facecolor('#f8f9fa')

        # Add value labels with descriptions
        descriptions = ['Fast response', 'Even distribution', 'Low fuel use', 'Balanced load']
        for i, (bar, val, desc) in enumerate(zip(bars, values, descriptions)):
            ax2.text(val + 5, i, f'{val:.1f}  ({desc})',
                    va='center', fontsize=9, fontweight='bold')

        # Convergence
        ax3 = plt.subplot(2, 2, 4)
        iterations = range(len(self.cost_history))
        ax3.plot(iterations, self.cost_history, linewidth=3, color='#ff6b6b',
                marker='o', markersize=3, markevery=5)
        ax3.fill_between(iterations, self.cost_history, alpha=0.3, color='#ff6b6b')
        ax3.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Cost (Fitness)', fontsize=11, fontweight='bold')
        ax3.set_title('📈 PSO Convergence Over Time', fontsize=12, fontweight='bold', pad=15)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_facecolor('#f8f9fa')

        # Annotations
        improvement = ((self.cost_history[0] - self.cost_history[-1]) / self.cost_history[0]) * 100
        info_text = (f'✅ Improvement: {improvement:.1f}%\n'
                    f'🎯 Initial Cost: {self.cost_history[0]:.4f}\n'
                    f'🏆 Final Cost: {self.cost_history[-1]:.4f}\n'
                    f'🔄 Iterations: {len(self.cost_history)}')
        ax3.text(0.98, 0.97, info_text, transform=ax3.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen',
                         edgecolor='darkgreen', linewidth=2, alpha=0.9),
                fontweight='bold', family='monospace')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/pso_visualization.png', dpi=300, facecolor='white')
        print(f"\n🎨 High-quality visualization saved to: results/pso_visualization.png")
        plt.show()

    def save_model(self):
        """Save optimized solution"""
        os.makedirs('models', exist_ok=True)

        data = {
            'drone_positions': self.drone_positions,
            'fire_locations': self.fire_locations,
            'cost_history': self.cost_history,
            'n_drones': self.n_drones
        }

        with open('models/drone_swarm_pso.pkl', 'wb') as f:
            pickle.dump(data, f)

        print(f"💾 Model saved to: models/drone_swarm_pso.pkl")


def run_demo():
    """Demo: Run complete PSO optimization"""

    print("\n" + "="*60)
    print("🔥 PYROGUARD AI - WILDFIRE DRONE SWARM OPTIMIZATION")
    print("="*60)

    swarm = WildfireDroneSwarm(n_drones=8, map_size=(1000, 1000))
    swarm.generate_fire_incidents(n_fires=6)
    best_cost, best_positions = swarm.optimize(n_iterations=50)
    swarm.evaluate_solution()
    swarm.visualize_solution()
    swarm.save_model()

    print("\n✅ PSO COMPONENT COMPLETE!")
    print("🎯 Next: Run resource_allocator_rl.py for RL component")


if __name__ == '__main__':
    run_demo()