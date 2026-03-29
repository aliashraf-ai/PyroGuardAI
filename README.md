# 🔥 PyroGuard AI: Multi-Agent Wildfire Simulation

Most AI projects stop at a single model predicting a single output. I built PyroGuard because I wanted to figure out how to orchestrate multiple distinct AI architectures into a single, closed-loop pipeline. 

This is a simulated disaster response environment. It forces four different AI modules—Predictive, Visual, Strategic, and Tactical—to share data and make decisions in real-time to contain a wildfire.

### ⚙️ System Architecture 

```text
[Environmental Data]      [Satellite/Drone Feed]
        │                           │
        ▼                           ▼
  ┌───────────┐               ┌───────────┐
  │    DNN    │               │    CNN    │
  │ Forecaster│               │   Eyes    │
  └─────┬─────┘               └─────┬─────┘
        │      (Risk Score)         │ (Fire Coordinates)
        └─────────────┐ ┌───────────┘
                      ▼ ▼
                ┌─────────────┐
                │ Q-LEARNING  │
                │  Commander  │
                └──────┬──────┘
                       │ (Deployment Strategy)
                       ▼
                ┌─────────────┐
                │     PSO     │
                │   The Fleet │
                └─────────────┘
