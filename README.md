# LLM-Based Heuristic Inventor

This project implements an **LLM-driven Evolutionary Program Search** to discover and refine novel heuristic algorithms for the Traveling Salesman Problem (TSP). By leveraging the coding and reasoning capabilities of Large Language Models (LLMs) like **Google Gemini** and **DeepSeek**, the system iteratively generates, tests, and improves Python code to solve combinatorial optimization problems.

## Academic Context

This project was developed as part of the course **Modeling and Optimization of Communication Networks** in the **Master's in Software Development & Cloud** program at the University of Macedonia.

## Project Overview

The core of this project is an automated discovery engine that:
1.  **Generates** initial heuristic code or starts with a seed.
2.  **Evaluates** the code's performance on a standardized 20-city TSP instance.
3.  **Feeds back** the performance and the code itself to the LLM.
4.  **Requests improvements** (Novelty Search), asking the LLM to invent new state variables or search dynamics to escape local optima and lower the tour cost.

### Key Components

*   **`HeuristicInventor.ipynb`**: The main research notebook containing the `HeuristicDiscoveryEngine`, `LLMClient`, and the experimental loop.
*   **`best_solver_gemini.py`**: The best heuristic discovered by Google's Gemini 2.5 Flash model. It implements a complex, multi-phase system with **6 novel state variables** (e.g., Crystallization Potential, Resonance Field) and adaptive move types.
*   **`best_solver_deepseek.py`**: The best heuristic discovered by the DeepSeek-R1 model. It converged on a highly efficient **Greedy approach with Spatial Repulsion**, sacrificing novelty for raw speed.

## Prerequisites

*   Python 3.9+
*   API Key for **Google Gemini** (if using Gemini)
*   Local Inference Server (e.g., LM Studio) compatible with OpenAI API (if using DeepSeek/Local models)

## Installation

1.  **Clone the repository** (if applicable).

2.  **Install Dependencies**:
    The project relies on standard data science libraries and LLM SDKs.
    ```bash
    pip install google-generativeai openai numpy matplotlib python-dotenv
    ```

3.  **Configuration**:
    Create a `.env` file in the project root to store your API keys securely:
    ```env
    # For Google Gemini
    GEMINI_API_KEY=your_gemini_api_key_here

    # For Local/DeepSeek (optional, defaults in notebook are usually localhost)
    # LOCAL_API_KEY=lm-studio
    ```

## Usage

### Running the Discovery Engine
Open `HeuristicInventor.ipynb` in Jupyter Notebook or VS Code. This notebook drives the entire process:
-   Initializes the problem environment.
-   Connects to the specified LLM provider (GEMINI or LOCAL).
-   Runs the evolutionary loop (e.g., for 10-20 iterations).
-   Plots the best routes and evolution of scores.

### Running the Best Solvers
You can run the standalone python scripts to test the best-discovered algorithms directly:

**Gemini's DPB Heuristic:**
```bash
python best_solver_gemini.py
