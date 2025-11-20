#!/usr/bin/env python3
"""
MLflow UI Launcher for Urdu Poetry Generation Experiments
Run this script to start the MLflow tracking UI
"""

import subprocess
import sys
import os

def start_mlflow_ui():
    """Start MLflow UI server"""
    print("🚀 Starting MLflow UI...")
    print("📊 MLflow UI will be available at: http://localhost:5000")
    print("🔍 Select 'Urdu_Poetry_Generation_Baseline' experiment to view results")
    print("Press Ctrl+C to stop the server\n")

    try:
        # Start MLflow UI
        cmd = ["mlflow", "ui", "--host", "127.0.0.1", "--port", "5000"]
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 MLflow UI stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting MLflow UI: {e}")
        print("Make sure MLflow is installed: pip install mlflow")
    except FileNotFoundError:
        print("❌ MLflow command not found. Make sure MLflow is installed and in PATH")

if __name__ == "__main__":
    start_mlflow_ui()