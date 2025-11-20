#!/usr/bin/env python3
"""
MLflow Cleanup Script for Urdu Poetry Generation Experiments
Run this script to clean up any existing MLflow runs before starting fresh
"""

import mlflow
import os

def cleanup_mlflow_runs():
    """Clean up any existing MLflow runs and experiments"""
    print("🧹 Cleaning up MLflow runs...")

    try:
        # End any active runs
        mlflow.end_run()
        print("✅ Ended any active MLflow runs")
    except Exception as e:
        print(f"ℹ️  No active runs to end: {e}")

    try:
        # Set the experiment (this will create it if it doesn't exist)
        mlflow.set_experiment("Urdu_Poetry_Generation_Baseline")
        print("✅ MLflow experiment ready: Urdu_Poetry_Generation_Baseline")
    except Exception as e:
        print(f"⚠️  Could not set experiment: {e}")

    print("🎉 MLflow cleanup complete!")
    print("\n📊 To view experiments:")
    print("   1. Run: python scripts/start_mlflow_ui.py")
    print("   2. Or manually: mlflow ui")
    print("   3. Open http://localhost:5000")

if __name__ == "__main__":
    cleanup_mlflow_runs()