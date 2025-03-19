#!/usr/bin/env python3
"""
test_confidence.py - Script to evaluate model identification confidence over multiple runs

This script tests the model identifier on any model multiple times and records
the confidence scores to evaluate the consistency and reliability of the identification.
"""

import os
import sys
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from statistics import mean, stdev

# Import the model identification system
from model_identifier import ModelIdentifier

# Number of test runs to perform
NUM_RUNS = 10

# Model to test
TARGET_MODEL = "phi4-mini:latest"

# Set up the identifier with default settings
identifier = ModelIdentifier(
    api_url="http://localhost:11434/api/generate",
    database_path="model_fingerprints.json",
    cache_dir=".cache"
)

def run_confidence_test():
    """Run multiple identification tests and track confidence."""
    
    print(f"Starting confidence testing for {TARGET_MODEL}...")
    print(f"Will run {NUM_RUNS} identification tests.")
    print("-" * 50)
    
    # Store results
    results = []
    
    # Run the tests
    for run_num in range(1, NUM_RUNS + 1):
        print(f"\nRun {run_num}/{NUM_RUNS}")
        
        # Record start time
        start_time = time.time()
        
        # Reset API call counter
        identifier.total_api_calls = 0
        
        # Clear identification history
        identifier.identification_history = []
        
        # Run identification
        try:
            result = identifier.identify_model(TARGET_MODEL, verbose=False)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record result
            result_data = {
                "run": run_num,
                "identified_model": result["identified_model"],
                "correct_identification": result["identified_model"] == TARGET_MODEL,
                "confidence": result["confidence"],
                "tests_run": result["tests_run"],
                "api_calls": result["api_calls"],
                "duration_seconds": duration,
                "top_matches": [
                    {"model": model, "score": score} 
                    for model, score, _ in result["all_matches"][:3]
                ]
            }
            
            # Add stage data if available
            if identifier.identification_history:
                for i, stage in enumerate(identifier.identification_history):
                    stage_name = stage["stage"]
                    result_data[f"stage_{i+1}_confidence"] = stage["matches"][0][1] if stage["matches"] else None
                    result_data[f"stage_{i+1}_tests"] = stage["tests_run"]
            
            results.append(result_data)
            
            # Print summary
            print(f"Identified as: {result['identified_model']} (Correct: {result_data['correct_identification']})")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Tests run: {result['tests_run']}")
            print(f"API calls: {result['api_calls']}")
            print(f"Duration: {duration:.2f} seconds")
            
        except Exception as e:
            print(f"Error in run {run_num}: {e}")
    
    # Return all results
    return results

def analyze_results(results):
    """Analyze and display test results."""
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Basic statistics
    accuracy = df["correct_identification"].mean() * 100
    avg_confidence = df["confidence"].mean() * 100
    avg_tests = df["tests_run"].mean()
    avg_api_calls = df["api_calls"].mean()
    
    print("\n" + "=" * 50)
    print(f"RESULTS SUMMARY FOR {TARGET_MODEL}")
    print("=" * 50)
    print(f"Number of runs: {len(df)}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Average confidence: {avg_confidence:.1f}% (±{df['confidence'].std()*100:.1f}%)")
    print(f"Average tests run: {avg_tests:.1f} (±{df['tests_run'].std():.1f})")
    print(f"Average API calls: {avg_api_calls:.1f} (±{df['api_calls'].std():.1f})")
    
    # Stage analysis if available
    if "stage_1_confidence" in df.columns:
        print("\nConfidence progression by stage:")
        stages = [col for col in df.columns if col.startswith("stage_") and col.endswith("_confidence")]
        for stage in stages:
            stage_num = stage.split("_")[1]
            avg_conf = df[stage].mean() * 100
            print(f"  Stage {stage_num}: {avg_conf:.1f}% (±{df[stage].std()*100:.1f}%)")
    
    # Create results directory
    os.makedirs("test_results", exist_ok=True)
    
    # Save detailed results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_results/confidence_test_{TARGET_MODEL.replace(':', '_')}_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to {results_file}")
    
    # Generate plots
    create_plots(df, TARGET_MODEL, timestamp)

def create_plots(df, target_model, timestamp):
    """Create visualizations of test results."""
    
    # Create figure directory
    os.makedirs("test_results/figures", exist_ok=True)
    
    # 1. Confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df["confidence"], bins=10, alpha=0.7, color="blue")
    plt.axvline(df["confidence"].mean(), color="red", linestyle="dashed", linewidth=2)
    plt.title(f"Confidence Score Distribution for {target_model}")
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"test_results/figures/confidence_dist_{target_model.replace(':', '_')}_{timestamp}.png")
    
    # 2. Confidence vs API calls
    plt.figure(figsize=(10, 6))
    plt.scatter(df["api_calls"], df["confidence"], alpha=0.7)
    plt.title(f"Confidence vs. API Calls for {target_model}")
    plt.xlabel("Number of API Calls")
    plt.ylabel("Confidence Score")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"test_results/figures/confidence_vs_calls_{target_model.replace(':', '_')}_{timestamp}.png")
    
    # 3. Stage progression (if available)
    stage_cols = [col for col in df.columns if col.startswith("stage_") and col.endswith("_confidence")]
    if len(stage_cols) > 1:
        plt.figure(figsize=(10, 6))
        
        # Calculate average confidence at each stage
        stage_data = []
        for col in stage_cols:
            stage_num = int(col.split("_")[1])
            stage_data.append((stage_num, df[col].mean(), df[col].std()))
        
        # Sort by stage number
        stage_data.sort(key=lambda x: x[0])
        
        # Plot
        stages, means, stds = zip(*stage_data)
        plt.errorbar(stages, means, yerr=stds, marker='o', linestyle='-', capsize=5)
        plt.title(f"Confidence Progression by Stage for {target_model}")
        plt.xlabel("Stage")
        plt.ylabel("Average Confidence Score")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"test_results/figures/stage_progression_{target_model.replace(':', '_')}_{timestamp}.png")
    
    print(f"Plots saved to test_results/figures/")

if __name__ == "__main__":
    # Check if the database has fingerprints
    if not os.path.exists("model_fingerprints.json"):
        print("Error: model_fingerprints.json not found. Please train some models first.")
        sys.exit(1)
    
    # Run the tests
    results = run_confidence_test()
    
    # Analyze and display results
    analyze_results(results)
