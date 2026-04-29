import matplotlib.pyplot as plt

def create_final_dashboard(time, actual, pred_old, pred_new):
    plt.figure(figsize=(12, 6))
    
    # Plotting the ground truth
    plt.plot(time, actual, 'ro-', label='Actual Traffic (Reality)', linewidth=3)
    
    # Plotting the old "lazy" model (Mock data for comparison)
    plt.plot(time, pred_old, 'b--', alpha=0.5, label='Initial Model (Loss 1.36)')
    
    # Plotting your new "smart" model
    plt.plot(time, pred_new, 'g*-', label='Final Optimized Model (Loss 0.55)', linewidth=2)
    
    plt.title('Project Milestone: From "Mean Guessing" to "Pattern Recognition"', fontsize=14)
    plt.xlabel('Time Horizon (Minutes)', fontsize=12)
    plt.ylabel('Traffic Speed (MPH)', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Save it too, just in case the window doesn't pop up
    plt.savefig('final_dashboard.png')
    print("📊 Dashboard saved to final_dashboard.png")
    plt.show()

if __name__ == "__main__":
    # Example usage with your data
    time_steps = [5, 10, 15, 20, 25, 30]
    actual = [31, 23, 63, 15, 46, 64]
    
    # "Lazy" model predictions (closer to the mean, missing the spikes)
    pred_old = [40, 38, 42, 39, 41, 40] 

    # This mimics your current improved model's ability to "see" the dips!
    improved_pred = [35, 28, 55, 22, 48, 63]

    create_final_dashboard(time_steps, actual, pred_old, improved_pred)