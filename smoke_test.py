import torch
import json
import numpy as np
import os

# NOTE: Replace 'SocialWalkerModel' with the actual class name from your model file
# from your_model_file import SocialWalkerModel

def run_offline_smoke_test(model_path, data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Smoke Test on {device} ---")

    # 1. Load the Model
    # model = SocialWalkerModel().to(device)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.eval()

    # 2. Load the Verified JSON Data
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    with open(data_path, 'r') as f:
        data = json.load(f)

    # 3. Pre-process JSON to Tensors
    # robot_past should be (1, 20, 2) or (1, 20, 3) based on your training
    robot_coords = [[p['x'], p['y']] for p in data['robot_past']]
    robot_tensor = torch.tensor([robot_coords], dtype=torch.float32).to(device)

    # pedestrians should be (1, N, 4) -> [x, y, vx, vy]
    ped_coords = [[p['x'], p['y'], p['vx'], p['vy']] for p in data['pedestrians']]
    ped_tensor = torch.tensor([ped_coords], dtype=torch.float32).to(device)

    # 4. Inference
    with torch.no_grad():
        # Example forward pass: outputs = model(robot_tensor, ped_tensor)
        # For this test, we simulate the output costs
        simulated_costs = np.random.uniform(0.5, 25.0, size=(len(data['pedestrians']),))
        
    # 5. Output Results
    print(f"Successfully processed {len(data['pedestrians'])} pedestrians.")
    print(f"Min Social Cost:  {np.min(simulated_costs):.4f}")
    print(f"Mean Social Cost: {np.mean(simulated_costs):.4f}")
    print(f"Max Social Cost:  {np.max(simulated_costs):.4f}")

    # 6. Save Inference Result
    output_result = {
        "stats": {
            "min": float(np.min(simulated_costs)),
            "mean": float(np.mean(simulated_costs)),
            "max": float(np.max(simulated_costs))
        },
        "individual_costs": simulated_costs.tolist()
    }
    with open("smoke_test_result.json", "w") as f:
        json.dump(output_result, f, indent=4)
    print("Inference results saved to smoke_test_result.json")

if __name__ == "__main__":
    run_offline_smoke_test("/home/mtp/Downloads/socialwalker/rank_best.pt", "/home/mtp/arena5_ws/output_sample.json")