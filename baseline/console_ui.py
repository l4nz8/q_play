import os
import torch


def sort_steps(filename):
    """
    Extracts the step count from a checkpoint filename for sorting purposes.

    Args:
    - filename (str): The checkpoint filename

    Returns:
    - int: The extracted step count or 0 if not found
    """
    parts = filename.split("_")
    steps_part = parts[3]  # Step count is the fourth part of the filename
    try:
        steps = int(steps_part)
    except ValueError:
        steps = 0  # Default to 0 if conversion fails
    return steps


def list_available_models_and_params(save_dir):
    """
    Lists available models in the save directory and prints their parameters.

    Args:
    - save_dir (str): Directory where checkpoint files are stored

    Returns:
    - List[str]: A sorted list of checkpoint filenames
    """
    if not os.path.exists(save_dir):
        print("Checkpoint directory not found, creating a directory.")
        os.makedirs(save_dir)
        return []

    checkpoints = [file for file in os.listdir(save_dir) if file.endswith(".chkpt")]
    checkpoints.sort(key=sort_steps)  # Sorts files by step count
    if not checkpoints:
        print("No checkpoints available, start a new training session.")
        return []

    print("Available model versions and their parameters:")
    for idx, model in enumerate(checkpoints, start=1):
        checkpoint_path = os.path.join(save_dir, model)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            exploration_rate = checkpoint.get("exploration_rate", "NaN")
            curr_step = checkpoint.get("curr_step", "NaN")
            avg_reward = checkpoint.get("avg_reward", 0)
            loss_log = checkpoint.get("avg_loss", 0)
            print(
                f"{idx}: {model[:9]} Step: {curr_step} "
                f"| Exploration Rate: {exploration_rate:.4f} "
                f"| avg_reward: {avg_reward:.4f} "
                f"| avg_loss: {loss_log:.4f}"
            )
        except Exception as e:
            print(f"Could not read {model}: {e}")
    return checkpoints


def load_model_interactively(available_checkpoints, save_dir, mario, args):
    """
    Interactively selects and loads a model checkpoint.

    Args:
    - available_checkpoints (List[str]): List of available checkpoint filenames
    - save_dir (str): Directory where checkpoint files are stored
    - mario: The Mario AI agent instance
    - args: Command-line arguments or other options
    """
    if len(available_checkpoints) == 0:
        return  # No checkpoints available

    model_version = None
    if len(available_checkpoints) > 1:
        version_input = input(
            "Please enter the version number (idx) to load (or press Enter to use the latest): "
        )
        if version_input:
            try:
                model_version = int(version_input)
                if model_version < 1 or model_version > len(available_checkpoints):
                    raise ValueError("Invalid Index number selected.")
            except ValueError:
                print("Invalid input. Exiting.")
                exit()

    # Select the latest or specified checkpoint
    selected_checkpoint = available_checkpoints[
        -1 if model_version is None else model_version - 1
    ]
    checkpoint_path = os.path.join(save_dir, selected_checkpoint)
    print(f"Loading model: {selected_checkpoint}")
    mario.load_model(checkpoint_path)  # Load the selected model

    # Adjust the exploration rate if specified
    if args.exploration is not None:
        mario.exploration_rate = args.exploration
        print(f"Exploration Rate set to {args.exploration}.")
