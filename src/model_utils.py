import itertools
from typing import Dict, Any, List, Type
import importlib

def get_model_class(model_path: str) -> Type:
    """
    Dynamically imports and returns a model class based on a string path.
    Example: 'sklearn.ensemble.RandomForestClassifier'
    """
    try:
        # Split the path into module and class name
        module_name, class_name = model_path.rsplit('.', 1)
        
        # Dynamically import the module
        module = importlib.import_module(module_name)
        
        # Get the class from the module
        model_class = getattr(module, class_name)
        
        return model_class
    except Exception as e:
        raise ImportError(f"Could not import model class '{model_path}'. Error: {e}")


def get_hyperparameter_combinations(hyperparams: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Takes a dict of hyperparameters (where values can be single items or lists)
    and returns a list of dictionaries, one for every combination.
    """
    # Identify parameters that are lists (i.e., parameters to be tuned)
    #    and parameters that are single values (i.e., fixed parameters)
    tuning_params = {k: v for k, v in hyperparams.items() if isinstance(v, list)}
    fixed_params = {k: v for k, v in hyperparams.items() if not isinstance(v, list)}

    # Extract keys and list-values for the Cartesian product
    keys = tuning_params.keys()
    values = tuning_params.values()

    # Generate the Cartesian product (all combinations)
    combinations = [
        dict(zip(keys, combination))
        for combination in itertools.product(*values)
    ]
    
    # Merge fixed parameters into every combination
    #    If no lists are found, 'combinations' will contain one empty dict, 
    #    which is correctly handled by the merge.
    final_combinations = []
    if not combinations:
        final_combinations = [fixed_params]
    else:
        for combo in combinations:
            # Merging the combination with the fixed parameters
            final_combinations.append({**fixed_params, **combo})
            
    return final_combinations