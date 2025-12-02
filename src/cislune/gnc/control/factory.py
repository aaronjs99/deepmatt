"""Factory for creating controller instances."""
from typing import Dict, Any

from cislune.gnc.control.base_controller import Controller
from cislune.gnc.control.sdre_controller import SDREController


def create_controller(controller_type: str, **kwargs: Any) -> Controller:
    """Factory function for creating controllers.
    
    Args:
        controller_type: Controller type string ('sdre', 'pd', 'smc', 'mpc')
        **kwargs: Controller-specific parameters
        
    Returns:
        Controller instance
        
    Raises:
        ValueError: If controller_type is not recognized
        
    Example:
        >>> controller = create_controller('sdre', Q=np.diag([1e4]*6), R=np.eye(3)*0.1)
        >>> u = controller.compute_control(state, t)
    """
    controllers: Dict[str, type] = {
        'sdre': SDREController,
        # 'pd': PDController,  # TODO: Implement
        # 'smc': SMCController,  # TODO: Implement
        # 'mpc': MPCController,  # TODO: Implement
    }
    
    controller_class = controllers.get(controller_type.lower())
    if controller_class is None:
        available = ', '.join(controllers.keys())
        raise ValueError(
            f"Unknown controller type '{controller_type}'. "
            f"Available types: {available}"
        )
    
    return controller_class(**kwargs)
