# /***********************************************************
# *                                                         *
# * Copyright (c) 2025                                      *
# *                                                         *
# * University of California, Los Angeles                   *
# *                                                         *
# * Author : Aaron John-Sabu                                *
# * Contact: aaronjs@g.ucla.edu                             *
# *                                                         *
# ***********************************************************/

from typing import Dict, Any

from deepmatt.gnc.control.base_controller import Controller
from deepmatt.gnc.control.sdre_controller import SDREController
from deepmatt.gnc.control.pd_controller import PDController
from deepmatt.gnc.control.smc_controller import SlidingModeController


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
        "sdre": SDREController,
        "pd": PDController,
        "smc": SlidingModeController,
        # 'mpc': MPCController,  # TODO: Implement
    }

    controller_class = controllers.get(controller_type.lower())
    if controller_class is None:
        available = ", ".join(controllers.keys())
        raise ValueError(
            f"Unknown controller type '{controller_type}'. "
            f"Available types: {available}"
        )

    return controller_class(**kwargs)
