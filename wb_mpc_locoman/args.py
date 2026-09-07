# Arguments for each dynamics model
DYN_ARGS = {
    "centroidal_vel": {
        "include_base": True,  # whether base velocity is part of the input
    },
    "centroidal_acc": {
        "include_base": True,  # whether base acceleration is part of the input
    },
    "whole_body_acc": {
        "include_base": True,  # whether base acceleration is part of the input
    },
    "whole_body_rnea": {
        "include_acc": True,  # whether to include accelerations in the input (necessary for Fatrop due to structure detection!)
    },
    "whole_body_aba": {}  # the input just contains joint torques
}

# Arguments for each solver
SOLVER_ARGS = {
    "fatrop": {
        "opts": {
            "expand": True,
            "structure_detection": "auto",
            "debug": True,
            "fatrop.print_level": 0,
            "fatrop.max_iter": 10,
            "fatrop.tol": 1e-3,
            "fatrop.mu_init": 1e-4,
            "fatrop.warm_start_init_point": True,
            "fatrop.warm_start_mult_bound_push": 1e-7,
            "fatrop.bound_push": 1e-7,
        }
    },
    "ipopt": {
        "opts": {
            "expand": True,
            "ipopt.print_level": 0,
            "ipopt.max_iter": 100,
            "ipopt.tol": 1e-3,
            "ipopt.mu_init": 1e-4,
            # "ipopt.warm_start_init_point": True,
            # "ipopt.warm_start_mult_bound_push": 1e-7,
            # "ipopt.bound_push": 1e-7,
        }
    },
    "osqp": {
        "iters": 2,  # number of SQP iterations
        "opts": {
            "max_iter": 20,  # number of sub-iterations for each QP
            "alpha": 1.4,
            "rho": 2e-2,
            "warm_start": True,
            "adaptive_rho": False,
        }
    }
}
