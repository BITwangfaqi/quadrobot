"""Export Pinocchio/CasADi expressions without mixing Python extension ABIs.

The controller loads serialized CasADi functions, never the worker's libraries.
"""
import json
from pathlib import Path
import sys


def main():
    from robot_description import RobotDescription
    from rigid_body_model import WholeBodyModel
    model = WholeBodyModel(RobotDescription.from_world(sys.argv[1]))
    if model.pin is None:
        raise RuntimeError('The export interpreter must provide working Pinocchio')
    functions = model.casadi_functions()
    if model.casadi_backend != 'pinocchio.casadi':
        raise RuntimeError('The export interpreter must provide pinocchio.casadi')
    directory = Path(sys.argv[2])
    for name, function in functions.items():
        function.save(str(directory / (name + '.casadi')))
    (directory / 'manifest.json').write_text(json.dumps(dict(
        functions=list(functions), pinocchio=model.pin.__version__, backend=model.casadi_backend)))


if __name__ == '__main__':
    main()
