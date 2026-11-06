from typing import ClassVar

class IcfSolverParameters:
    __fields__: ClassVar[tuple] = ...  # read-only
    alpha_max: float
    enable_hessian_reuse: bool
    hessian_reuse_target_iterations: int
    linesearch_tolerance: float
    max_iterations: int
    max_linesearch_iterations: int
    min_tolerance: float
    print_solver_stats: bool
    use_dense_algebra: bool
    def __init__(self, **kwargs) -> None: ...
    def __copy__(self) -> IcfSolverParameters: ...
    def __deepcopy__(self, arg0: dict) -> IcfSolverParameters: ...
