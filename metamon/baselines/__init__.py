import poke_env

FORMATS = ["ou", "uu", "ubers", "nu"]
_blank = lambda: {f: [] for f in FORMATS}
BASELINES_BY_GEN = {i: _blank() for i in range(1, 5)}
ALL_BASELINES = {}
GENS = BASELINES_BY_GEN.keys()
GEN_DATA = {gen: poke_env.data.GenData(gen) for gen in GENS}


def register_baseline(gens: list[int] = GENS, formats: list[str] = FORMATS):
    def _register(Cls):
        for gen in gens:
            if gen in BASELINES_BY_GEN:
                gen_dict = BASELINES_BY_GEN[gen]
                for format in formats:
                    if format in gen_dict:
                        gen_dict[format].append(Cls)
                        ALL_BASELINES[Cls.__name__] = Cls
                    else:
                        raise ValueError(
                            f"Attempted to register Baseline `{Cls.__name__}` to an unsupported format `{format}` for generation `{gen}`"
                        )
            else:
                raise ValueError(
                    f"Atempted to register Baseline `{Cls.__name__}` to an unsupported generation `{gen}`"
                )
        return Cls

    return _register


from .base import Baseline
from . import heuristic
from . import model_based
