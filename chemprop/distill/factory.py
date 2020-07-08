NO_DISTILL_ERR = "Distill {} not in DISTILL_REGISTRY! Available distills are {}"
DISTILL_REGISTRY = {}


def RegisterDistill(distill_name):
    """Registers a distill."""

    def decorator(f):
        DISTILL_REGISTRY[distill_name] = f
        return f

    return decorator



def get_distill(args):
    if args.distill not in DISTILL_REGISTRY:
        raise Exception(
            NO_DISTILL_ERR.format(args.distill, DISTILL_REGISTRY.keys()))

    return DISTILL_REGISTRY[args.distill](args)
