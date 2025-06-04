import stim
import argparse
from distance import distance

def stimify_stabs(stabs):
    """
    Convert a stabilizer tableau into stim convention.
    """
    return [stim.PauliString(x) for x in stabs]

def stimify_symplectic(stabs):
    """
    Convert a symplectic tableau into stim convention.
    TODO
    """
    pass

def get_stabilizers(code):
    """
    Get the stim stabilizers of a given code
    """
    stabs = None
    if code == "5qubit":
        stabs = stimify_stabs(['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'])
    elif code == "repetition":
        stabs = stimify_stabs(['ZZI', 'ZIZ'])
    elif code == "cookie":
        pass
    elif code == "BB":
        offsets = {1, -1, 1j, -1j, 3 + 6j, -6 + 3j}
        w = 24
        h = 12

        def wrap(c: complex) -> complex:
            return c.real % w + (c.imag % h)*1j

        def index_of(c: complex) -> int:
            c = wrap(c)
            return int(c.real + c.imag * w) // 2
        
        stabs: list[stim.PauliString] = []
        for x in range(w):
            for y in range(h):
                if x % 2 != y % 2:
                    continue  # This is a data qubit.
                m = x + 1j*y
                basis = 'XZ'[x % 2]
                sign = -1 if basis == 'Z' else +1
                stabilizer = stim.PauliString(w * h // 2)
                for offset in offsets:
                    stabilizer[index_of(m + offset * sign)] = basis
                stabs.append(stabilizer)
    else:
        assert False, f"Unrecognized code name {code}"
    return stabs

def main(args):
    code = args.code
    stabs = get_stabilizers(code)
    dist = distance(stabs)
    return dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the distance of some CSS and non-CSS codes"
    )

    parser.add_argument(
        "--code", "-c",
        type=str,
        default='BB',
    )
    args = parser.parse_args()
    main(args)