import argparse

from barnes_hut import simulate
from profiler import profile_and_save


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--frames', type=int, default=5)
    parser.add_argument('-p', '--profile', action='store_true', default=False)

    args = parser.parse_args()
    return args


def main(frames: int, profile: bool):
    # if profile:
    #     profile_and_save(lambda: simulate(frames))()
    # else:
        simulate(frames)


if __name__ == '__main__':
    args = parse()
    print(args)
    main(args.frames, args.profile)    