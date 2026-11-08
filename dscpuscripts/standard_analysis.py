"""Current dataset reporting; FD001 remains on its frozen reporting path."""
from __future__ import annotations

import argparse

from dscpuscripts.standard_runner import CAMPAIGNS


def main():
    from dscpuscripts.common import cpu_environment
    cpu_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', choices=CAMPAIGNS)
    parser.add_argument('mode', choices=('check', 'run', 'plots'), nargs='?', default='run')
    args = parser.parse_args()
    if args.campaign == 'fd001':
        from dscpuscripts.fd001_analysis import run as analyze_fd001
        return analyze_fd001(args.mode)
    from dscpuscripts.application_analysis import run
    return run(args.campaign, args.mode)


if __name__ == '__main__':
    main()
