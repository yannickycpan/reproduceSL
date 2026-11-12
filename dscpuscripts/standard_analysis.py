"""Current dataset reporting; FD001 remains on its frozen reporting path."""
from __future__ import annotations

import argparse

from dscpuscripts.standard_runner import CAMPAIGNS


def main():
    from dscpuscripts.common import cpu_environment
    cpu_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', choices=CAMPAIGNS)
    parser.add_argument('mode', choices=('check', 'run', 'plots', 'endpoints'),
                        nargs='?', default='run')
    parser.add_argument('--workers', type=int, default=1,
                        help='Application test-inference processes, default 1')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be a positive integer')
    if args.campaign == 'fd001':
        if args.mode == 'endpoints' or args.workers != 1:
            parser.error('FD001 uses its unchanged original analysis; endpoints/workers apply to applications')
        from dscpuscripts.fd001_analysis import run as analyze_fd001
        return analyze_fd001(args.mode)
    from dscpuscripts.application_analysis import run
    return run(args.campaign, args.mode, workers=args.workers)


if __name__ == '__main__':
    main()
