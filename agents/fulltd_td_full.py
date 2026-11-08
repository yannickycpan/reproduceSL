"""Retired application name; import-only compatibility for frozen FD001.

The frozen common factory imports this symbol when constructing MSE/NLL.
No current campaign can construct this removed objective.
"""


class TDFull:
    def __init__(self, *args, **kwargs):
        raise ValueError(
            "td_full was removed. Use td_typed through "
            "dscpuscripts.application_objectives and application_protocol.")
