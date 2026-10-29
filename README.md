# reproduceSL

The TD v7 neural application campaign on branch `fullTD` has its own
[HPC instructions](README_fullTD.md), PyTorch objectives, and frozen protocol.
Use those entry points for Electricity, METR-LA and FD001.

Example to run: python main.py tdsreg.json 0 

This will run the first configuration in the tdsreg.json file. The integer is used to index a paramter setting in the json file. 
