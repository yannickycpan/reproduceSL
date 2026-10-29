# reproduceSL

The TD v7 neural application campaign on branch `fullTD` has separate PyTorch
objectives and a [frozen protocol](docs/fulltd_protocol.md) for Electricity,
METR-LA and FD001. Each explicit JSON manifest index identifies one candidate
run. HPC operating instructions are provided directly in the accompanying chat.

Example to run: python main.py tdsreg.json 0 

This will run the first configuration in the tdsreg.json file. The integer is used to index a paramter setting in the json file. 
