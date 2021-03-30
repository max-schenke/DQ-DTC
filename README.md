# Deep Q Direct Torque Control (DQ-DTC) - Code Examples

The jupyter notebooks in this folder were created to present the training and usage of a DQ-DTC reinforcement learning drive control system. A deep Q direct torque controller can be trained via the DQ_DTC_training.ipynb notebook. The trained controller can then be tested with the DQ_DTC_validation_profile.ipynb notebook.
A benchmark for the performance of conventional control methods can be received via the MP_DTC_validation_profile.ipynb notebook.

The needed python packages are listed in the requirements.txt and can e.g. be installed via
```
	pip install -r requirements.txt
```
	

The python code files CustomKerasRL2Callbacks_torqueCtrl.py and Plot_TimeDomain_torqueCtrl.py implement extensions to the code
that were outsourced from the notebooks, they do not need to be executed manually. 
