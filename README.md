# QHACK2021
Code repo for QHACK 2021 

QHACK 2021

# Proposed project: QOSE (Quantum Optimal Subarchitecture Extraction)
_pronounciation: "cozy"_ 

Determining the optimal quantum circuit structure for a parameterized model that will be trained as a quantum circuit classifier.  Heuristic ansatz design involves searching over a large parameter space including: depth, number of qubits, entangler layer design, initialization, optimizer choice (and associated hyperparameters), embedding..

Since running exponentially large circuits are liable to run into barren plateau problems, there is a need for fast searching over circuit structures coupled with intialization choices.

QOSE is a version of OSE (optimal subarchitecture extraction) that is an adaptation of the FPTAS algorithm introduced in arxiv.org/abs/2010.08512 for parameterized quantum circuits and written using PennyLane. 

## Installation instruction 
_don't do this yet_

```
git clone <this repo>
cd QHACK2021
pip install .
```

To run QOSE from the terminal call

```
python3 driver_code.py
```


References

    De Wynter (2020) "An Approximation Algorithm for Optimal Subarchitecture Extraction" [arxiv.org/abs/2010.08512]
    De Wynter and Perry (2020) "Optimal Subarchitecture Extraction for BERT" [arxiv.org/abs/2010.10499]


Contributers:
* Jelena Mackeprang 
* Roland Wiersema
* Aroosa Ijaz
* Kathleen Hamilton
* Yash Chitgopekar


*What are we keeping fixed when running SQOSE*:
* Parameter initializations are set to 0
* Optimizer?

