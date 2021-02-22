# QHACK2021
Code repo for QHACK 2021 

QHACK 2021

Proposed project: POSE (Parameterized Quantum Cirucit OSE) (SQOSE? Some Quantum Optimal Subarchitecture Extraction)

Determining the optimal quantum circuit structure for a parameterized model (either a quantum classifier or regressor) involves searching over a large parameter space including: depth, number of qubits, entangler layer design, initialization, optimizer choice (and associated hyperparameters), embedding..

Since running exponentially large circuits are liable to run into barren plateau problems, there is a need for fast searching over circuit structures coupled with intialization choices.

POSE is a version of OSE (optimal subarchitecture extraction) that is adapted for Parameterized quantum circuits. It is the adaption of the FPTAS algorithm introduced in arxiv.org/abs/2010.08512.

From the AWS blog post: "We wanted to produce a version of BERT (replace BERT with a PQC) whose architectural parameters minimized its parameter size, inference speed, and error rate, and I wanted to show that these architectural parameters were optimal."

Q1: Do quantum circuits have the 𝐴𝐵𝑛𝐶 property? From the AWS blog post: "Under some circumstances, the algorithm’s error rate has a similar correlation. Furthermore, whenever the “cost” associated with the first (call it A) and last layers of the network is lower than that of the middle layers (B), the runtime of a solution will not explode. I call all these assumptions the ABnC property, which BERT turns out to have."

Q2: Are loss functions L-Lipschitz smooth?

Q3: Are quantum circuit gradients bounded (yes)

References

    De Wynter (2020) "An Approximation Algorithm for Optimal Subarchitecture Extraction" [arxiv.org/abs/2010.08512]
    De Wynter and Perry (2020) "Optimal Subarchitecture Extraction for BERT" [arxiv.org/abs/2010.10499]


Contributers:
* Jelena Mackeprang 
* Roland Wiersema
* Aroosa Ijaz
* Kathleen Hamilton
* Yash Chitgopekar
