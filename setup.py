from distutils.core import setup

#TODO: Add page to docsrc as url and make separate email adress
setup(name='qose',
      version='0.1.0',
      description='a PennyLane library to implement OSE for parameterized quantum circuits.  It is pronounced cozy/cosy',
      author='Aroosa Ijaz, Kathleen Hamilton, Jelena Mackeprang, Roeland Wiersema',
      author_email='',
      url='',
      install_requires=['numpy', 'pennylane', 'torch', 'autograd'],
      python_requires='>=3.6',
      packages=['qose',],
     )