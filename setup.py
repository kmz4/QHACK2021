from setuptools import setup, find_packages
# placeholder for now
setup(
    name='qose',
    version='0.1',
    packages=['qose'],
    url='',
    license='',
    author='Aroosa Ijaz, Kathleen Hamilton, Jelena Mackeprang,Roeland Wiersema',
    author_email='',
    description='',
    install_requires=['numpy', 'pennylane', 'torch', 'autograd'],
    python_requires='>=3.6',
    package_dir = {'qose': 'qose'},
)
