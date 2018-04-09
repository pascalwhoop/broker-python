from setuptools import setup

setup(
    name="PythonBroker",
    version="0.1",
    py_modules=['main'],
    install_requires=[
        'Click',
        'numpy',
        'keras',
        'scikit-learn',
        'tensorflow-gpu',
        ],
    entry_points='''
        [console_scripts]
        agent=main:cli
    ''')
