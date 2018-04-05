from setuptools import setup

setup(
    name="PythonBroker",
    version="0.1",
    py_modules=['main'],
    install_requires=[
        'Click',
        'numpy',
        'keras',
        ],
    entry_points='''
        [console_scripts]
        agent=main:cli
    ''')
