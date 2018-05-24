from setuptools import setup

setup(
    name="BrokerPython",
    version="0.1",
    py_modules=['main'],
    install_requires=[
        'Click',
        'grpcio',
        'grpcio-tools'
        'gym',
        'h5py',
        'keras',
        'keras-rl',
        'mypy',
        'mypy-protbuf',
        'numpy',
        'pandas',
        'PyDispatcher',
        'protobuf',
        'scikit-learn',
        'tensorflow-gpu=1.8.0',
        ],
    extras_require = {
        'visualize':[
            'jupyter',
            'tensorboard'
            ],
        'tests':[
            'nose'
            ]
        },
    entry_points='''
        [console_scripts]
        agent=main:cli
    ''')
