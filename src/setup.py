import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
VERSION = open(os.path.join(here, 'VERSION')).read().strip()

required_eggs = [
    'psycopg2>=2.7.4'
]

setup(
    name='ipts-lambda',
    version=VERSION,
    packages=[
        'bx.awslambda',
    ],
    install_requires=required_eggs,
    include_package_data=True,
    zip_safe=False,
    entry_points={
    }
)
