from setuptools import setup, find_packages


def parse_requirements(filename: str):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return [
        line.strip()
        for line in lines
        if line and not line.startswith('#')
    ]

setup(
    name='mask_the_face',
    version='0.0.1',
    install_requires=parse_requirements('./relaxed_requirements.txt'),
    packages=find_packages(),
    url='https://github.com/trassir/MaskTheFace',
    license='free',
    author='trassir',
    author_email='s.vakhreev@dssl.ru',
    description='Augmentation via mask pasting on faces'
)
