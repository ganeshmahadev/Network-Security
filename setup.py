from setuptools import find_packages, setup
from typing import List

requirement_lst:List[str]=[]
def get_requirements() -> List[str]: #this function returns list of requirements
    try:
        with open('requirements.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement=line.strip()
                ## ignore empty lines and -e .
                if requirement and requirement!= '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst

setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="Ganesh",
    author_email="ganeshmahadev2463@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)
