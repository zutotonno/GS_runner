import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='gs_runner',
     version='0.1',
     scripts=['gs_runner'] ,
     author="Antonio Ritacco",
     author_email="antonio.ritacco@santannapisa.it",
     description="A grid search and training tool for time-series analysis",
     long_description=long_description,
   long_description_content_type="text/markdown",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )