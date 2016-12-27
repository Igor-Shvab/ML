# Feature Engineering for Machine Learning
This repository will give you a basic idea about how to 
use feature engineering along side with machine learning.
All the new feature engineering processes that describe here
have been carried out using python featureeng package.


#### Sample Datasets
1. NASA   : 
[Turbofan engine degradataion](https://ti.arc.nasa.gov/c/6/)

2. Yahoo  : 
[Webscope](https://webscope.sandbox.yahoo.com/)

Download datasets from above links. Copy extracted files into 
*dataset* folder in relevant project.

#### Setup Environment
All the dependencies given below need to be installed using 
*python pip install* in python 2.7

##### 1. Numpy
```
# pip install numpy
```
##### 2. Scipy
```
# pip install scipy
```
##### 3. Pandas
```
# pip install pandas
```
##### 4. H20
```
# pip install requests
# pip install tabulate
# pip install scikit-learn
# pip uninstall h2o
# pip install http://h2o-release.s3.amazonaws.com/h2o/rel-tutte/1/Python/h2o-3.10.2.1-py2.py3-none-any.whl
```
##### 5. Sklearn-Pandas
```
# pip install sklearn-pandas
```

##### NASA - Turbofan Engine Degradation

After copying data files into *dataset* folder executes following
python scripts in order.

###### 01-ConvertToCSV.py
This will convert all the data files in .txt 