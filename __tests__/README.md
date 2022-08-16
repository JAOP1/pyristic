# Unit test - pyristic
This folder has the test cases that cover currently the utilities that has pyristic. 

## How to execute
Before to execute the tests, you should install the following:
`pip install coverage` and include the required enviroment variables `export PYTHONPATH=$(pwd)/pyristic:$(pwd)/__tests__`.


Finally, if you wanted execute only one test file execute as follow in the root pyristic's folder:
```
coverage run __tests__/test_{name_file}.py
```
or check the coverage obtained after all tests:
```
cd __tests__/
coverage run -m unittest discover
coverage report -m
```
