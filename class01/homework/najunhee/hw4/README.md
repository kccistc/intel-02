# How to make virtual environment

## Why we have to use virtual environment?
* Python has a dependency. Thus, there are many conflicts between libraries' version
* Make each virtual env and install each library to use

## How to make?
1. Go to the homework directory.
~~~shell
cd github-training/intel_project/intel-02/class01/homework/najunhee
~~~
2. Make a directory to submit HW files.
~~~shell
mkdir hw4
~~~
3. Make a virtual environment (I assigned my environment name as ".open_env")
~~~shell
python3 -m venv .open_env
~~~
4. Activate the virtual environment
~~~shell
source .open_env/bin/activate
~~~

5. Deactivate the virtual environment
~~~shell
deactivate
~~~

