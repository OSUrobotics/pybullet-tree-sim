## TODO
1. Add basic cylinder to world. Dynamically create URDF.
1. Separate camera class
    1. ToF class inherits camera classs
1. Dynamically populate UR URDF. Allow for various end-effectors and robot configurations.
    1. Make sure to include camera and other sensors.
1. Add generic robot class (from Abhinav's code)
    1. Panda/UR5
    1. End-effector
1. Make mini orchard from a set of URDFs
    1. Space out like normal orchard
    1. Be aware of memory issues in PyB
1. Change pkl files to hdf5. (lower priority)
1. Test
    1. Various tof configurations

## Installation

#### General use
```
python -m pip install .
```

#### Development
```
python -m pip install -e.
```


### Useful RegEx commands...
For replacing .mtl file paths
```
From: ([a-zA-Z0-9_]*).mtl\n
To:   ../mtl/$1.mtl\n

```
