## TODO
1. Dynamically populate UR URDF. Allow for various end-effectors and robot configurations.
    1. Make sure to include camera and other sensors.
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
