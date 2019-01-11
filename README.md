# AQUARIUM

**(art)ificial** *project02*

<img src="https://raw.githubusercontent.com/greentfrapp/artificial-02-aquarium/master/AQUARIUM/all.jpg" width="250" alt="AQUARIUM"></img>

This repository contains the script that created [AQUARIUM](https://github.com/greentfrapp/artificial-02-aquarium/tree/master/AQUARIUM).

### Instructions

Run the following to generate an image with the default parameters. The default resulting image should be classified as 'tench' by Keras' ResNet50.

```
$ python3 main.py
```

**Parameters**

- ImageNet Class (by ID): `-class=949`
- Number of steps: `-steps=2560`
- Output image size (pixels): `-size=1024`
- Output file: `-path=image.jpg`

### Help

Run with the `--help` flag to get a list of possible flags.

```
$ python3 main.py --help
```
