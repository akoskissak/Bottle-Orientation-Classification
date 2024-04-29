# Bottle orientation classification

The goal of this project is to detect the orientation of a bottle based on its contours. Utilizes the MobileNetSSD model to provide real-time classification of bottle orientations through a webcam feed. Users can instantly determine whether a bottle
is positioned upwards, downwards, to the left, or to the right. Leveraging the power of computer vision, it offers a seamless and intuitive solution for assessing bottle
orientation. The orientation is determined by analyzing the position of the first white pixel from the left in relation to the midpoint of the contour image or the first white pixel from the top.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.

```bash
pip install opencv-python
pip install numpy
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.
