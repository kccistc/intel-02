# Installation
```
sudo apt install v4l-utils
```

# Supported format / resolution / fps list
```
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

# Supported tuning parameters
```
v4l2-ctl --device=/dev/video0 --list-ctrls-menus
```

# How to tune
* Get value
```
v4l2-ctl --device=/dev/video0 --get-ctrl brightness
v4l2-ctl --device=/dev/video0 --get-ctrl contrast
v4l2-ctl --device=/dev/video0 --get-ctrl saturation
v4l2-ctl --device=/dev/video0 --get-ctrl hue
v4l2-ctl --device=/dev/video0 --get-ctrl gamma
v4l2-ctl --device=/dev/video0 --get-ctrl power_line_frequency
v4l2-ctl --device=/dev/video0 --get-ctrl sharpness
v4l2-ctl --device=/dev/video0 --get-ctrl backlight_compensation
```
* Set value
```
v4l2-ctl --device=/dev/video0 --set-ctrl brightness=0
v4l2-ctl --device=/dev/video0 --set-ctrl contrast=0
v4l2-ctl --device=/dev/video0 --set-ctrl saturation=0
v4l2-ctl --device=/dev/video0 --set-ctrl hue=0
v4l2-ctl --device=/dev/video0 --set-ctrl gamma=0
v4l2-ctl --device=/dev/video0 --set-ctrl power_line_frequency=0
v4l2-ctl --device=/dev/video0 --set-ctrl sharpness=0
v4l2-ctl --device=/dev/video0 --set-ctrl backlight_compensation=1
```
