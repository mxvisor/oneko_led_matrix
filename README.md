# oneko_led_matrix


**Proof of concept** — a port of the classic `oneko` to a 64×64 RGB LED matrix with a desktop simulation mode.


[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)


## Overview


This project reproduces the on-screen cat behaviour on a physical LED matrix driven by the `rpi-rgb-led-matrix` library and on a host using `pygame` for development and testing.


Key points:
- Most of the Python source was obtained by converting/from the original C implementation using AI-assisted translation, then adapted and improved manually.
- Original C source (reference): http://www.daidouji.com/oneko/distfiles/oneko-1.2.sakura.5.tar.gz
- Hardware driver used: https://github.com/hzeller/rpi-rgb-led-matrix.
- Target hardware matrix size: **64×64**.
- Host simulation: runs in a `pygame` window to emulate the LED matrix when developing on a host without hardware.




## Demos


- Raspberry Pi Zero + 64×64 LED matrix demo:


![Pi demo](docs/pi_demo.gif)


- Host simulation (pygame window) demo:


![Pygame demo](docs/pygame_demo.gif)




## Requirements


- Python 3.10+
- `Pillow`
- For hardware mode on Raspberry Pi:
- `rpi-rgb-led-matrix` (build and install following the project docs)
- a compatible 64×64 RGB LED matrix and the necessary power and wiring
- run with root or with proper permissions to access GPIO
- For simulation mode (host):
- `pygame`
