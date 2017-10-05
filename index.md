## Blog
### 9/28/17 
We’re still waiting for our hardware to arrive. Most of our progress is blocked by this. Once we have the hardware available, we will be able to begin our first steps - reverse engineering the R/C car, setting up the Jetson, and hooking the two up.

In the meantime, we’ve been doing some research into how we’ll implement lane detection, and setup steps for the Jetson once we get it.

#### Lane Detection 
For lane detection, we researched how to find edges in images. There are algorithms to draw lines on the image where the edges are, and that will be helpful in determining where in the frame the lane lines are. The most concise algorithm involves using Sobel filters on the image, which will then be convolved with the image to get a new image where the edges will be highlighted.

A preprocessing step before doing line detection is a perspective transform on the image to turn the image into a top-down view. This will make the lane edges become parallel with each other, allowing for a more straightforward line to lane conversion. After detecting the lines, we will need to find some sort of regression on the lane lines, since it’s possible that the lines are curved if the car is turning.

Once we can accurately detect the lanes in the image, we will use their placement and curvature on the frame to determine which way the car would need to turn once we have the Jetson and RC car hooked up and drivers written.

#### Jetson Setup
The setup for the Jetson is pretty straight forwards. Installing linux is a 1 line command-line prompt. There is an optional GUI that can be installed, though we will not be doing so. The fastest way of downloading the drivers we need is by running an ubuntu vm and installing JetPack using the IP address of the Jetson. This process also installs CUDA, which will be a fundamental part of optimizing our lane detection algorithms. The Jetson contains only 1 USB 3.0 port, so we will need to use a hub in order to connect more peripherals (if we choose to connect the peripherals via USB). The board does contain a 125 pin expansion slot, which can be used for UART with peripherals, and some of the pins can be used for GPIO, though we will probably only use this for interfacing with the motors, as those will probably have a set of power and ground wires each (we aren’t sure yet since we don’t have the physical RC car). The only issue is there are just 7 GPIO pins, and we may need 8 (depending if there are 2 motor controls + steering control, in which we need 6 or if there is an independent motor per wheel, in which case we would probably need 8). There is a JTAG port which can come in handy when debugging. 

### 10/5/17
We're still waiting on our hardware to arrive, though we know what libraries and languages we will be using, so this week we focused on setting up our personal machines to match those. Below is a brief description of what was done by each member

### Kevin
Used openCV to do a perspective trasnform on an image to get a top down view of an image taken at an anlge. This is an important step in our algorithm as it will make it easier to detect turns and curves in lanes. 

### Anubhav
Installed libraries such as openCV and numPY, and read up on image transformation algorithms to transform the field of view to point straight down at the subject rather than at an angle. 

### Prashant
Used MATLAB to see if edge detection algorithms from a still camera image is a viable option. Is currently finetuning the code to test this hypothesis. 

