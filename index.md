## Blog
### 9/28/17 
We’re still waiting for our hardware to arrive. Most of our progress is blocked by this. Once we have the hardware available, we will be able to begin our first steps - reverse engineering the R/C car, setting up the Jetson, and hooking the two up.

In the meantime, we’ve been doing some research into how we’ll implement lane detection, and setup steps for the Jetson once we get it.

### Lane Detection 
For lane detection, we researched how to find edges in images. There are algorithms to draw lines on the image where the edges are, and that will be helpful in determining where in the frame the lane lines are. The most concise algorithm involves using Sobel filters on the image, which will then be convolved with the image to get a new image where the edges will be highlighted.

A preprocessing step before doing line detection is a perspective transform on the image to turn the image into a top-down view. This will make the lane edges become parallel with each other, allowing for a more straightforward line to lane conversion.

Once we can accurately detect the lanes in the image, we will use their placement on the frame to determine which way the car would need to turn once we have the Jetson and RC car hooked up and drivers written.

### Jetson Setup
The setup for the Jetson is pretty straight forwards. Installing linux is a 1 line command-line prompt. There is an optional GUI that can be installed, though we will not be doing so. The fastest way of downloading the drivers we need is by running an ubuntu vm and installing JetPack using the IP address of the Jetson. This process also installs CUDA, which will be a fundamental part of optimizing our lane detection algorithms. The Jetson contains only 1 USB 3.0 port, so we will need to use a hub in order to connect more peripherals (if we choose to connect the peripherals via USB). 
