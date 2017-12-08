## Blog
## 11/23/17
### Kevin 
Has lane detection roughly working via color filtering. Lane detection through edge detection also works but has too much noise. 

### Prashant
Is working on creating a classifier for a stop sign so the car stops for a stop sign like a normal car would. 

### Anubhav
Is working on creating positive and negative images in order to train a classifier to recognize a frowny face, and have the car stop once that face is detected. 

## 11/16/17

### Prashant
Trying to integrate my polyfit algorithm with the code to binarize the images. However, the code is still running slowly relative to the speed at which we need the calculations, so I am researching possible ways to do incremental transforms using the frames instead of recalculating the perspective each time. 

Also going to look into using live feeds for images in order to integrate the CV side with the data the camera will be sending.

### Anubhav
Looking into other CV algorithms in order to have an alternative for the CV algorithm Prashant is working on. Was busy this week because am leaving today, though I will have plenty of time over break to write a first draft algorithm. The goal will be to finish polyfit and be able to remove noise from the image. My approach will probably be simpler than the one that Prashant is working on.

### Kevin 
I've been reviewing this paper: http://ieeexplore.ieee.org/document/1603550/ to see if we can get any insights into how we can improve the way we're approaching the problem. One of the main insights is to use steerable filters to efficiently extract signals (eg. lane markings) in the image going in certain directions. This will help us filter out some of the extraneous lines and noise from the image. Additionally, we're working on fine-tuning our model of the road, to help make our models more efficient and robust.

### 11/9/17

### Prashant
Working on optimizing the algorithm for still images for now, didnt have much time to work on capstone this week.

### Kevin and Anubhav
Finished the smooth driver for the car so transitions are gradual. Looking into ways to assemble car, ordered batteries and holders in order to mount the Jetson on the RC car. We will be transitioning to help Prashant with the CV side now. 

### 11/2/17

Had midsemester demos on Tuesday. 

### Prashant
Figured out a solid lane detection algorithm using Canny. Give or take some noise, the algorithm is able to find the lane lines and do a polynomial fit on them. The coefficients could then determine how sharp the curvature of the lane ahead is. The only issue with the algorithm at this point is runtime. With about a 0.5 second runtime, it can only use 2 frames a second, which is not at all good enough when it comes to real time analysis of the road. Currently trying to find ways to reduce runtime, such as using smaller images and not having to do the perspective transform for every frame of the video, and instead using deltas to estimate the next frame. 

### Kevin and Anubhav
We managed to get the native PWM working by manipulating the kernel. After that we wrote a driver to control both steering and speed so that our car can completely be controlled via keyboard input to the Jetson. We plan to change this once we know the outputs of the CV algorithm to work with that. Currently we are working on a way to smooth out the transitions in the driver.


### 10/26/27

### Prashant
Working with image transforms and drawing lines from Canny edge detection. The only issue that seems to arise based on this transform is that the lines become somewhat blurry, and these are the most important parts of the vision (in terms of being able to follow the lane). As of right now, I’m trying to code an effective line detector that will only focus on the lane lines and nothing else.
![Output from transform](https://user-images.githubusercontent.com/25559078/32071143-5fea58ee-ba5c-11e7-8878-a8b7be78662e.png)

### Kevin and Anubhav
Tried using the built in PWM on the Jetson, thought that did not work, leading us to update the OS and drivers on the device. The built in PWM still does not work, so we are in the process of writing C drivers based on the system clock. We are also going to try using an Arduino for PWM as that might be more accurate and simpler. (The jetson will compute what values to send, and send them to the arduino which will be in charge of outputting the correct signal). 

### 10/19/17
#### Issues
We are having a difficult time actually connecting to our Jetson since we are unable to SSH into it as previously planned because CMU NetReg is unaware of the ethernet ports in HH 1307, so we cannot obtain a static IP and register our device. A workaround we are trying is by ordering an HDMI cable and a USB hub so we can directly interface with the Jetson. This is not a permanent solution, however, since it limits the number of people working on it to one at a time, and the user needs to have physical control of the Jetson to run any code.

### Kevin
Haven’t been able to do much this week due to other commitments. I spent some time refining the lane detection transformation to yield more consistent images, and collected some images from the camera we have to test on. I also researched some next steps in the processing pipeline to go from the lanes detected to steering instructions.

Because it’s hard to collect (image, steering direction) training data by ourselves at this point (since our car isn’t hooked up to the Jetson yet), it seems like a reasonable next step would be to use some [public data](https://github.com/udacity/self-driving-car/tree/master/datasets) to test a few models before we get our own data.

### Anubhav 
Started writing a driver for the forwards/reverse motor to output a given duty cycle from the GPIO on the Jetson. Looked into ways of interfacing with the Jetson after trying to establish a static connection via ethernet. Is going to look into whom to talk to in order to fix the issue on NetReg and register the device. Is staying on campus for midsemester break so will have time to work on teh driver. 

### Prashant
Been doing some research on generalized hough transforms, regular hough transforms only allow for straight line edge detection. There are no implementations in python for it, but there is a c++ algorithm for it. Possibly can engineer a python version for it. Generalized hough transforms are useful for finding contours in images, which allows the car to find curves in a road. 
Looking through this paper for clues: http://homepages.inf.ed.ac.uk/rbf/BOOKS/BANDB/LIB/bandb4_3.pdf



### 10/12/17
We received our parts on Tuesday and are getting familiar with them. 

Kevin looked into the RC car and discovered the following about how the controllers works

#### Radio controller. 
This controller accepts radio signals from the remote control, and emits PWM encoded signals for the speed and steering controllers. It’s powered by DC input from the speed controller, and supplies power for the steering controller.

#### Speed controller. 
This controller is connected to the main drive motor, and the battery of the car. It accepts a PWM signal from the radio controller, which determines in what direction and how fast the motor should spin. It also downsteps the battery voltage, and supplies DC power to the radio controller, which is subsequently used by the steering controller.

#### Steering controller. 
This controller is connected to the steering motor of the car. It accepts a PWM signal and DC power from the radio controller, which powers and determines the position it should place the steering motor in.

#### Next steps. 
We now need to reverse engineer the radio controller component into software that can be run by the Jetson, so that we can control the car via software. This should be fairly straightforward, as the control lines for the speed and steering controllers are just simple PWM signals.
### Kevin
I took apart the RC car, and poked around its internals. The car is composed of three main circuits: A radio receiver, a speed controller, and a steering controller.

### Anubhav 
Had a heavy workload this week, so was unable to put as much effort as hoped. Configured local environment to match that on the jetson and researched how to write the necessary drivers to connect to the car. Will be coming in Friday to work some more. 

### Prashant
Looked into CV algorithms in order to calculate the probability of which direction the car is turning based on camera input. 

### 10/5/17
We're still waiting on our hardware to arrive, though we know what libraries and languages we will be using, so this week we focused on setting up our personal machines to match those. Below is a brief description of what was done by each member. We focused primarily on implementing an image transformation algorithm like the one below in order to process the images received from our camera. 

![example transform](https://user-images.githubusercontent.com/25559078/31245676-1da2ba14-a9d9-11e7-83f1-f762091515bb.png)


### Kevin
Used openCV to do a perspective trasnform on an image to get a top down view of an image taken at an anlge. This is an important step in our algorithm as it will make it easier to detect turns and curves in lanes. 

### Anubhav
Installed libraries such as openCV and numPY, and read up on image transformation algorithms to transform the field of view to point straight down at the subject rather than at an angle. 

### Prashant
Used MATLAB to see if edge detection algorithms from a still camera image is a viable option. Is currently finetuning the code to test this hypothesis. 



### 9/28/17 
We’re still waiting for our hardware to arrive. Most of our progress is blocked by this. Once we have the hardware available, we will be able to begin our first steps - reverse engineering the R/C car, setting up the Jetson, and hooking the two up.

In the meantime, we’ve been doing some research into how we’ll implement lane detection, and setup steps for the Jetson once we get it.

#### Lane Detection 
For lane detection, we researched how to find edges in images. There are algorithms to draw lines on the image where the edges are, and that will be helpful in determining where in the frame the lane lines are. The most concise algorithm involves using Sobel filters on the image, which will then be convolved with the image to get a new image where the edges will be highlighted.

A preprocessing step before doing line detection is a perspective transform on the image to turn the image into a top-down view. This will make the lane edges become parallel with each other, allowing for a more straightforward line to lane conversion. After detecting the lines, we will need to find some sort of regression on the lane lines, since it’s possible that the lines are curved if the car is turning.

Once we can accurately detect the lanes in the image, we will use their placement and curvature on the frame to determine which way the car would need to turn once we have the Jetson and RC car hooked up and drivers written.

#### Jetson Setup
The setup for the Jetson is pretty straight forwards. Installing linux is a 1 line command-line prompt. There is an optional GUI that can be installed, though we will not be doing so. The fastest way of downloading the drivers we need is by running an ubuntu vm and installing JetPack using the IP address of the Jetson. This process also installs CUDA, which will be a fundamental part of optimizing our lane detection algorithms. The Jetson contains only 1 USB 3.0 port, so we will need to use a hub in order to connect more peripherals (if we choose to connect the peripherals via USB). The board does contain a 125 pin expansion slot, which can be used for UART with peripherals, and some of the pins can be used for GPIO, though we will probably only use this for interfacing with the motors, as those will probably have a set of power and ground wires each (we aren’t sure yet since we don’t have the physical RC car). The only issue is there are just 7 GPIO pins, and we may need 8 (depending if there are 2 motor controls + steering control, in which we need 6 or if there is an independent motor per wheel, in which case we would probably need 8). There is a JTAG port which can come in handy when debugging. 

