# Jetson TK1 Kernel Notes

## Overview

The kernel tree given here is based off of Nvidia's r21.6 Linux4Tegra release,
Linux kernel version 3.10. It contains a few modifications to get things
working exactly the way we want them.

The modifications:

  - Make the internal PWM chip and pins work.
  - Make the PS3 Eye camera work.
  - Make the TP-Link TL-WN725N USB WiFi adapter work.

## Installation

First, you'll need to flash the base Nvidia r21.6 kernel. You can find steps to
do that in [Nvidia's Quick Start
Guide](https://developer.nvidia.com/linux-tegra-r216). This will roughly
involve downloading the L4T Driver package on the host, plugging the USB cable
into the Jetson, and running a few commands. Note that if you use a virtual
machine to do the USB flashing, make sure you set the VM's to use a USB2.0
device, otherwise the Jetson won't accept the flash.

Next, boot up the Jetson, and copy the `jetson-kernel` directory of this kernel
to `/usr/src/kernel` on the Jetson.

```
git clone <Autonomobile.git>
cp -r Autonomobile/jetson-kernel /usr/src/kernel
```

Next, install dependencies.

```
sudo apt-add-repository universe
sudo apt-get update
sudo apt-get install libncurses5-dev
```

Next, configure the kernel.

```
cd /usr/src/kernel
zcat /proc/config.gz > .config
make menuconfig
```

Select the following:

  - "General setup -> Local version": Make a suffix, like `-ece500`.
  - "General setup -> Automatically append version information to the version
    string": Exclude. This will help prevent drivers from complaining about
    non-matching magic version strings.
  - "Device drivers -> Multimedia support -> Media USB Adapters -> GSPCA based
    webcams -> OV534 OV772x USB Camera Driver": Modularize. This is for PS3 Eye
    support.
  - "Device drivers -> Staging drivers -> Realtek 8188E USB WiFi": Modularize.
    This is for TL-WN725Nv2 USB WiFi adapter support.

Next, compile the kernel. This will take a few minutes, and generate a kernel
image and device trees.

```
make prepare
make -j4
```

Next, compile and install the modules. This will compile and copy over all the
LKMs that need to go in `/lib/modules`, including device the drivers.

```
make modules_prepare
make modules -j4
make modules_install
```

Next, install the kernel. All you need to do is copy the compiled kernel image,
`arch/arm/boot/zImage`, from the kernel tree, into the Jetson's `/boot`
directory.

```
cp /usr/src/kernel/arch/arm/boot/zImage /boot`
```

Reboot, and check that the new kernel is running.

```
uname -a
```

should give you a suffix of what you set in `make menuconfig`.

Note that the PWM fix might not work properly at first. For some reason, the
device tree might get compiled into `arch/arm/boot/dts` (note the `dts`
subdirectory), but copying the device tree over manually should fix this:

```
cp /usr/src/kernel/arch/arm/boot/dts/tegra124-jetson_tk1-pm375-000-c00-00.dts /boot
```

If the WiFi gives you trouble, make sure that the firmware is installed:

```
apt-get install linux-firmware
```

## Kernel modification details

### PWM Fix

The Jetson's PWM pins don't work out of the box, since the PWM chip isn't hooked
up properly in the device tree. Commit `98ea11` in this repository contains the
device tree patch to make them work. Recompiling and reinstalling the device
tree is enough to make the pins work.

### PS3 Eye

The stock kernel doesn't include the driver for the PS3 Eye, but the driver does
exist in the stock kernel tree. Therefore, it needs to be enabled as a module
(as described in the installation steps above), and installed into
`/lib/modules`.

### TP-Link TL-WN725Nv2 USB WiFi Driver

The stock kernel is too old (Linux 3.10), and the driver for this USB adapter,
`RTL8188EU`, doesn't exist in the source tree yet. Therefore, we need to
manually add it ourselves. Commit `4fc803` adds the vendor driver (downloaded
from the TP-Link website) into the appropriate place in the source tree, and
commit `55a7c8` fixes a bug in the driver. The driver then needs to be
configured as a kernel module (as described above), compiled, and installed into
`/lib/modules`.

Note that the `RTL8188EU` driver available on Github doesn't work (doesn't
compile properly), the `.ko` file from the Grinch kernel doesn't work (refuses
to load), and none of the `staging/rtl8188eu` drivers from the mainline kernel
work either (newer ones don't compile due, older ones segfault). The only driver
that seems to work is the one distributed by TP-Link, which also contains a bug
we had to fix.
