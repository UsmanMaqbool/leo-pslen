
% box problem (


%drawRectangle(image, Xmin, Ymin, width, height)
img = drawRectangle(I, bb(2), bb(1), bb(4), bb(3));

direct_and_sort_0.5 : new method jis me 14 tk tabahi results aty thy es dafa just ds_all save raha 
Master: has the previous method to save the feat file. Can take huge space (100MB for each)

why-so-deep (leo-previous)
why-so-deep-3 = full-correlation-check


```
git clone https://github.com/UsmanMaqbool/Maqbool.git

cd Maqbool/ && mkdir 3rd-party-support
git clone https://github.com/vlfeat/matconvnet.git # follow full compilation tutorial below

git clone https://github.com/Relja/netvlad.git
Download the databases file (tokyo247.mat) and set the correct dsetSpecDir in localPaths.m and also add paths.libReljaMatlab 

git clone https://github.com/zchrissirhcz/edges # Not official edges, but fixed error for matlab > 2017
in matlab 'cd edges' and 'run linux_startup.m' and replace the 'edgeBoxes.m' with our edges boxes (to get the edges images as well')
git clone https://github.com/zchrissirhcz/toolbox.git
in matlab 'cd toolbox' and 'run linux_startup.m'

```

#### Install Support (Ubuntu 20.04, Matlab 2019b, Cuda Driver 10.1)

##### matconvnet
git clone https://github.com/vlfeat/matconvnet.git
[Instuctions to install matconvnet](https://www.vlfeat.org/matconvnet/install/)

## MATCOVNET
In centos please use previous version
'wget https://github.com/vlfeat/matconvnet/archive/v1.0-beta18.zip' and 

```matlab
addpath matlab 
%and 
vl_compilenn('enableGpu', true)
```
Also you need to disable the C++11 flag in .matlab/R2017a/mex_C++_glnxa64.xml to C++0x

Run Matlab

```m
cd matconvnet
addpath matlab
vl_compilenn('enableGpu', true) 
```
**Possible Errors**

- if there is NVCC error, try installing Cudatoolkit
`sudo apt install nvidia-cuda-toolkit`
- if there is GCC version issue, try switching using update-alternatives. You can follow the tutorial below
  
`Downgrade to GCC 7/8`

[Useful guide to install specific version of gcc](https://unix.stackexchange.com/questions/410723/how-to-install-a-specific-version-of-gcc-in-kali-linux)

GCC 7 is available on linux it can be installed as follow :
```
sudo apt install g++-7 gcc7 g++-8 gcc8
```    

To switch between gcc7 or gcc8

```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 1 --slave /usr/bin/g++ g++ /usr/bin/g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 2 --slave /usr/bin/g++ g++ /usr/bin/g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 1 --slave /usr/bin/g++ g++ /usr/bin/g++-7
sudo update-alternatives --config gcc

# sample output:

Selection |   Path       |     Priority  | Status
---------|------------------------|----------------|-----------
* 0     |       /usr/bin/gcc-9 |  2     |    auto mode
1       |     /usr/bin/gcc-6  | 2        | manual mode
2       |     /usr/bin/gcc-7  | 1      |   manual mode
      Selection |   Path       |     Priority  | Status

Press <enterto keep the current choice[*], or type selection number: 2
```  