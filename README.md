Manga Line Extraction
--------------

## erika_unstable.h5(weight)

To download the file please follow this link:
https://www.dropbox.com/s/m1f5n8xbrpkl37r/erika_unstable.h5?dl=0
Please use this file as the pre-trained weight for testing.


## Requirement

+  Python3
+  keras==1.2.0 (1.2.2 is okay too)
+  theano==0.9
+  python-opencv
+  h5py
+  cuda is optional but highly recommended

It's strongly recommended to use virtualenv to build the testing environment.


## How To Setup
### python libraries
```bash
conda install pygpu=0.6
conda install theano=0.9
conda install h5py
pip installl opencv-python
pip install keras==1.2.2
```

### environment
Set cuda8.0_cudnn5.1 on $HOME/.local/cuda8.0_cudnn5.1

```bash
$ source envrc
```

## Forward Compatibility

As this project is build against theano, you need to modify the dim orderding into "th"/"channels_first" in your keras.json.

For details of testing with latest version of tensorflow, please refer to #1 .

## Usage

        demo_mse.py [source folder] [output folder]

Example:

        demo_mse.py ./Arisa ./output

The outputs will be clipped to 0-255.




For more details please take a look at the source files. Thanks.
