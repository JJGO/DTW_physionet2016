# Heart Sound Classification based on Temporal Alignment Techniques

This is the code repository for our Dynamic Time Warping based classifier submitted to the Physionet Challenge 2016. The description of the algorithm and the results obtained can be in associated [paper](https://github.com/JJGO/DTW_physionet2016/blob/master/GonzalezPerngWiens_CinC2016.pdf)

The code includes Python scripts for:

* Computing intrinsic DTW features for the RR intervals of a PCG recording. Features can also be extracted for individual heart sounds.
* Clustering recordings based on DTW distance and deriving interDTW features that measure the similarity between two PCG recordings.
* Obtaining MFCC features specially8 tuned for Phonocardiograms.
* Training sklearn models from the features described above as well as other ones such as wavelets
* Custom cross validation for unbalanced databases with potentially different sizes recording databases
* Exhaustive and customizable experiment design suite with training, validation and testing
* Caching of most results and feature sets to speed up model training and testing.

The Physionet 2016 dataset can be found [here](http://physionet.org/challenge/2016) and the code used for the PCG segmentation [here](http://physionet.org/physiotools/hss)

## Installation

The package requires Python >=3.3 since it uses the [multiprocessing](https://docs.python.org/3.5/library/multiprocessing.html) package to release the GIL.

The implementation also requires `Cython` so if you do not have it installed you will need to install it

```sh
pip install Cython
```

To compile the dependencies you will have to run the following commands

```sh
cd utils/dtwpy
python setup_dtw.py build_ext --inplace
```

## How to cite
Authors of scientific papers including results generated using the code provided here are encouraged to cite the paper.

```xml
@inproceedings{ortiz2016cinc,
  title={Heart Sound Classification based on Temporal Alignment Techniques},
  author={ Gonz\'alez Ortiz, Jos\'e Javier and Perng Phoo, Cheng and Wiens, Jenna},
  booktitle={2016 43th Computing in Cardiology Conference, CinC 2016},
  year={2016}
}
```

## Credit

* Dataset from the Physionet 2016 [competition](http://physionet.org/challenge/2016/)

* Segmentation algorithm thanks to [David Springer](http://physionet.org/physiotools/hss/)

* MFCC feature implementation thanks to [James Lyons](https://github.com/jameslyons/python_speech_features)

* Low level C implementation of DTW thanks to [mlpy](http://mlpy.sourceforge.net/)

* SVC model implementation thanks to [sklearn](http://scikit-learn.org/)