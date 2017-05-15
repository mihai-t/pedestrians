# Pedestrian detection 

- convolutional networks with sliding window approach
- [deeplearning4j](https://deeplearning4j.org/) implementation
- experiments on [Daimler benchmark dataset](http://www.lookingatpeople.com/Datasets/Daimler_Pedestrian_Benchmark_D/Daimler_Stereo_Ped__Detection_/daimler_stereo_ped__detection_.html)

## Dependencies and configuration

- Java 8 solution ([jdk 1.8](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html))
- Apache [maven 3](https://maven.apache.org/)
- The _ML_ solution is based on [deeplearning4j](https://deeplearning4j.org/)
- Java IDE ([Intellij Idea Community/Ultimate](https://www.jetbrains.com/idea/) recommended)
- GPU integration using [CUDA 8](https://developer.nvidia.com/cuda-downloads)

## Approach

- Convolutional neural networks (_CNN_) 
- This neural network architecture is fully supported by [deeplearning4j](https://deeplearning4j.org/convolutionalnets)

## Data representation

- The _CNN_ model is meant to classify 48x96 windows into 2 classes: pedestrian vs non pedestrian

### Network architecture

- several convolutional architectures are implemented

## Further improvements

- Test different evaluation measures
- Reduce the false positive rate
- Improve the evaluation execution speed
- Data augmentation
- Non-Maximum Suppression algorithm
