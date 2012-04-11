# Option #1

_______

## Overview

###Training
Viola-Jones face detection is a machine learning technique that generates a set of features that are useful in identify faces.  These "useful" features are found by measuring of the response a set of postive and negative rectangles within a 24x24 pixel square.  A simplistic approach was taken in determining which rectangles should comprise the feature, rather than try out all 180,000+ possible features.  A rectangle type was chosen from five types of rectangles generating the rectangle, as shown below.

 ![Rectangles](https://github.com/KnownSubset/CSE559-facedetector/raw/master/rectangles.jpg "Rectangles") There also is a rotated version of the three part feature calculated and not all parts of the features will have the same dimensions as the othe parts.

In the training phase, using the generated rectangle, I determined its response against all of the faces and nonfaces.  The more images that it correctly identified will give it a higher score.  After the training phase determines the features that have the best scores, I combined them into a single feature.

Here are some examples of the best classifiers:

Here are some examples of the worst classifiers:

###Classification
Once all the features have been generated from the training phase, these features can be ran against any image to detect the faces.  Each 24x24 square of the image is ran against the set of features, just as all the faces were during the training phase.  A problem occurs when the same face appears in multiple rectangle as demonstrated with these images:

  ![face1](https://github.com/KnownSubset/CSE559-facedetector/raw/master/face1.jpg "bad face") ![face2](https://github.com/KnownSubset/CSE559-facedetector/raw/master/face2.jpg "good face") ![face3](https://github.com/KnownSubset/CSE559-facedetector/raw/master/face3.jpg "bad face")

To mitigate this factor, the maximum response from within a local area is usually determined to be the face.  It also helps to have really good features that will not pick up half a face as being a face.


    Re-train the classifier without the last 100 example faces and without the last 100 example non-faces, then use those 200 examples as "test-cases", and report classification accuracy (False Positive, True Positive, False Negative, and True Negative percentages).
    Report on total running time of both the training phase and the per-image testing phase.
    Report on running time when using the integral images, versus not using the integral images. 

* Pseudo-code description of the algorithm, highlighting things like specifics of your image pyramid (how much smaller is each layer than the next).
    

    ```matlab
    %faces & nonfaces are already available


    %Declare Success
    %Map images onto a surface
    ```
