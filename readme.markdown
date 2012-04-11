# Option #1

_______

## Overview
 Viola-Jones face detection is a machine learning technique that generates a set of features that are useful in identify faces.  These "useful" features are found by measuring of the response a set of postive and negative rectangles within a 24x24 pixel square.  A simplistic approach was taken in determining which rectangles should comprise the feature, rather than try out all 180,000+ possible features.  A rectangle type was chosen from five types of rectangles generating the rectangle, as shown below.

 ![Rectangles](https://github.com/KnownSubset/CSE559-facedetector/raw/master/rectangles_types.jpg "Rectangles") There also is a rotated version of the three part feature calculated and not all parts of the features will have the same dimensions as the othe parts.

##Training

In the training phase, using the generated rectangle, I determined its response against all of the faces and nonfaces.  The more images that it correctly identified will give it a higher score.  After the training phase determines the features that have the best scores, I combined them into a single feature.

There are more noval approaches to learning classifier than the approach I took to learning the classifier.  If I had more time I would go back and correctly implement the Ada-Boost that was metioned in the paper, as I would expect to generated better classifiers from these learning functions.

Here are some examples of the best classifiers:

     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 + + + - - - 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 + + + - - - 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 + + + - - - 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 + + + - - - 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 + + + - - - 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 + + + - - - 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 - - - + + + 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 - - - + + + 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 - - - + + + 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 - - - + + + 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 - - - + + + 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 - - - + + + 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 + + + + 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 + + + + 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 + + + + 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 + + + + 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 + + + + 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 + + + + 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 + + + + 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 - - - - 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 - - - - 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 - - - - 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 - - - - 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 - - - - 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 - - - - 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 - - - - 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

Here are some examples of the worst classifiers:

     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 + + + + + + - - - - - - 0 0 0 0 0 0 0 0 0
     0 0 0 + + + + + + - - - - - - 0 0 0 0 0 0 0 0 0
     0 0 0 + + + + + + - - - - - - 0 0 0 0 0 0 0 0 0
     0 0 0 + + + + + + - - - - - - 0 0 0 0 0 0 0 0 0
     0 0 0 + + + + + + - - - - - - 0 0 0 0 0 0 0 0 0
     0 0 0 + + + + + + - - - - - - 0 0 0 0 0 0 0 0 0
     0 0 0 + + + + + + - - - - - - 0 0 0 0 0 0 0 0 0
     0 0 0 + + + + + + - - - - - - 0 0 0 0 0 0 0 0 0


     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 - - - - - - - - - - - - - -
     0 0 0 0 0 0 0 0 0 0 - - - - - - - - - - - - - -
     0 0 0 0 0 0 0 0 0 0 - - - - - - - - - - - - - -
     0 0 0 0 0 0 0 0 0 0 - - - - - - - - - - - - - -
     0 0 0 0 0 0 0 0 0 0 - - - - - - - - - - - - - -
     0 0 0 0 0 0 0 0 0 0 + + + + + + + + + + + + + +
     0 0 0 0 0 0 0 0 0 0 + + + + + + + + + + + + + +
     0 0 0 0 0 0 0 0 0 0 + + + + + + + + + + + + + +
     0 0 0 0 0 0 0 0 0 0 + + + + + + + + + + + + + +
     0 0 0 0 0 0 0 0 0 0 + + + + + + + + + + + + + +
     0 0 0 0 0 0 0 0 0 0 - - - - - - - - - - - - - -
     0 0 0 0 0 0 0 0 0 0 - - - - - - - - - - - - - -
     0 0 0 0 0 0 0 0 0 0 - - - - - - - - - - - - - -
     0 0 0 0 0 0 0 0 0 0 - - - - - - - - - - - - - -
     0 0 0 0 0 0 0 0 0 0 - - - - - - - - - - - - - -
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

* Training Pseudo-code

    ```matlab
    %faces & nonfaces are already available
    FINALFEAT = [];
    FINALTHRESH = [];
    for numFeats = 1:100
        for jx = 1:20
            FEAT = generate_feature;                  % make a random feature.
            scores = allFaces' * FEAT(:);             % compute its score for all faces.
            thresholdList = linspace(min(scores),max(scores),1000);  % make 1000 thresholds.
            for ix = 1:1000
                % compute classification result with this threshold
                classifierResult = sign(scores-thresholdList(ix));
                % compute "weighted" score for each face (classifier score is the sum of these weighted scores)
                tempScorce = sum(classifierResult .* desiredClassification .* weights);
                if (better than pervious score)
                    cThresh = thresholdList(ix);
                end
            end
            % go back and get classification for the best threshold you found.
            weakClassifier = sign(scores-cThresh);
            % recompute how good that actually was.
            weakClassifierScore = sum(weakClassifier .* desiredOut .*weights);
            % if better than we've seen so far, then save it.
            if weakClassifierScore > bestWeakClassifierScore
                bestWeakClassifierScore = weakClassifierScore;
                FINALFEAT(:,:,numFeats) = FEAT;
                FINALTHRESH(numFeats) = cThresh;
            end
        end
        % now, let's update the weights of the samples.
        weights = weights .* exp(-desiredClassification .* weakClassifier);
        % normalize the weights or they'll go crazy.
        weights = weights./sum(weights(:));
    end
    ```

### Results from a sample training run


    Non integral images

    finished training 3 mins -2.993770e+01 secs for #12876 faces
    true positive: 8.303499e-01 %  or 4082 out of 4916 faces
    true negative: 9.488693e-01 %  or 7553 out of 7960 faces
    false negatives: 1.696501e-01 %  or 834 out of 4916 faces
    false positives: 5.113065e-02 %  or 407 out of 7960 faces

    finished training 3 mins -2.993770e+01 secs for #12876 faces
    true positive: 8.303499e-01 %  or 4082 out of 4916 faces
    true negative: 9.488693e-01 %  or 7553 out of 7960 faces
    false negatives: 1.696501e-01 %  or 834 out of 4916 faces
    false positives: 5.113065e-02 %  or 407 out of 7960 faces
    total running time ???

 - - -

    Integral images

    time to calculate integral image on training data : 0.1074
    finished training 2 mins 2.313361e+01 secs for #12876 faces
    true positive: 8.567941e-01 % or 4212 out of 4916
    true negative: 6.432161e-01 % or 5120 out of 7960
    false negatives: 1.432059e-01 % or 704 out of 4916
    false positives: 3.567839e-01 % or 2840 out of 7960

    finished training 3 mins -4.012887e+01 secs for #12676 faces
    true positive: 2.600000e-01 % or 26 out of 100
    true negative: 9.100000e-01 % or 91 out of 100
    false negatives: 7.400000e-01 % or 74 out of 100
    false positives: 9.000000e-02 % or 9 out of 100
    total running time ???

As we can see that the total running time for the integral images is actually higher than the running time not using integral images. This is due to matlab being heavily optimized for matrix operations and the integral operations having iterate through the sets of points that comprised the positive and negative regions as demonstrated through this psuedo-code.


   ```matlab
    %generate_feature and have it return the corners of positive regions, and corners of negative regions
    %score of face = (sum up positive - sum of negative regions)
    [POSITIVE NEGATIVE] = gen_interval_feature;
    scores = zeros(1, 1, numFaces + numNonFaces);
    for px = 1 : size(POSITIVE,1)
        points = POSITIVE(px, 1:4);
        row1 = points(1);
        row2 = points(3);
        col1 = points(2);
        col2 = points(4);
        scores = scores + (allFaces(row1, col1 ,:) + allFaces(row2, col2 ,:) - allFaces(row1, col2, :) - allFaces(row2, col1, :) );
    end
    for px = 1 : size(NEGATIVE,1)
        points = NEGATIVE(px, 1:4);
        row1 = points(1);
        row2 = points(3);
        col1 = points(2);
        col2 = points(4);
        scores = scores - (allFaces(row1, col1 ,:) + allFaces(row2, col2 ,:) - allFaces(row1, col2, :) - allFaces(row2, col1, :)) ;
    end
    scores = reshape(scores,numFaces+numNonFaces,1);
   ```

##Classification
Once all the features have been generated from the training phase, these features can be ran against any image to detect the faces.  Each 24x24 square of the image is ran against the set of features, just as all the faces were during the training phase.  A problem exists that the same face appears in multiple rectangle as demonstrated with these images:

  ![face1](https://github.com/KnownSubset/CSE559-facedetector/raw/master/face1.jpg "bad face") ![face2](https://github.com/KnownSubset/CSE559-facedetector/raw/master/face2.jpg "good face") ![face3](https://github.com/KnownSubset/CSE559-facedetector/raw/master/face3.jpg "bad face")

It's quite likely that the same face will be detected several times for multiple levels of resolution.  To help mitigate this factor, the maximum response from within a local area is usually determined to be the face.  It also helps to have really good features that will not pick up half a face as being a face.   To calculate the local maximum I used this [code](http://stackoverflow.com/questions/1856197/how-can-i-find-local-maxima-in-an-image-in-matlab), it works reasonably well as you see from the results below.  It would generate better results if it would work within a threshold as it is still easy to recognize that multiple subsquares that overlap should be merged as a possible face. If more time was available, I code try to implement the approach listed in the paper for "Integration of Multiple Detections" by merging overlapping regions into a single detection.
Then these steps are repeated for the image pyramid, until the next image cannot contain a 24x24 pixel feature.


* Classification pseudo-code

    ```matlab
    %% display training results
    FF = reshape(FINALFEAT,24*24,[]);                   % Reshape all the good features into one matrix
    scores = FF'*allFaces;                              % Compute the score of every face with every feature.
    thresholds = repmat(FINALTHRESH',1,size(AS,2));     % create matrix of all thresholds, replicating it so its same size as AS
    VOTES = sign( scores - thresholds);                 % compute weak classification  of all faces for all features
    CLASSIFICATION = sign(sum(VOTES));                  % sum the classifications.
    ```

### Cascade Filters

The idea of using cascade filters is to help quickly reduce the search space by applying a subset of filters.  The subsquares of an image that are labeled as faces are then passed on to the next set of filters.  The process repeats until all filters have been processed. ![cascade](https://github.com/KnownSubset/CSE559-facedetector/raw/master/cascade_filter.jpg "cascade")

To determine which filters to apply first, I sorted the filters based upon its bestClassifierScore that was generated during the training phase.  I tried various schemes of how to apply the filters.

### Results from a sample classification run


    image size  |   time    | time using cascade
    221 250     |   1.1370  |   1.7805
    177 200     |   0.7513  |   1.3505
    142 160     |   0.4657  |   0.8743
    114 128     |   0.2546  |   0.5496
    92 103      |   0.1624  |   0.3339
    74 83       |   0.0878  |   0.2210
    60 67       |   0.0511  |   0.1128
    48 54       |   0.0247  |   0.0465
    39 44       |   0.0112  |   0.0185
    32 36       |   0.0034  |   0.0075
    26 29       |   0.000941|   0.0052

total time to classify image pyramids squares: 9.2413

I used impyramid that 80% smaller than the next layer

###Integral Images
 ![Integral Image](https://github.com/KnownSubset/CSE559-facedetector/raw/master/integral_image_example.jpg "Integral Image")
Integral areas (or summed area tables) are really useful for in the calculation because you can calculate the response of image to feature using four calculations for every subsquare, instead of 24x24 operations for every subsquare.  However I did experience a set back with this as I during the responses for every subsquare within a image.  I was doing each subsquares calculation seperatly and was befuddled as to why I was not seeing similar or better performance than the original method.  Then I finally realized that I could perform the calculation for all subsquares at the same time.  This was a lesson well learned from using matlab, that operations are faster on array then on each individual element of the array.

Another nice part of the integral image is that it is not neccessary to calculate the image pyramid to find "larger faces" than at 24x24 pixels.  Due to the fact that integral image is already calculated it is just as effecient to upsize the features, since it will still only require four operations to calculate the response of subsquare to a feature.  As an aside, I did not upsize the features, and I am sure if I had more time to implement the functionality it would lead to a performance boost the integral images of the image pyramid would not have to be calculated.



## Results
There are two functions that will run to train the classifiers, report on the accuracy of the classifiers, and run the classifiers against an image.

[face_detector.m](https://github.com/KnownSubset/CSE559-FaceDetector/blob/master/face_detector.m) will run without using integral images.
[face_detector_II.m](https://github.com/KnownSubset/CSE559-FaceDetector/blob/master/face_detector_II.m) will run using integral images.

Both of matlab functions use a mix of other functions contained within the same repository, some of which I am surprised work all together.

 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/data/lotr_cast1.jpg "lotr 441x500")

 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_cascade_441_500.jpg "lotr 441x500")
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_noncascade_441_500.jpg "lotr 441x500")
 - - -
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_cascade_375_425.jpg "lotr 441x500")
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_noncascade_375_425.jpg "lotr 441x500")
 - - -
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_cascade_319_362.jpg "lotr 441x500")
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_noncascade_319_362.jpg "lotr 441x500")
 - - -
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_cascade_272_308.jpg "lotr 441x500")
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_noncascade_272_308.jpg "lotr 441x500")
 - - -
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_cascade_232_262.jpg "lotr 441x500")
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_noncascade_232_262.jpg "lotr 441x500")
 - - -
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_cascade_198_223.jpg "lotr 441x500")
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_noncascade_198_223.jpg "lotr 441x500")
 - - -
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_cascade_169_190.jpg "lotr 441x500")
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_noncascade_169_190.jpg "lotr 441x500")
 - - -
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_cascade_144_162.jpg "lotr 441x500")
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_noncascade_144_162.jpg "lotr 441x500")
 - - -
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_cascade_123_138.jpg "lotr 441x500")
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_noncascade_123_138.jpg "lotr 441x500")
 - - -
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_cascade_105_118.jpg "lotr 441x500")
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_noncascade_105_118.jpg "lotr 441x500")
 - - -
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_cascade_90_101.jpg "lotr 441x500")
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_noncascade_90_101.jpg "lotr 441x500")
 - - -
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_cascade_77_86.jpg "lotr 441x500")
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_noncascade_77_86.jpg "lotr 441x500")
 - - -
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_cascade_66_74.jpg "lotr 441x500")
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_noncascade_66_74.jpg "lotr 441x500")
 - - -
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_cascade_57_63.jpg "lotr 441x500")
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_noncascade_57_63.jpg "lotr 441x500")
 - - -
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_cascade_49_54.jpg "lotr 441x500")
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_noncascade_49_54.jpg "lotr 441x500")
 - - -
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_cascade_42_46.jpg "lotr 441x500")
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_noncascade_42_46.jpg "lotr 441x500")
 - - -
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_cascade_36_40.jpg "lotr 441x500")
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_noncascade_36_40.jpg "lotr 441x500")
 - - -
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_cascade_31_34.jpg "lotr 441x500")
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_noncascade_31_34.jpg "lotr 441x500")
 - - -
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_cascade_27_29.jpg "lotr 441x500")
 ![lotr 441x500](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/lotr_cast1_noncascade_27_29.jpg "lotr 441x500")

 = = =

  ![golf ](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/JJsts.jpg "golf")

  ![golf 477_240](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_noncascade_477_240.jpg "golf 477_240")
  ![golf 477_240](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_cascade_477_240.jpg "golf 477_240")
 - - -
  ![golf 382_192](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_noncascade_382_192.jpg "golf 382_192")
  ![golf 382_192](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_cascade_382_192.jpg "golf 382_192")
 - - -
  ![golf 306_154](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_noncascade_306_154.jpg "golf 306_154")
  ![golf 306_154](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_cascade_306_154.jpg "golf 306_154")
 - - -
  ![golf 245_124](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_noncascade_245_124.jpg "golf 245_124")
  ![golf 245_124](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_cascade_245_124.jpg "golf 245_124")
 - - -
  ![golf 196_100](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_noncascade_196_100.jpg "golf 196_100")
  ![golf 196_100](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_cascade_196_100.jpg "golf 196_100")
 - - -
  ![golf 157_80](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_noncascade_157_80.jpg "golf 157_80")
  ![golf 157_80](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_cascade_157_80.jpg "golf 157_80")
 - - -
  ![golf 126_64](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_noncascade_126_64.jpg "golf 126_64")
  ![golf 126_64](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_cascade_126_64.jpg "golf 126_64")
 - - -
  ![golf 101_52](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_noncascade_101_52.jpg "golf 101_52")
  ![golf 101_52](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_cascade_101_52.jpg "golf 101_52")
 - - -
  ![golf 81_42](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_noncascade_81_42.jpg "golf 81_42")
  ![golf 81_42](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_cascade_81_42.jpg "golf 81_42")
 - - -
  ![golf 65_34](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_noncascade_65_34.jpg "golf 65_34")
  ![golf 65_34](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_cascade_65_34.jpg "golf 65_34")
 - - -
  ![golf 52_28](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_noncascade_52_28.jpg "golf 52_28")
  ![golf 52_28](https://github.com/KnownSubset/CSE559-facedetector/raw/master/images/golf_cascade_52_28.jpg "golf 52_28")

