function [FINALFEAT FINALTHRESH featureRanking] = train_classifiers(allFaces, desiredOut, weights)
FINALFEAT = [];
FINALTHRESH = [];
featureRanking = [];
startClock = clock;
for numFeats = 1:100
   
    % so, 200 times, we're going to find a good classifier.  we're going
    % to try a bunch of classifiers each time, so lets' set up a variable
    % to see keep track of the best one we've seen so far
    bestWeakClassifierScore = 0;
    
    
    for jx = 1:20  % boring for loops to always count up!
      
        FEAT = generate_feature;                  % make a random feature.
        scores = allFaces' * FEAT(:);       % compute its score for all faces.
         
        %generate_feature and have it return the corners of positive regions, and corners of negative regions 
        %score of face = (sum up positive - sum of negative regions) 
        
        % now try different thresholds.
        thresholdList = linspace(min(scores),max(scores),1000);  % make 1000 thresholds.
        cScore = 0;    %initialize some stuff about those thresholds.
        cThresh = 0;
        for ix = 1:1000
            % compute classification result with this threshold
            classifierResult = sign(scores-thresholdList(ix));
            
            % compute "weighted" score for each face.
            tmp = classifierResult .* desiredOut .* weights;
            
            % classifier score is the sum of these weighted scores.
            tmpScore = sum(tmp);
            if tmpScore > cScore   % if it is better, set this threshold, score as current best...
                cThresh = thresholdList(ix);
                cScore = tmpScore;
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
    featureRanking(numFeats) = bestWeakClassifierScore;
    
    % ok... so the above loop picked the best of 100 possible features.
    % now, let's update the weights of the samples.
    weights = weights .* exp(-desiredOut .* weakClassifier);
    
    % normalize the weights or they'll go crazy.
    weights = weights./sum(weights(:));
end
startClock =  clock - startClock;
fprintf('finished training %d mins %d secs for #%d faces \r', startClock(5), startClock(6), size(allFaces,2));