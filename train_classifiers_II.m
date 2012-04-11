function [FINALFEAT_II FINALTHRESH featureRanking] = train_classifiers_II(allFaces, desiredOut, weights, numFaces, numNonFaces)
% initial some variables
FINALFEAT_II = zeros(2,8,100);
FINALTHRESH = zeros(1,100);
featureRanking = zeros(1,100);
startClock = clock;
for numFeats = 1:100
    bestWeakClassifierScore = 0;    
    for jx = 1:20  
      
        [POSITIVE NEGATIVE] = gen_interval_feature;         
        %generate_feature and have it return the corners of positive regions, and corners of negative regions 
        %score of face = (sum up positive - sum of negative regions) 
        
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
            FINALFEAT_II(1, 1:size(POSITIVE,1)*size(POSITIVE,2), numFeats) = POSITIVE(:);
            FINALFEAT_II(2, 1:size(NEGATIVE,1)*size(NEGATIVE,2), numFeats) = NEGATIVE(:);
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
fprintf('finished training %d mins %d secs for #%d faces \r', startClock(5), startClock(6), size(allFaces,3));