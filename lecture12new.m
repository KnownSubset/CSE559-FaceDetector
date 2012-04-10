

%% make viola jones features
FEAT = generate_feature;   % look inside this function.  This returns a 24 x 24 image of values 0,-1,1.
subplot(1,2,1);
imagesc(FEAT);  %look at the feature.
% lets compute the score of some features:
subplot(1,2,2);
Fvec = reshape(faces,24*24,[]);
NFvec = reshape(nonfaces,24*24,[]);

scores = Fvec' * FEAT(:);    % compute the feature response

% show histograms.
[counts bins] = hist(Fvec' * FEAT(:),100);
[counts2 bins2] = hist(NFvec' * FEAT(:),100);
plot(bins,counts,'r');
hold on;
plot(bins2,counts2,'b');
hold off;


%% sweet!  now let's do Robert's crummy but intuitive boosting...
allFaces = [Fvec NFvec];
numFaces = size(Fvec,2);
numNonFaces = size(NFvec,2);
desiredOut = [ones(1,size(Fvec,2)) -ones(1,size(NFvec,2))]';
% make the total weight of faces and non faces the same (so that just
% calling everything "not a face" isn't a win...
weights = [numNonFaces.*ones(1,size(Fvec,2)) numFaces.*ones(1,size(NFvec,2))]';
weights = weights./sum(weights(:));
%% now, make 20 Features
% initial some variables
FINALFEAT = [];
FINALTHRESH = [];
bests = [];
clock
for numFeats = 1:100
   
    % so, 200 times, we're going to find a good classifier.  we're going
    % to try a bunch of classifiers each time, so lets' set up a variable
    % to see keep track of the best one we've seen so far
    bestWeakClassifierScore = 0;
    
    
    for jx = 1:10  % boring for loops to always count up!
      
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
    bests(numFeats) = bestWeakClassifierScore;
    
    % ok... so the above loop picked the best of 100 possible features.
    % now, let's update the weights of the samples.
    weights = weights .* exp(-desiredOut .* weakClassifier);
    
    % normalize the weights or they'll go crazy.
    weights = weights./sum(weights(:));
end
clock
%%
[y i] = sort(bests,2,'descend');
FF = reshape(FINALFEAT,576,[]);         % Reshape all the good features into one matrix
AS = FF'*allFaces;                      % Compute the score of every face with every feature.
AT = repmat(FINALTHRESH',1,size(AS,2)); % create matrix of all thresholds, replicating it so its same size as AS
VOTES = sign( AS - AT);                 % compute weak classification  of all faces for all features
CLASSIFICATION = sign(sum(VOTES)-eps);  % sum the classifications.  if something has EXACTLY the same number of 
                                        % yes and no votes, then it is 0
                                        % instead of -1 of +1, so -eps
                                        % makes sure that doesn't happen.
                                        
                                        
disp('Number of correctly labelled faces: ');
sum(CLASSIFICATION' == 1 &  desiredOut == 1)
disp('Number of correctly labelled nonfaces: ');
sum(CLASSIFICATION' == -1 &  desiredOut == -1)

disp('Number of false negatives: ');
sum(CLASSIFICATION' == -1 &  desiredOut == 1)
disp('Number of false positives: ');
sum(CLASSIFICATION' == 1 &  desiredOut == -1)

%% classify an image
image = rgb2gray(imread('/Users/nathan/Development/CSE559/Project3/data/lotr_cast1.jpg'));
%%LOTR cast
%image2 = rgb2gray(imread('/Users/nathan/Development/CSE559/Project3/data/lotr_cast2.jpg'));
%image3 = rgb2gray(imread('/Users/nathan/Development/CSE559/Project3/data/princess_bride.jpg'));
%%LOTR cast
image2 = imresize(image, .75);
disp('classify squares');
startClock = clock
while (size(image2,1) > 24 && size(image2,2) > 24)
    classify_image(image2, FF, FINALTHRESH, bests);
    image2 = imresize(image, size(image2)*.5);
end

clock