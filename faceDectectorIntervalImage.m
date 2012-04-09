%% generate interval images for each face and nonface
startClock = clock;
facesII = zeros(size(faces));
for ix = 1:size(faces,3)
    facesII(:,:,ix) = integralImage(faces(:,:,ix));
end
nonfacesII = zeros(size(nonfaces));
for ix = 1:size(nonfaces,3)
    nonfacesII(:,:,ix) = integralImage(nonfaces(:,:,ix));
end
disp('integral image calcs');
clock - startClock

%% make viola jones features
Fvec = reshape(faces,24*24,[]);
NFvec = reshape(nonfaces,24*24,[]);


%% sweet!  now let's do Robert's crummy but intuitive boosting...
numFaces = size(facesII,3);
numNonFaces = size(nonfacesII,3);
desiredOut = [ones(1,size(facesII,3)) -ones(1,size(nonfacesII,3))]';
% make the total weight of faces and non faces the same (so that just
% calling everything "not a face" isn't a win...
weights = [numNonFaces.*ones(1,size(facesII,3)) numFaces.*ones(1,size(nonfacesII,3))]';
weights = weights./sum(weights(:));
%% now, make 20 Features
% initial some variables
FINALFEAT = [];
FINALTHRESH = [];
bests = [];
startClock = clock;
for numFeats = 1:100
   
    % so, 200 times, we're going to find a good classifier.  we're going
    % to try a bunch of classifiers each time, so lets' set up a variable
    % to see keep track of the best one we've seen so far
    bestWeakClassifierScore = 0;
    
    
    for jx = 1:10  % boring for loops to always count up!
        %generate_feature and have it return the corners of positive regions, and corners of negative regions 
        %score of face = (sum up positive - sum of negative regions) 
        
        [POSITIVE NEGATIVE] = gen_interval_feature;                  % make a random feature.
        scores = zeros(1, numFaces + numNonFaces);
        for fx = 1:numFaces
            for px = 1:size(POSITIVE, 1)
                scores(1,fx) = scores(1,fx) + calculateIntegralImageSection(facesII(:,:,fx),POSITIVE(px,:));
            end
        end
        for sx = 1:numNonFaces
            for nx = 1:size(NEGATIVE, 1)
                scores(1,sx+numFaces) = scores(1,sx+numFaces) - calculateIntegralImageSection(nonfacesII(:,:,sx),NEGATIVE(nx,:));
            end
        end
      
        % now try different thresholds.
        thresholdList = linspace(min(scores),max(scores),1000);  % make 1000 thresholds.
        cScore = 0;    %initialize some stuff about those thresholds.
        cThresh = 0;
        for ix = 1:1000
            % compute classification result with this threshold
            classifierResult = sign(scores-thresholdList(ix));
            
            % compute "weighted" score for each face.
            tmp = (classifierResult' .* desiredOut .* weights)';
            
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
        weakClassifierScore = sum(weakClassifier' .* desiredOut .*weights);
        
        % if better than we've seen so far, then save it.
        if weakClassifierScore > bestWeakClassifierScore
            bestWeakClassifierScore = weakClassifierScore;
            %FINALFEAT(:,:,numFeats) = FEAT;
            FINALTHRESH(numFeats) = cThresh;
        end
    end
    bests(numFeats) = bestWeakClassifierScore;
    
    % ok... so the above loop picked the best of 100 possible features.
    % now, let's update the weights of the samples.
    weights = weights .* exp(-desiredOut .* weakClassifier');
    
    % normalize the weights or they'll go crazy.
    weights = weights./sum(weights(:));
end
disp('training time');
clock - startClock
%%
[y i] = sort(bests,2,'descend');
FF = reshape(FINALFEAT,576,[]);         % Reshape all the good features into one matrix
AS = FF'*allFaces;                      % Compute the score of every face with every feature.
AT = repmat(FINALTHRESH',1,size(AS,2)); % create matrix of all thresholds, replicating it so its same size as AS
VOTES = sign( AS - AT);                 % compute weak classification  of all faces for all features

for ix = 1:size(i,2)
   VOTES(ix,:) = VOTES(ix,:)*(y(ix)); 
end

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
%image = rgb2gray(imresize(imread('http://i.imgur.com/02npE.jpg'),.5));
image2 = imresize(image, .25);
%while (size(image2,1) > 24 && size(image2,2) > 24)
    squares = zeros(24,24,(size(image2,1)-23)*(size(image2,2)-23));
    squares2 = zeros(24*24,(size(image2,1)-23)*(size(image2,2)-23));
    rowRange = size(image2,1) - 23;
    colRange = size(image2,2) - 23;
    for ix = 1:rowRange
        for iy = 1:colRange
            squares(:,:,(ix-1)*colRange + iy) = image2(ix:ix+23, iy:iy+23);
            squares2(:,(ix-1)*colRange + iy) = reshape(squares(:,:,(ix-1)*colRange + iy),576,[]);
        end
    end
    AS = FF'*squares2;                      % Compute the score of every face with every feature.
    AT = repmat(FINALTHRESH',1,size(AS,2)); % create matrix of all thresholds, replicating it so its same size as AS
    VOTES = sign( AS - AT);                 % compute weak classification  of all faces for all features
    CLASSIFICATION = sign(sum(VOTES)-eps);  % sum the classifications.  if something has EXACTLY the same number of 
    if (sum(CLASSIFICATION' == 1) > 0)
        disp('Number of labelled faces: ');
        size(image)
        sum(CLASSIFICATION' == 1) 
        sum(CLASSIFICATION' == -1)
    end
    
    locs =  localmax(reshape(sum(VOTES),rowRange,colRange));
    for ix = 1:size(locs,2)
        row = round(locs(ix) / colRange) + 1;
        col = round(mod(locs(ix),colRange))+1;
        image2(row,col:col+23) = 255;
        image2(row+23,col:col+23) = 255;
        image2(row:row+23,col) = 255;
        image2(row:row+23,col+23) = 255;
    end
    figure, colormap gray;
    imagesc(image2);

%    image2 = imresize(image, size(image2)*.5);
%end
