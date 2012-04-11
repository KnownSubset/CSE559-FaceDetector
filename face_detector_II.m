function [] = face_detector_II(faces, nonfaces, image)
%% generate interval images for each face and nonface
startClock = clock;
facesII = zeros(size(faces));
for ix = 1:size(faces,3)
    facesII(:,:,ix) = cumsum(cumsum(faces(:,:,ix),1),2);
end
nonfacesII = zeros(size(nonfaces));
for ix = 1:size(nonfaces,3)
    nonfacesII(:,:,ix) = cumsum(cumsum(nonfaces(:,:,ix),1),2);
end
disp('integral image calcs');
disp(clock - startClock);


%% sweet!  now let's do Robert's crummy but intuitive boosting...

numFaces = size(faces,3);
numNonFaces = size(nonfaces,3);
allFaces = zeros(24,24, numFaces + numNonFaces);
allFaces(:,:,1:numFaces) = facesII;
allFaces(:,:,numFaces+1:numFaces+numNonFaces) = nonfacesII;

desiredOut = [ones(1,numFaces) -ones(1,numNonFaces)]';
% make the total weight of faces and non faces the same (so that just
% calling everything "not a face" isn't a win...
weights = [numNonFaces.*ones(1,numFaces) numFaces.*ones(1,numNonFaces)]';
weights = weights./sum(weights(:));
%% now, make 20 Features
% initial some variables
FINALFEAT_II = zeros(2,8,100);
FINALTHRESH = zeros(1,100);
bests = [];
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
    bests(numFeats) = bestWeakClassifierScore;
    
    % ok... so the above loop picked the best of 100 possible features.
    % now, let's update the weights of the samples.
    weights = weights .* exp(-desiredOut .* weakClassifier);
    
    % normalize the weights or they'll go crazy.
    weights = weights./sum(weights(:));
end

startClock =  clock - startClock;
fprintf('finished training %d mins %d secs for #%d faces \r', startClock(5), startClock(6), size(allFaces,3));

%%
[y i] = sort(bests,2,'descend');
AS = zeros(100,1, numFaces + numNonFaces);
for fx = 1:100
    POSITIVE = reshape(FINALFEAT_II(1,:,fx),4,2)';        
    NEGATIVE = reshape(FINALFEAT_II(2,:,fx),4,2)';
    for px = 1 : 2
        points = POSITIVE(px, 1:4);
        row1 = points(1);
        row2 = points(3);
        col1 = points(2);
        col2 = points(4);
        if (row1 > 0 && row2 > 0)
            AS(fx,1,:) = AS(fx,1,:) + (allFaces(row1, col1 ,:) + allFaces(row2, col2 ,:) - allFaces(row1, col2, :) - allFaces(row2, col1, :) );
        end
    end
    for px = 1 : 2
        points = NEGATIVE(px, 1:4);
        row1 = points(1);
        row2 = points(3);
        col1 = points(2);
        col2 = points(4);
        if (row1 > 0 && row2 > 0)
            AS(fx,1,:) = AS(fx,1,:) - (allFaces(row1, col1 ,:) + allFaces(row2, col2 ,:) - allFaces(row1, col2, :) - allFaces(row2, col1, :)) ;
        end
    end
end
beep

AS = reshape(AS, 100, numFaces+numNonFaces);

%AS = FF'*allFaces;                      % Compute the score of every face with every feature.
AT = repmat(FINALTHRESH',1,size(AS,2)); % create matrix of all thresholds, replicating it so its same size as AS
VOTES = sign(AS - AT);                 % compute weak classification  of all faces for all features
CLASSIFICATION = sign(sum(VOTES)-eps);  % sum the classifications.  if something has EXACTLY the same number of 
                                        % yes and no votes, then it is 0
                                        % instead of -1 of +1, so -eps
                                        % makes sure that doesn't happen.                                      
                                        
fprintf('true positive: %d %% \n',sum(CLASSIFICATION' == 1 &  desiredOut == 1)/numFaces);
fprintf('true negative: %d %% \n', sum(CLASSIFICATION' == -1 &  desiredOut == -1)/numNonFaces);
fprintf('false negatives: %d %% \n',sum(CLASSIFICATION' == -1 &  desiredOut == 1)/numFaces);
fprintf('false positives: %d %% \n', sum(CLASSIFICATION' == 1 &  desiredOut == -1)/numNonFaces);

%% classify an image
%image = rgb2gray(imresize(imread('http://i.imgur.com/02npE.jpg'),.5));
% image2 = imresize(image, .25);
% %while (size(image2,1) > 24 && size(image2,2) > 24)
%     squares = zeros(24,24,(size(image2,1)-23)*(size(image2,2)-23));
%     squares2 = zeros(24*24,(size(image2,1)-23)*(size(image2,2)-23));
%     rowRange = size(image2,1) - 23;
%     colRange = size(image2,2) - 23;
%     for ix = 1:rowRange
%         for iy = 1:colRange
%             squares(:,:,(ix-1)*colRange + iy) = image2(ix:ix+23, iy:iy+23);
%             squares2(:,(ix-1)*colRange + iy) = reshape(squares(:,:,(ix-1)*colRange + iy),576,[]);
%         end
%     end
%     AS = FF'*squares2;                      % Compute the score of every face with every feature.
%     AT = repmat(FINALTHRESH',1,size(AS,2)); % create matrix of all thresholds, replicating it so its same size as AS
%     VOTES = sign( AS - AT);                 % compute weak classification  of all faces for all features
%     CLASSIFICATION = sign(sum(VOTES)-eps);  % sum the classifications.  if something has EXACTLY the same number of 
%     if (sum(CLASSIFICATION' == 1) > 0)
%         disp('Number of labelled faces: ');
%         size(image)
%         sum(CLASSIFICATION' == 1) 
%         sum(CLASSIFICATION' == -1)
%     end
%     
%     locs =  localmax(reshape(sum(VOTES),rowRange,colRange));
%     for ix = 1:size(locs,2)
%         row = round(locs(ix) / colRange) + 1;
%         col = round(mod(locs(ix),colRange))+1;
%         image2(row,col:col+23) = 255;
%         image2(row+23,col:col+23) = 255;
%         image2(row:row+23,col) = 255;
%         image2(row:row+23,col+23) = 255;
%     end
%     figure, colormap gray;
%     imagesc(image2);

%    image2 = imresize(image, size(image2)*.5);
%end
