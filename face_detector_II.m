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
[FINALFEAT_II FINALTHRESH featureRanking] = train_classifiers_II(allFaces, desiredOut, weights, numFaces, numNonFaces);
calc_score_II(allFaces,desiredOut, FINALFEAT_II, FINALTHRESH, featureRanking, numFaces, numNonFaces);

%% now retain on everything but the last 100, and test against those last 100
numFaces = size(faces,3) - 100;
numNonFaces = size(nonfaces,3) - 100;
allFaces = zeros(24,24, numFaces + numNonFaces);
allFaces(:,:,1:numFaces) = facesII(:,:,1:numFaces);
allFaces(:,:,numFaces+1:numFaces+numNonFaces) = nonfacesII(:,:,1:numNonFaces);

desiredOut = [ones(1,numFaces) -ones(1,numNonFaces)]';
% make the total weight of faces and non faces the same (so that just
% calling everything "not a face" isn't a win...
weights = [numNonFaces.*ones(1,numFaces) numFaces.*ones(1,numNonFaces)]';
weights = weights./sum(weights(:));
[FINALFEAT_II FINALTHRESH featureRanking] = train_classifiers_II(allFaces, desiredOut, weights, numFaces, numNonFaces);

allFaces = zeros(24,24, 200);
allFaces(:,:,1:100) = facesII(:,:,numFaces+1:numFaces+100);
allFaces(:,:,101:200) = nonfacesII(:,:,numNonFaces+1:numNonFaces+100);
calc_score_II(allFaces,[ones(1,100) -ones(1,100)]', FINALFEAT_II, FINALTHRESH, featureRanking, 100, 100);

%% classify an image
startClock = clock;
image2 = im2double(image);
while (size(image2,1) > 24 && size(image2,2) > 24)
    totalSquares = (size(image2,1)-23)*(size(image2,2)-23);
    squares = zeros(24,24,totalSquares);
    rowRange = size(image2,1) - 23;
    colRange = size(image2,2) - 23;
    for ix = 1:rowRange
        for iy = 1:colRange
            squares(:,:,(ix-1)*colRange + iy) = cumsum(cumsum(image2(ix:ix+23, iy:iy+23),1),2);
        end
    end
    
    AS = zeros(100,1, totalSquares);
    for fx = 1:100
        POSITIVE = reshape(FINALFEAT_II(1,:,fx),4,2)';        
        NEGATIVE = reshape(FINALFEAT_II(2,:,fx),4,2)';
        for px = 1 : 2
            points = POSITIVE(px, 1:4);
            row1 = points(1);
            row2 = points(3);
            col1 = points(2);
            col2 = points(4);
            if (row1 > 0 )
                AS(fx,1,:) = AS(fx,1,:) + (squares(row1, col1 ,:) + squares(row2, col2 ,:) - squares(row1, col2, :) - squares(row2, col1, :) );
            end
        end
        for px = 1 : 2
            points = NEGATIVE(px, 1:4);
            row1 = points(1);
            row2 = points(3);
            col1 = points(2);
            col2 = points(4);
            if (row1 > 0)
                AS(fx,1,:) = AS(fx,1,:) - (squares(row1, col1 ,:) + squares(row2, col2 ,:) - squares(row1, col2, :) - squares(row2, col1, :)) ;
            end
        end
    end
    beep

    AS = reshape(AS, 100, totalSquares);
    AT = repmat(FINALTHRESH',1,size(AS,2)); % create matrix of all thresholds, replicating it so its same size as AS
    VOTES = sign(AS - AT);                     
    image2 = imresize(image, size(image2));
    locs =  localmax(reshape(sum(VOTES),rowRange,colRange));
    for ix = 1:size(locs,2)
        row = round(locs(ix) / colRange) + 1;
        col = round(mod(locs(ix),colRange))+1;
        image2(row,col:col+23) = 255;
        image2(row+23,col:col+23) = 255;
        image2(row:row+23,col) = 255;
        image2(row:row+23,col+23) = 255;
    end
    %figure, colormap gray;
    %imagesc(image2);
    imwrite(image2, sprintf('/Users/nathan/Development/CSE559/Project3/images/%s_II_%d_%d.jpg',prefix, size(image2,1),size(image2,2)));
    image2 = im2double(imresize(image, size(image2)*.8));
end

disp('time to classify image pyramids squares ');
clock-startClock
