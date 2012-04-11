function [] = face_detector(faces, nonfaces, image)

Fvec = reshape(faces,24*24,[]);
NFvec = reshape(nonfaces,24*24,[]);

%% sweet!  now let's do Robert's crummy but intuitive boosting...
allFaces = [Fvec NFvec];
numFaces = size(Fvec,2);
numNonFaces = size(NFvec,2);
desiredOut = [ones(1,size(Fvec,2)) -ones(1,size(NFvec,2))]';
% make the total weight of faces and non faces the same (so that just
% calling everything "not a face" isn't a win...
weights = [numNonFaces.*ones(1,size(Fvec,2)) numFaces.*ones(1,size(NFvec,2))]';
weights = weights./sum(weights(:));
%% now, make Features
[FINALFEAT FINALTHRESH featureRanking] = train_classifiers(allFaces, desiredOut, weights);

%% display training results
FF = reshape(FINALFEAT,576,[]);         % Reshape all the good features into one matrix
AS = FF'*[Fvec NFvec];                      % Compute the score of every face with every feature.
AT = repmat(FINALTHRESH',1,size(AS,2)); % create matrix of all thresholds, replicating it so its same size as AS
VOTES = sign( AS - AT);                 % compute weak classification  of all faces for all features
CLASSIFICATION = sign(sum(VOTES)-eps);  % sum the classifications.  if something has EXACTLY the same number of 
                                        % yes and no votes, then it is 0
                                        % instead of -1 of +1, so -eps
                                        % makes sure that doesn't happen.
fprintf('true positive: %d %%  or %d out of %d faces \n',sum(CLASSIFICATION' == 1 &  desiredOut == 1)/numFaces,sum(CLASSIFICATION' == 1 &  desiredOut == 1),numFaces);
fprintf('true negative: %d %%  or %d out of %d faces \n', sum(CLASSIFICATION' == -1 &  desiredOut == -1)/numNonFaces, sum(CLASSIFICATION' == -1 &  desiredOut == -1),numNonFaces);
fprintf('false negatives: %d %%  or %d out of %d faces \n',sum(CLASSIFICATION' == -1 &  desiredOut == 1)/numFaces,sum(CLASSIFICATION' == -1 &  desiredOut == 1),numFaces);
fprintf('false positives: %d %%  or %d out of %d faces \n', sum(CLASSIFICATION' == 1 &  desiredOut == -1)/numNonFaces, sum(CLASSIFICATION' == 1 &  desiredOut == -1),numNonFaces);

%% now let's do it for a subset of the faces...
testCases = [Fvec(:,size(Fvec,2)-99:size(Fvec,2)) NFvec(:,size(NFvec,2)-99:size(NFvec,2))];
Fvec = Fvec(:,1:size(Fvec,2)-100);
NFvec = NFvec(:,1:size(NFvec,2)-100);
allFaces = [Fvec, NFvec];
numFaces = size(Fvec,2);
numNonFaces = size(NFvec,2);
desiredOut = [ones(1,size(Fvec,2)) -ones(1,size(NFvec,2))]';
% make the total weight of faces and non faces the same (so that just
% calling everything "not a face" isn't a win...
weights = [numNonFaces.*ones(1,size(Fvec,2)) numFaces.*ones(1,size(NFvec,2))]';
weights = weights./sum(weights(:));
%% now, make Features
[FINALFEAT FINALTHRESH featureRanking] = train_classifiers(allFaces, desiredOut, weights);

desiredOut = [ones(1,100) -ones(1,100)]';

%% display training results
FF = reshape(FINALFEAT,576,[]);         % Reshape all the good features into one matrix
AS = FF'*testCases;                      % Compute the score of every face with every feature.
AT = repmat(FINALTHRESH',1,size(AS,2)); % create matrix of all thresholds, replicating it so its same size as AS
VOTES = sign( AS - AT);                 % compute weak classification  of all faces for all features
CLASSIFICATION = sign(sum(VOTES)-eps);  % sum the classifications.  if something has EXACTLY the same number of 
                                        % yes and no votes, then it is 0
                                        % instead of -1 of +1, so -eps
                                        % makes sure that doesn't happen.
                
fprintf('true positive: %d %%  or %d out of %d faces \n',sum(CLASSIFICATION' == 1 &  desiredOut == 1)/100,sum(CLASSIFICATION' == 1 &  desiredOut == 1),100);
fprintf('true negative: %d %%  or %d out of %d faces \n', sum(CLASSIFICATION' == -1 &  desiredOut == -1)/100, sum(CLASSIFICATION' == -1 &  desiredOut == -1),100);
fprintf('false negatives: %d %%  or %d out of %d faces \n',sum(CLASSIFICATION' == -1 &  desiredOut == 1)/100,sum(CLASSIFICATION' == -1 &  desiredOut == 1),100);
fprintf('false positives: %d %%  or %d out of %d faces \n', sum(CLASSIFICATION' == 1 &  desiredOut == -1)/100, sum(CLASSIFICATION' == 1 &  desiredOut == -1),100);



%% classify an image
%image = rgb2gray(imread('/Users/nathan/Development/CSE559/Project3/data/lotr_cast1.jpg'));
%%LOTR cast
image2 = image;
%image3 = rgb2gray(imread('/Users/nathan/Development/CSE559/Project3/data/princess_bride.jpg'));
%%LOTR cast
startClock = clock;
while (size(image2,1) > 24 && size(image2,2) > 24)
    combo_classify_image(image2, FF, FINALTHRESH,featureRanking, 'lotr_cast2');
    image2 = imresize(image, size(image2)*.8);
end
disp('time to classify image pyramids squares ');
clock-startClock