function [] = calc_score_II(allFaces, desiredOut, FINALFEAT_II, FINALTHRESH, featureRanking, numFaces, numNonFaces)

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
                                        
fprintf('true positive: %d %% or %d out of %d \n',sum(CLASSIFICATION' == 1 &  desiredOut == 1)/numFaces,sum(CLASSIFICATION' == 1 &  desiredOut == 1),numFaces);
fprintf('true negative: %d %% or %d out of %d \n', sum(CLASSIFICATION' == -1 &  desiredOut == -1)/numNonFaces, sum(CLASSIFICATION' == -1 &  desiredOut == -1),numNonFaces);
fprintf('false negatives: %d %% or %d out of %d \n',sum(CLASSIFICATION' == -1 &  desiredOut == 1)/numFaces,sum(CLASSIFICATION' == -1 &  desiredOut == 1),numFaces);
fprintf('false positives: %d %% or %d out of %d \n', sum(CLASSIFICATION' == 1 &  desiredOut == -1)/numNonFaces, sum(CLASSIFICATION' == 1 &  desiredOut == -1),numNonFaces);