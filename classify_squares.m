function [CLASSIFICATION VOTES] = classify_squares(squares, FF, FINALTHRESH)
    squares2 = zeros(24*24,size(squares,3));
    rowRange = size(squares,1);
    colRange = size(squares,2);
    if (rowRange < 1 || colRange < 1)
        return
    end
    for ix = 1:size(squares,3)
        squares2(:,ix) = reshape(squares(:,:,ix),576,[]);
    end
    AS = FF'*squares2;                      % Compute the score of every face with every feature.
    AT = repmat(FINALTHRESH',1,size(AS,2)); % create matrix of all thresholds, replicating it so its same size as AS
    VOTES = sign( AS - AT);                 % compute weak classification  of all faces for all features
    CLASSIFICATION = sign(sum(VOTES)-eps);  % sum the classifications.  if something has EXACTLY the same number of 
