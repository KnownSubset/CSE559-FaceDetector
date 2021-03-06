function [VOTES] = classify_image(image, FF, FINALTHRESH)
    image2 = image;
    squares = zeros(24,24,(size(image2,1)-23)*(size(image2,2)-23));
    squares2 = zeros(24*24,(size(image2,1)-23)*(size(image2,2)-23));
    rowRange = size(image2,1) - 23;
    colRange = size(image2,2) - 23;
    if (rowRange < 1 || colRange < 1)
        return
    end
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

    