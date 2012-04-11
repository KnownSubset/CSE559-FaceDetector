function [VOTES] = cascade_classify_image_II(image, FINALFEAT_II, FINALTHRESH, featureRanking)
    image2 = im2double(image);
   
    %% Create a square for each possible face
    
    rowRange = size(image2,1) - 23;
    colRange = size(image2,2) - 23;
    squares = zeros(24,24,rowRange*colRange);
    if (rowRange < 1 || colRange < 1)
        return
    end
    for ix = 1:rowRange
        for iy = 1:colRange
            squares(:,:,(ix-1)*colRange + iy) = cumsum(cumsum(image2(ix:ix+23, iy:iy+23),1),2);
        end
    end
   
   %% Select the cascade of features to classify possible faces 
   [~, i] = sort(featureRanking,2,'descend');
   
   squaresIndexes = [1:size(squares,3)];
   
   %% classify 
   for ij = 1:100    
        startClock = clock;
        AS = zeros(100,1, rowRange * colRange);
        for fx = 1:ij*100
            POSITIVE = reshape(FINALFEAT_II(1,:,i(fx)),4,2)';        
            NEGATIVE = reshape(FINALFEAT_II(2,:,i(fx)),4,2)';
            for px = 1 : 2
                points = POSITIVE(px, 1:4);
                row1 = points(1);
                row2 = points(3);
                col1 = points(2);
                col2 = points(4);
                if (row1 > 0 && row2 > 0)
                    AS(fx,1,:) = AS(fx,1,:) + (squares(row1, col1 ,:) + squares(row2, col2 ,:) - squares(row1, col2, :) - squares(row2, col1, :) );
                end
            end
            for px = 1 : 2
                points = NEGATIVE(px, 1:4);
                row1 = points(1);
                row2 = points(3);
                col1 = points(2);
                col2 = points(4);
                if (row1 > 0 && row2 > 0)
                    AS(fx,1,:) = AS(fx,1,:) - (squares(row1, col1 ,:) + squares(row2, col2 ,:) - squares(row1, col2, :) - squares(row2, col1, :)) ;
                end
            end
        end
        beep

        AS = reshape(AS, 100, rowRange * colRange);

        %AS = FF'*allFaces;                      % Compute the score of every face with every feature.
        AT = repmat(FINALTHRESH',1,size(AS,2)); % create matrix of all thresholds, replicating it so its same size as AS
        VOTES = sign(AS - AT);                 % compute weak classification  of all faces for all features
        CLASSIFICATION = sign(sum(VOTES)-eps);  % sum the classifications.  if something has EXACTLY the same number of 

       %% build squares out of 'positive' faces for next cascade 
       newSquares = zeros(24,24, sum(CLASSIFICATION == 1));
       face_ndx = 1;
       temp = zeros(1, sum(CLASSIFICATION == 1));
       
       for ix = 1:size(squares,3)
           if (CLASSIFICATION(ix) == 1)
            temp(face_ndx) = squaresIndexes(ix);
            newSquares(:, :, face_ndx) = squares(:,:,ix);
            face_ndx = face_ndx + 1;
           end
       end
       %disp(fprintf('cascade # %d found %d faces out of %d possible faces', ij, size(newSquares,3), size(squares,3)));
       %clock - startClock
       squares = newSquares;
       squaresIndexes = temp;
   end
   
   ALL_VOTES = zeros(100, (size(image2,1)-23)*(size(image2,2)-23));
   for ix = 1:size(squaresIndexes,2)
       ALL_VOTES(:,squaresIndexes(ix)) = VOTES(ix);
   end
   VOTES = ALL_VOTES;
   