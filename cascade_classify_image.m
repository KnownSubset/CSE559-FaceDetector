function [VOTES] = cascade_classify_image(image, FF, FINALTHRESH, featureRanking)
    image2 = image;
   
    %% Create a square for each possible face
    squares = zeros(24,24,(size(image2,1)-23)*(size(image2,2)-23));
    rowRange = size(image2,1) - 23;
    colRange = size(image2,2) - 23;
    if (rowRange < 1 || colRange < 1)
        return
    end
    for ix = 1:rowRange
        for iy = 1:colRange
            squares(:,:,(ix-1)*colRange + iy) = image2(ix:ix+23, iy:iy+23);
        end
    end
   
   %% Select the cascade of features to classify possible faces 
   [~, i] = sort(featureRanking,2,'descend');
   
   squaresIndexes = [1:size(squares,3)];
   for ij = 1:10
       startClock = clock;

       %% classify 
       [CLASSIFICATION VOTES] = classify_squares(squares, FF(:,1:(10*ij)), FINALTHRESH(1:(10*ij)));

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
       disp(fprintf('cascade # %d found %d faces out of %d possible faces', ij, size(newSquares,3), size(squares,3)));
       clock - startClock
       squares = newSquares;
       squaresIndexes = temp;
   end
   ALL_VOTES = zeros(100, (size(image2,1)-23)*(size(image2,2)-23));
   for ix = 1:size(squaresIndexes,2)
       ALL_VOTES(:,squaresIndexes(ix)) = VOTES(ix);
   end
   VOTES = ALL_VOTES;
   