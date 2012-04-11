function [] = combo_classify_image(image, FF, FINALTHRESH, featureRanking, prefix)
    rowRange = size(image,1) - 23;
    colRange = size(image,2) - 23;
    %% using regular way
    VOTES = classify_image(image, FF, FINALTHRESH);    
    %figure, colormap gray;
    %subplot(1,3,1);
    %imagesc(image);
    %image2 = image;
    locs =  localmax(reshape(sum(VOTES),rowRange,colRange));
    for ix = 1:size(locs,2)
        row = floor(locs(ix) / colRange) + 1;
        col = floor(mod(locs(ix),colRange))+1;
        image2(row,col:col+23) = 255;
        image2(row+23,col:col+23) = 255;
        image2(row:row+23,col) = 255;
        image2(row:row+23,col+23) = 255;
    end
    %subplot(1,3,2);
    %imagesc(image2);
    imwrite(image2, sprintf('/Users/nathan/Development/CSE559/Project3/images/%s_noncascade_%d_%d.jpg',prefix, size(image,1),size(image,2)));
    
    %% classify using cascade
    image2 = image;
    VOTES2 = cascade_classify_image(image, FF, FINALTHRESH, featureRanking); 
    locs =  localmax(reshape(sum(VOTES2),rowRange,colRange));
    for ix = 1:size(locs,2)
        row = round(locs(ix) / colRange) + 1;
        col = round(mod(locs(ix),colRange))+1;
        image2(row,col:col+23) = 255;
        image2(row+23,col:col+23) = 255;
        image2(row:row+23,col) = 255;
        image2(row:row+23,col+23) = 255;
    end
    %subplot(1,3,3);
    %imagesc(image2);
    imwrite(image2, sprintf('/Users/nathan/Development/CSE559/Project3/images/%s_cascade_%d_%d.jpg',prefix, size(image,1),size(image,2)));