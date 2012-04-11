function [value] = calcII(image, points)
row1 = points(1);
row2 = points(3);
col1 = points(2);
col2 = points(4);
value = 0;
if (row1 >0 && row2 >0 && col1 >0 && col2 >0) 
    value = image(row1, col1) + image(row2, col2) - image(row1, col2) - image(row2, col1);
end
value = 255 * value;