function [value] = calculateIntegralImageSection(image, points)
row1 = points(1);
row2 = points(3);
col1 = points(2);
col2 = points(4);
value = image(row1, col1) + image(row2, col2) - image(row1, col2) - image(row2, col1);
value = 255 * value;