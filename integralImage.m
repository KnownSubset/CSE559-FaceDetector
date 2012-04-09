function [integralValues] = integralImage(image)
integralValues = double(zeros(size(image) + 1));
for row = 1:size(image,1)
   for column = 1:size(image,2)
       one = integralValues(row+1, column);
       two = integralValues(row, column + 1);
       integralValues(row+1, column + 1) = double(image(row,column))/255 + one + two + integralValues(row, column); 
   end
end
integralValues = integralValues(2:size(integralValues,1),2:size(integralValues,2));