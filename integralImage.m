function [integralValues] = integralImage(image)

integralValues = zeros(size(image));
integralValues(1,:) = image(1,:);
previousRow = integralValues(1, :);
for row = 2:size(image,1)
  integralValues(row, 1) = image(row, 1) + integralValues(row-1, 1); 
end
for row = 2:size(image,1)
   for column = 2:size(image,2)
      integralValues(row, column) = previousRow(column) + integralValues(row, column -1); 
   end
   previousRow = integralValues(row,:);
end