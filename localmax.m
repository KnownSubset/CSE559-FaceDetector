function [maximas] = localmax(matrix)   
    
    bw = matrix > imdilate(matrix,[1 1 1; 1 0 1; 1 1 1]);
    maximas = zeros(1, sum(sum(bw)));
    index = 1;
    for ix = 1:size(bw,1)
        for iy = 1:size(bw, 2)
            if (bw(ix,iy) > 0)
                maximas(index) = (ix - 1)*size(bw,2) + iy;
                index = index + 1;
            end
        end
    end
end
