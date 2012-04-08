function [FEAT] = generate_feature()
rectType = ceil(rand(1,1)*5);

% make rectangular feature:
FEAT = zeros(24,24);
switch rectType
    case 1,   % horizontal, 2 part.
        xmin = ceil(rand(1,1)*22);
        xsizePossible = 24-xmin;
        xmax = xmin+2*ceil(rand(1,1)*xsizePossible/2);
        xmid = (xmin+xmax)/2;
        xmax = xmax-1; % max indexing easier).
        ymin = ceil(rand(1,1)*23);
        ysizePoss = 24-ymin;
        ymax = ymin + ceil(rand(1,1)*ysizePoss);
        FEAT(xmin:xmid-1,ymin:ymax) = 1;
        FEAT(xmid:xmax,ymin:ymax) = -1;
    case 2,  % vertical feature.  but we compute as if horizontal feature
        xmin = ceil(rand(1,1)*22);
        xsizePossible = 24-xmin;
        xmax = xmin+2*ceil(rand(1,1)*xsizePossible/2);
        xmid = (xmin+xmax)/2;
        xmax = xmax-1; % max indexing easier).
        ymin = ceil(rand(1,1)*23);
        ysizePoss = 24-ymin;
        ymax = ymin + ceil(rand(1,1)*ysizePoss);
            % ... then with horrible horrible coding practice, use the
            % wrong variables to index into the array
        FEAT(ymin:ymax,xmin:xmid-1) = 1;
        FEAT(ymin:ymax,xmid:xmax) = -1;
    case 3,  % 3 part feature.
        xmin = ceil(rand(1,1)*21);
        xsizePossible = 22-xmin;
        xmax = xmin+3*ceil(rand(1,1)*xsizePossible/3);
        xthird1 = (2*xmin+xmax)/3;
        xthird2 = (xmin+2*xmax)/3;
        ymin = ceil(rand(1,1)*23);
        ysizePoss = 24-ymin;
        ymax = ymin + ceil(rand(1,1)*ysizePoss);
        FEAT(xmin:xthird1-1,ymin:ymax) = 1;
        FEAT(xthird1:xthird2-1,ymin:ymax) = -1;
        FEAT(xthird2:xmax-1,ymin:ymax) = 1;
    case 4,
        xmin = ceil(rand(1,1)*21);
        xsizePossible = 22-xmin;
        xmax = xmin+3*ceil(rand(1,1)*xsizePossible/3);
        xthird1 = (2*xmin+xmax)/3;
        xthird2 = (xmin+2*xmax)/3;
        ymin = ceil(rand(1,1)*23);
        ysizePoss = 24-ymin;
        ymax = ymin + ceil(rand(1,1)*ysizePoss);
        FEAT(ymin:ymax,xmin:xthird1-1) = 1;
        FEAT(ymin:ymax,xthird1:xthird2-1) = -1;
        FEAT(ymin:ymax,xthird2:xmax-1) = 1;
    case 5
        xmin = ceil(rand(1,1)*22);
        xsizePossible = 22-xmin;
        xmax = xmin+2*ceil(rand(1,1)*xsizePossible/2);
        xmid = (xmin+xmax)/2;
        % my god, this is even worse coding...
        ymin = xmin;
        ymid = xmid;
        ymax = xmax;
        % now, err, recompute x
        xmin = ceil(rand(1,1)*22);
        xsizePossible = 24-xmin;
        xmax = xmin+2*ceil(rand(1,1)*xsizePossible/2);
        xmid = (xmin+xmax)/2;
        %
        FEAT(xmin:xmax-1,ymin:ymax-1) = -1;
        FEAT(xmin:xmid-1,ymin:ymid-1) = 1; 
        FEAT(xmid:xmax-1,ymid:ymax-1) = 1;
end

% sweet, as long as we are trying random features, let's randomize the
% sign.
if rand(1,1)<0.5, FEAT = FEAT.*-1; end