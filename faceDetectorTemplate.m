
%%
clear;              % clear all variables from your workspace
%      equivalent to:  ".... and now for something completely different"

% the following two files require that you download a bit of matlab code
% and a data file from:
% http://www.cs.ubc.ca/~pcarbo/viola-traindata.tar.gz
%faces = getFaces('data/faces.mat',24,0);        % reads from a file
%nonfaces = getFaces('data/nonfaces.mat',24,0);  % reads from a file

for ix = 1:4916            % loop over all faces
  imagesc(nonfaces(:,:,ix));  % display the ix-th face.
 % imagesc(faces(:,:,ix));  % display the ix-th face.
  drawnow;
end
%%
imagesc(mean(faces,3));  % take the mean in the third dimension.

imagesc(var(faces,3));  % gives an error (happens to me every time)

help var                % figure out how not to get the error

imagesc(var(faces,1,3)); % makes a color coded map of how much different
                        % regions vary.
			
imagesc(mean(nonfaces,3));   % show some love to your negative examples too
imagesc(var(nonfaces,1,3));			

% montage is an awesome command, but a little dirty.  look up for help if
% you want.
%%
figure(1);
montage(permute(faces(:,:,1:10:640)./255,[1 2 4 3]));
figure(2);
montage(permute(nonfaces(:,:,1:100:6400)./255,[1 2 4 3]));

%%%%% Now, in class break.  everyone spend 5 minutes thinking of some
%test that we can run on a box.


%% compelling matlab trickery.
% Quickly: try "difference from mean face" as a face classifier.
nonfaces = nonfaces(:,:,1:size(faces,3));

c = mean(faces,3);

cBig = repmat(c, [1 1 size(faces,3)]);

diff = faces - cBig;
diffScore = sum(sum(diff.^2,2),1);

dBig = repmat(c, [1 1 size(nonfaces,3)]);
diff2 = nonfaces - dBig;
diffScore2 = sum(sum(diff2.^2,2),1);

[counts bins] = hist(diffScore(:),100);
[counts2 bins2] = hist(diffScore2(:),bins);
plot(bins,counts,'r');
hold on;
plot(bins2,counts2,'b');
hold off;

%% set up for feature scoring
features = [];
featScores = [];

for ix = 1:1000
    FEAT = generate_feature();
    subplot(1,2,1);
    imagesc(FEAT);


    % lets compute the score of some features:
    subplot(1,2,2);
    Fvec = reshape(faces,24*24,[]);
    NFvec = reshape(nonfaces,24*24,[]);

    scores = Fvec' * FEAT(:);

    [counts bins] = hist(Fvec' * FEAT(:),100);
    [counts2 bins2] = hist(NFvec' * FEAT(:),100);
    plot(bins,counts,'r');
    colormap gray;
    hold on;
    plot(bins2,counts2,'b');
    hold off;
    

    % Now, evaluate a feature:

    features = cat(2, features, [FEAT(:)]);
    featScore = max(abs(counts - counts2));
    featScores = cat(1, featScores, [featScore ix]);

end

