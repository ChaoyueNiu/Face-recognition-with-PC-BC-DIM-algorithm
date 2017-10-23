function getAlignedFace_demo()
%%  批处理 规整化人脸
close all
%===================================================================
imsize = [100, 100];
%===================================================================
pwd
% imgName = 'Angelina_Jolie_0017.jpg';
[imgName, PathName] = uigetfile('*.*', 'window title', 'D:\FreqRountine\label_allign_face\pic');
% [imgName, PathName] = uigetfile('*.*', 'window title', fullfile(pwd, '\pic'));

pts = label_pts_demo(fullfile(PathName, imgName));

im = imread(fullfile(PathName, imgName));
%     pts = [ReyeX(lp), ReyeY(lp), LeyeX(lp), LeyeY(lp)];
croppedFace = my_alignment_2points_demo(im, pts);

croppedFace = imresize(croppedFace, imsize, 'bilinear');


figure(1);
subplot(1, 2, 1);
imshow(im);
hold on
plot(pts(1), pts(2), 'g+','LineWidth',4, 'MarkerSize', 15)
plot(pts(3), pts(4), 'g+','LineWidth', 4, 'MarkerSize', 15);
hold off

subplot(1, 2, 2);
imshow(croppedFace);
title('allgned face ');

figure(2);
subplot(1, 2, 1);
imshow(im);
hold on
plot(pts(1), pts(2), 'y+','LineWidth', 3, 'MarkerSize', 13)
plot(pts(3), pts(4), 'y+','LineWidth', 3, 'MarkerSize', 13);
hold off

subplot(1, 2, 2);
imshow(croppedFace);
% title('allgned face ');





