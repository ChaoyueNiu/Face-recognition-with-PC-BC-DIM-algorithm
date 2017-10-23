function F = my_alignment_2points_demo(im, pts)
%%im is the image to be transfromed, pts is the coordinate of landmarks in the input image,
%with the form [x1 y1 x2 y2]
%
% inSize=[130 150]; 
% %% this routinue use two landmark
% outLE = [30 45];
% outRE = [100 45]; % 

% inter_ocular = norm([pts(1:2) - pts(3:4)]);
% inSize = [130*(inter_ocular/70), 150*(inter_ocular/70)];
% inSize = round(inSize);
% outLE = [30*(inter_ocular/70), 45*(inter_ocular/70)];
% outLE = round(outLE);
% outRE = [100*(inter_ocular/70), 45*(inter_ocular/70)];
% outRE = round(outRE);

inSize = [64, 64];
outLE = [16, 20];
outRE = [48, 20];
F = f_crop_face_region_2points1(im, pts, [outLE outRE], inSize);

