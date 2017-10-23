
reqToolboxes = {'Computer Vision System Toolbox', 'Image Processing Toolbox'};
if( ~checkToolboxes(reqToolboxes) )
 error('detectFaceParts requires: Computer Vision System Toolbox and Image Processing Toolbox. Please install these toolboxes.');
end

for i=1:40
      datadir=['att_faces/orl_faces/s',int2str(i),'/'];
      filenames=[datadir,'*.pgm'];

      files=dir(filenames);
      
      for j=1:length(files)
        img=imread([datadir,files(j).name]);
        name = strsplit(files(j).name, '.');
        
        if length(size(img)) == 2 
            tmp = ones(112, 92, 3, 'uint8');
            tmp(:, :, 1) = img;
            tmp(:, :, 2) = img;
            tmp(:, :, 3) = img;
            img = tmp;
        end
        detector = buildDetector();
        [bbox, ~, ~, bbfaces] = detectFaceParts(detector,img,2);
        
        dir_ = ['s', int2str(i), '/'];
        if ~exist(dir_)
            mkdir(dir_)
        end
        path = ['s', int2str(i), '/', name{1}, '.jpg'];
        if size(bbox, 1) == 1
            lx = bbox(5) + bbox(7) / 2;
            ly = bbox(6) + bbox(8) / 2;
            rx = bbox(9) + bbox(11) / 2;
            ry = bbox(10) + bbox(12) / 2;

            pts = round([lx, ly rx, ry]);
            Face = my_alignment_2points_demo(img, pts);
            size(Face)
            imwrite(Face, path);
            
        end
      end
end


