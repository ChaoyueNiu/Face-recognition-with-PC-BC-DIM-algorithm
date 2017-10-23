function [data,class,inTrain,inTest,imDims,numClasses,plotPolarity]=load_classify_image_dataset(task);
switch task
  case 'USPS'
    load('../../Data/usps.mat');
    imDims=[16,16];
    fixedSplit=7291;
    class=data(:,257)';
    data=data(:,1:256);
    data=data-min(min(data));
    data=data./max(max(data));
    plotPolarity=-1;
    
  case 'USPSaffine'
    filename='usps_affine.mat';
    if ~exist(filename,'file')
      load('../../Data/usps.mat');
      imDims=[16,16];
      fixedSplit=7291;
      class=data(:,257)';
      data=data(:,1:256);
      data=data-min(min(data));
      data=data./max(max(data));
      [data,class,fixedSplit]=expand_affine_image_set(data,class,fixedSplit,imDims);
      save(filename,'imDims','fixedSplit','class','data');
    else
      load(filename);
    end
    plotPolarity=-1;
  
  case {'MNIST1000','MNIST2000','MNIST4000','MNIST5000'}
    imDims=[28,28];
    if strcmp(task,'MNIST1000') == 1
      trainingImagesPerDigit=1000;
    elseif strcmp(task,'MNIST2000') == 1
      trainingImagesPerDigit=2000;
    elseif strcmp(task,'MNIST4000') == 1
      trainingImagesPerDigit=4000;
    elseif strcmp(task,'MNIST5000') == 1
      trainingImagesPerDigit=5000;
    end
    data=[];class=[];
    for i=1:10
      %if i>=9, trainingImagesPerDigit=trainingImagesPerDigit*2, end
      load(['../../Data/LeCun-MNIST_handwrittendigits/digit',int2str(i-1)]);
      class=[class,i*ones(1,trainingImagesPerDigit)];
      data=[data;D(1:trainingImagesPerDigit,:)];
    end
    fixedSplit=size(data,1);
    for i=1:10, 
      load(['../../Data/LeCun-MNIST_handwrittendigits/test',int2str(i-1)]); 
      class=[class,i*ones(1,size(D,1))];
      data=[data;D];
    end
    plotPolarity=-1;
  
  case 'MNIST'
    imDims=[28,28];
    data=[];class=[];
    for i=1:10
      load(['../../Data/LeCun-MNIST_handwrittendigits/digit',int2str(i-1)]);
      numImagesPerDigit=size(D,1);
      class=[class,i*ones(1,numImagesPerDigit)];
      data=[data;D];
    end
    fixedSplit=size(data,1);
    for i=1:10, 
      load(['../../Data/LeCun-MNIST_handwrittendigits/test',int2str(i-1)]); 
      numImagesPerDigit=size(D,1);
      class=[class,i*ones(1,numImagesPerDigit)];
      data=[data;D];
    end
    plotPolarity=-1;
  
  case {'YALE1','YALE2','YALE4','YALE8'}
    if strcmp(task,'YALE1') == 1
      imDims=[24,21].*1;
    elseif strcmp(task,'YALE2') == 1
      imDims=[24,21].*2;
    elseif strcmp(task,'YALE4') == 1
      imDims=[24,21].*4;
    elseif strcmp(task,'YALE8') == 1
      imDims=[24,21].*8;
    end
    subjNum=[1:13,15:39];
    datatrain=[];datatest=[];
    classtrain=[];classtest=[];
    for i=subjNum
      if i<10
        datadir=['CroppedYale/yaleB0',int2str(i),'/'];
        filenames=[datadir,'yaleB0',int2str(i),'_P00A*.pgm'];
      else
        datadir=['CroppedYale/yaleB',int2str(i),'/'];
        filenames=[datadir,'yaleB',int2str(i),'_P00A*.pgm'];
      end
      files=dir(filenames);
            
      for j=1:length(files)
        I=imread([datadir,files(j).name]);
        x=single(im2double(imresize(I,imDims,'bilinear')))';
        if isodd(j)
          datatrain=[datatrain,x(:)];
          classtrain=[classtrain,i];
        else
          datatest=[datatest,x(:)];
          classtest=[classtest,i];
        end
      end
    end
    fixedSplit=size(datatrain,2);
    data=[datatrain,datatest]';
    class=[classtrain,classtest];
 
    imDims=fliplr(imDims);
    plotPolarity=1;
    
  case 'ORL'
      
    imDims=[112, 92];
    subjNum = [1:40];  % subjNum = [1:5, 10:20]
    datatrain=[];datatest=[];
    classtrain=[];classtest=[];
    for i=subjNum

      datadir=['att_faces/orl_faces/s',int2str(i),'/'];
      filenames=[datadir,'*.pgm'];

      files=dir(filenames);
      for j=1:length(files)
        I=imread([datadir,files(j).name]);
        x=single(im2double(imresize(I,imDims,'bilinear')))';
        if isodd(j)
          datatrain=[datatrain,x(:)];
          classtrain=[classtrain,i];
        else
          datatest=[datatest,x(:)];
          classtest=[classtest,i];
        end
      end
    end
    fixedSplit=size(datatrain,2); %???úY’í’ó
    data=[datatrain,datatest]';
    class=[classtrain,classtest]; % half train, half test
    
    imDims=fliplr(imDims);
    plotPolarity=1;
    
  case 'NORB'
    numImages=6000 %24300;

    datafile='../../Data/SmallNORB/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat';
    catfile='../../Data/SmallNORB/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat';
    [data,classTrain]=read_norb_data(datafile,catfile,numImages);
    fixedSplit=numImages;
    datafile='../../Data/SmallNORB/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat';
    catfile='../../Data/SmallNORB/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat';
    [test,classTest]=read_norb_data(datafile,catfile,numImages);
    data=[data,test];
    class=[classTrain,classTest];
    plotPolarity=1;
  
  case 'NORBreduced'
    numImages=6000 %24300;

    load('../../Data/SmallNORB/smallnorb-5x46789x9x18x6x2x32x32-training-dat.mat');
    dataTrain=data(:,1:numImages);
    classTrain=class(1:numImages);
    fixedSplit=numImages;
    load('../../Data/SmallNORB/smallnorb-5x46789x9x18x6x2x32x32-testing-dat.mat');
    dataTest=data(:,1:numImages);
    classTest=class(1:numImages);
    data=[dataTrain,dataTest];
    class=[classTrain,classTest];
    plotPolarity=1;

  case 'CIFAR10'
    alldata=[]; class=[];
    for i=1:5
      load(['../../Data/cifar-10/data_batch_',int2str(i)]);
      data=convert_to_grey(data,[32,32]);
      alldata=[alldata,data'];
      class=[class,labels'];
    end
    fixedSplit=size(alldata,2);
    load('../../Data/cifar-10/test_batch');
    data=convert_to_grey(data,[32,32]);
    alldata=[alldata,data'];
    class=[class,labels'];
    data=alldata;
    class=double(class);
    plotPolarity=1;
  
  otherwise
    disp('ERROR: unknown data set');

end
numExemplars=length(class);
numExemplars
if exist('fixedSplit')
  inTrain=1:fixedSplit;
  inTest=fixedSplit+1:numExemplars;    
end
%ensure class labels are sequentially numbered starting from 1
class=class+max(0,1-min(class));
k=0;
for c=unique(class)
  k=k+1;
  class(class==c)=k;
end
numClasses=k;
numClasses
class=single(class);
data=single(data);
