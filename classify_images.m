function [classification_error,outputData,class,inTrain,inTest,numClasses]=classify_images(task,sigmaLGN,data,class,inTrain,inTest,imDims,numClasses,plotPolarity)
%Apply 2-stage hierarchical PC/BC-DIM to classifying images
%Use clustering to learn weights for first processing stage
%Define second processing stage weights as the summed 1st layer response to all members of each category

if nargin<1 || isempty(task)
  task='USPS';%   'MNIST';  % 'YALE4';%
end

%set parameters
if nargin<2 || isempty(sigmaLGN)
  sigmaLGN=16
end
patchClustering='agglom';%'exemplars'; %'dim'; %
ONOFF=1;
figoff=0;

%LOAD DATA.
if nargin<3
  [data,class,inTrain,inTest,imDims,numClasses,plotPolarity]=load_classify_image_dataset(task);
end
%CLUSTER TRAINING IMAGES TO FORM DICTIONARY
Xtrain=data(inTrain,:)'; % data:400 * 10304
switch patchClustering
  case 'dim'
    n=5000;
    skipTraining=0;
    filename=['classify_images_dim_weights',task,'_n',int2str(n),'.mat'];
    if skipTraining
      load(filename);
    else
      beta=0.1
      alpha=1
      batchLen=1 %length(inTrain)    %number of training patterns used in each learning batch
      cycs=fix(40*max(n,length(inTrain))/sqrt(batchLen)) %number of training cycles 
      show=fix(1e4/sqrt(batchLen));   %how often to plot receptive field data
      [W,V]=weight_initialisation_random(n,size(Xtrain,1));
      max_y=0;sum_nse=0;sum_sparsity=0;
      for cyc=1:cycs
        %choose a batch of input stimuli to train on
        order=randperm(length(inTrain));
        %calculate responses to training data
        [Y,E]=dim_activation(W,Xtrain(:,order(1:batchLen)),V);
        %update weights
        [W,V]=dim_learn(W,V,Y,E,beta,alpha);
        
        %record data
        max_y=max([max_y,max(max(Y))]);
        sum_nse=sum_nse+measure_nse(Xtrain(:,order(1:batchLen)),V'*Y)./batchLen; 
        sum_sparsity=sum_sparsity+mean(measure_sparsity_hoyer(Y));
        if rem(cyc,show)==0 || cyc==cycs 
          fprintf(1,'.%i.',cyc); 
          disp([' ymax=',num2str(max_y,3),...
                ' NSE=',num2str(sum_nse./show,3),...
                ' Sparsity=',num2str(sum_sparsity./show,3)]);
          max_y=0;sum_nse=0;sum_sparsity=0; 
        end
        
      end
      %save(filename, 'W', 'V');
    end
    
  case 'exemplars'
    V=Xtrain';
    n=length(inTrain);
    
  case 'agglomAllPatches'
    numClusterMembersReqd=0
    similarityThres=0.85

    clusterIndex = clusterdata(Xtrain','cutoff',1-similarityThres,'criterion','distance','linkage','complete','distance',@distance_measure);%note clustering is based on distance, not similarity, so need to use 1-similarity threshold

    numClusters=max(clusterIndex);
    k=0;
    for i=1:numClusters
      clustInd=find(clusterIndex==i);
      if length(clustInd)>numClusterMembersReqd
        k=k+1;
        V(k,:)=mean(Xtrain(:,clustInd),2); %each cluster is mean of all members
      end
    end
    n=k;
    
  case 'agglom'
    numClusterMembersReqd=0
    similarityThres=0.9

    k=0;
    for c=1:numClasses
      classInd=find(class(inTrain)==c);
      clusterIndex = clusterdata(Xtrain(:,classInd)','cutoff',1-similarityThres,'criterion','distance','linkage','complete','distance',@distance_measure);%note clustering is based on distance, not similarity, so need to use 1-similarity threshold

      numClusters(c)=max(clusterIndex);
      for i=1:numClusters(c)
        clustInd=find(clusterIndex==i);
        if length(clustInd)>numClusterMembersReqd
          k=k+1;
          V(k,:)=mean(Xtrain(:,classInd(clustInd)),2); %each cluster is mean of all members
        end
      end
    end
    numClusters
    n=k;
end
toPlot=randperm(n);
toPlot=toPlot([1:min(48,n)]);
if exist('W'); 
  figured(figoff+1),clf, 
  plot_weights(W,toPlot,imDims,plotPolarity);
end
figured(figoff+2),clf, 
plot_weights(V,toPlot,imDims,plotPolarity);
drawnow
print('-dpdf',[task,'_dictionary.pdf'])

n=size(V,1)

if ONOFF
  %CONVERT DICTIONARY ENTRIES TO ON/OFF WEIGHTS
  V=imnorm_batch(V',imDims,sigmaLGN)';

  %preprocess input images
  filename=['classify_images_imnormed_data_',task,'_sigmaLGN',num2str(sigmaLGN),'.mat'];
  data=imnorm_batch(data',imDims,sigmaLGN)';
  %save(filename, 'data');
  imDims=[imDims,2];
end
%normalise weights
W=bsxfun(@rdivide,V,max(1e-6,sum(V,2)));
V=bsxfun(@rdivide,V,max(1e-6,max(V,[],2)));
%recale each input to range from 0 to 1
data=bsxfun(@minus,data,min(data')');
data=bsxfun(@rdivide,data,max(data')');

figured(figoff+3),clf, 

plot_weights(V,toPlot,imDims,plotPolarity);
drawnow
print('-dpdf',[task,'_stage1_weights.pdf'])
figured(figoff+4),clf, 
plot_weights(data,toPlot,imDims,plotPolarity);
drawnow

%MATCH DICTIONARY PATCHES TO TRAINING IMAGES using DIM
outputData=dim_activation(W,data',V)';
%count the number of votes corresponding to each cluster
threshold=1e-3;
outputData(outputData<threshold)=0;
for c=1:numClasses
  ind=class(inTrain)==c;
  Wvotes(c,:)=sum(outputData(inTrain(ind),:));
end
Wvotes=bsxfun(@rdivide,Wvotes,max(1e-6,sum(Wvotes,1))); %would have no effect when using convolution
Vvotes=bsxfun(@rdivide,Wvotes,max(1e-6,max(Wvotes,[],2)));  %std dim as used in ism
%Vvotes=bsxfun(@rdivide,Wvotes,max(1e-6,max(Wvotes,[],1))); %slightly worse
figured(figoff+5),clf, imagesc(Wvotes); colorbar
figured(figoff+6),clf, imagesc(Vvotes); colorbar


%MATCH DICTIONARY PATCHES TO TESTING IMAGES using DIM
XclassifierTest=outputData(inTest,:)';
[Yclassifier]=dim_activation(Wvotes,XclassifierTest,Vvotes);
threshold=1e-3;
Yclassifier(Yclassifier<threshold)=0;
%calculate classification error
[~,classPredicted]=max(Yclassifier);
classification_error=100*sum(classPredicted~=class(inTest))./length(inTest)  

%plot example responses for a few test images
toPlot=randperm(length(inTest));
toPlot=toPlot(1:10);
fig=figoff+10;
for k=toPlot
  fig=fig+1;
  figured(fig), clf
  plot_network(data(inTest(k),:),outputData(inTest(k),:),Yclassifier(:,k),imDims,V,plotPolarity);
  print('-dpdf',[task,'_response_example',int2str(fig),'.pdf'])
end
%show which images from original test dataset were mis-classified
[data,class,inTrain,inTest,imDims]=load_classify_image_dataset(task);
figured(figoff+21),clf,plot_misclassified(classPredicted,class(inTest),data(inTest,:),imDims(1:2),plotPolarity,1);
print('-dpdf',[task,'_misclassified.pdf'])
    


function plot_network(input,Y1,Y2,imDims,Wvis,plotPolarity)

subplot(3,5,13); 
if length(imDims)==3
  plot_image(diff(reshape(input,imDims),1,3)');
elseif length(imDims)==2
  plot_image(reshape(input,imDims)');
end

cmap=colormap('gray');if plotPolarity>0, cmap=1-cmap;colormap(cmap); end

axProp=subplot(3,1,2);
top=max(0.01,1.05.*max(Y1));
width=max(1,length(Y1)/2000);
bar([1:length(Y1)]-1,Y1,width,'FaceColor','r','EdgeColor','r','LineWidth',1);
axis([0.5,length(Y1)+0.5,0,top])

ax=axProp.Position;
[m,ind]=sort(Y1.*(1+0.001.*rand(size(Y1))),'descend');
numToLabel=min(25,length(find(m>0.25*m(1))));
for i=1:numToLabel, 
  axes('Position',[ax(1)+(ax(3).*(ind(i)-1)./length(Y1))-0.025,ax(2)+ax(4)*min(1,m(i)/top),0.06,0.06])
  if length(imDims)==3
    plot_image(diff(reshape(Wvis(ind(i),:),imDims),1,3)');
  elseif length(imDims)==2
    plot_image(reshape(Wvis(ind(i),:),imDims)');
  end
end

subplot(3,3,2); 
top=max(0.01,1.05.*max(Y2)); 
width=max(1,length(Y2)/2000);
bar([1:length(Y2)]-1,Y2,width,'FaceColor','r','EdgeColor','r');
axis([-0.5,length(Y2)-0.5,0,top])

set(gcf,'PaperSize',[18 10],'PaperPosition',[0 0 18 10],'PaperOrientation','Portrait');
drawnow;




function [X]=imnorm_batch(I,imDims,sigma,gain,leavepadded)
if nargin<3, sigma=[]; end
if nargin<4, gain=[]; end
if nargin<5, leavepadded=[]; end

[a,batchLen]=size(I);
I=single(I);

X=zeros(2*a,batchLen,'single');
for t=1:batchLen 
  It=reshape(I(:,t),imDims);
  
  [~,~,Xon,Xoff]=imnorm(It,sigma,gain,leavepadded);
  
  X(:,t)=[Xon(:);Xoff(:)];
end

