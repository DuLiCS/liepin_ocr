function getPicChar    
%%�����ַ���ȡ��������MATLABƽ̨��ֱ�����м���  
%����MATLAB��UI��ֱ�Ӵ�����Ҫ��ȡ���ַ�ͼƬ���� 
[filename,  pathname,~]=uigetfile({'*.jpg';'*.bmp';'*.png'},'Chose a picture'); 
picstr=[pathname filename]; 
if ~ischar(picstr)    
    return; 
end
pic=imread(picstr); 


%��ͼƬ    
if length(size(pic))==3     
    %�ж�ͼƬ��ά����ͳһΪ�Ҷ�ͼƬ      
    pic=rgb2gray(pic); 
end
pic=(pic<127);      %ת��Ϊ��ֵͼƬ    
pic=xylimit(pic);       %ͼƬ����ĵ�һ�α߽��޶�    %%%%%%%��һ�׶�%%%%%% 
m=size(pic,1);  
Ycount=zeros(1,m); 
for i=1:m      
    Ycount(i)=sum(pic(i,:)); 
end
lenYcount=length(Ycount); 
Yflag=zeros(1,lenYcount); 
for k=1:lenYcount-2      
    if Ycount(k)<3 && Ycount(k+1)<3 && Ycount(k+2)<3         
        Yflag(k)=1;     
    end
end
for k=lenYcount:1+2      
    if Ycount(k)<3 && Ycount(k-1)<3 &&Ycount(k-2)<3         
        Yflag(k)=1;     
    end
end


Yflag2=[0 Yflag(1:end-1)];  
Yflag3=abs(Yflag-Yflag2);  %��������� 
[~,row]=find(Yflag3==1);     %��ͻ��λ�� 
row=[1 row m];                     %����ͻ��λ�õ� 
row1=zeros(1,length(row)/2);  %��ȡͼ�����ʼλ������  
row2=row1;                               %��ȡͼ�����ֹλ������  
for k=1:length(row)      
    if mod(k,2)==1;                     %����Ϊ��ʼ         
        row1((k+1)/2)=row(k);      
    else                                        %ż��Ϊ��ֹ         
        row2(k/2)=row(k);     
    end
end
pic2=pic(row1(1):row2(1),:);  %��ȡ��һ���ַ� 
alpha=1024/size(pic2,2);       %����������� 
pic2=imresize(pic2,alpha);    %������һ���ַ�ͼƬ��С����Ϊ��׼ 
for k=2:length(row)/2      
    pictemp=imresize(pic(row1(k):row2(k),:),[size(pic2,1) size(pic2,2)]);      
    pic2=cat(2,pic2,pictemp);  %��������ͼ��� 
end
pic=xylimit(pic2);    %�޶�ͼ������    %%%%%%%�ڶ��׶�%%%%%% 
[~,n]=size(pic); 
Xcount=zeros(1,n); 
for j=1:n
    Xcount(j)=sum(pic(:,j)); 
end
lenXcount=length(Xcount); 
Xflag=zeros(1,lenXcount); 
for k=1:lenXcount-2      
    if Xcount(k)<3 && Xcount(k+1)<3 && Xcount(k+2)<3         
        Xflag(k)=1;     
    end
end
for k=lenXcount:1+2      
    if Xcount(k)<3 && Xcount(k-1)<3 && Xcount(k-2)<3         
        Xflag(k)=1;     
    end
end
Xflag2=[0 Xflag(1:end-1)]; 
Xflag3=abs(Xflag-Xflag2); 
[~,col]=find(Xflag3==1); 
col=[1 col size(pic,2)];  
coltemp=col(2:end)-col(1:end-1); 
[~,ind]=find(coltemp<3); 
col(ind)=0; 
col(ind+1)=0; 
col=col(col>0);  
col1=zeros(1,length(col)/2); 
col2=col1;  
for k=1:length(col)     
    if mod(k,2)==1          
        col1((k+1)/2)=col(k);     
    else
        col2(k/2)=col(k);     
    end
end
picnum2=length(col)/2; 
piccell2=cell(1,picnum2); 
for k=1:picnum2      
    piccell2{k}=pic(:,col1(k):col2(k));     
    piccell2{k}=xylimit(piccell2{k});      
    piccell2{k}=imresize(piccell2{k},[128 128]); 
end  %��ʾ��ȡ�����ַ���ÿ��������8���ַ� 
if mod(picnum2,8)      
    rownum=ceil(picnum2/8)+1; 
else
    rownum=picnum2/8; 
end

for k=1:picnum2      
    subplot(rownum,8,k);     
    imshow(piccell2{k}); 
end    %%����xylimit���£�  