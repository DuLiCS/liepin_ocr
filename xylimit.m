function newpic=xylimit(pic) %function name:XYLIMIT %         Input  pic:binary image %Output newpic:binary image  %��;���Զ�ֵͼ��߽�����޶���Ҫ��ͼ���Ǻڵװ�ͼ %example:  % % pic=imread('�����ַ�.jpg'); % % pic=rgb2gray(pic); % % pic=(pic<127); % % pic=xylimit(pic); % % imshow(pic);    
[m,n]=size(pic);  %%%%����ɨ��%%% 
Ycount=zeros(1,m); 
for i=1:m      
    Ycount(i)=sum(pic(i,:));    %��ȡÿһ�е����ص���� 
end
Ybottom=m;                          %�ײ����� 
Yvalue=Ycount(Ybottom); 
while(Yvalue<3)      
    Ybottom=Ybottom-1;      
    Yvalue=Ycount(Ybottom); 
end
Yceil=1;                                 %�������� 
Yvalue=Ycount(Yceil); 
while(Yvalue<3)     
    Yceil=Yceil+1;      
    Yvalue=Ycount(Yceil); 
end  %%%����ɨ��%%% 
Xcount=zeros(1,n); 
for j=1:n      
    Xcount(j)=sum(pic(:,j));   %��ȡÿһ�е����ص���� 
end

Xleft=1;                                %��ඨ�� 
Xvalue=Xcount(Xleft); 
while(Xvalue<2)     
    Xleft=Xleft+1;      
    Xvalue=Xcount(Xleft); 
end
Xright=n;

Xvalue=Xcount(Xright); 
while(Xvalue<2)     
    Xright=Xright-1;      
    Xvalue=Xcount(Xright); 
end  %%%��ȡͼƬ%%%  
newpic=pic(Yceil:Ybottom,Xleft:Xright); 
