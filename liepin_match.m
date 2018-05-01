clear
clc
Rearrangement_img = imread('b_1.png');

Rearrangement_img = rgb2gray(Rearrangement_img);

Rearrangement_verify = imread('s_3.png')




Rearrangement_verify = rgb2gray(Rearrangement_verify)

First_char = Rearrangement_verify(7:28,6:25);

Second_char = Rearrangement_verify(7:28,25:42);

Third_char = Rearrangement_verify(7:28,42:61);

Fourth_char = Rearrangement_verify(7:28,61:79);

Char_seq = {First_char,Second_char,Third_char,Fourth_char};

imwrite(First_char,'s_10.png')

imwrite(Second_char,'s_11.png')

imwrite(Third_char,'s_12.png')

imwrite(Fourth_char,'s_13.png')

Rearrangement_img = im2bw(Rearrangement_img,0.1);

Rearrangement_verify = im2bw(Rearrangement_verify,0.3);

First_char = Rearrangement_verify(7:28,6:25);

Second_char = Rearrangement_verify(7:28,25:42);

Third_char = Rearrangement_verify(7:28,42:61);

Fourth_char = Rearrangement_verify(7:28,61:79);

Char_seq = {First_char,Second_char,Third_char,Fourth_char};





for k = 1:4
    

reource_p=Rearrangement_img;
reource_p_sub=Char_seq{k};  
[m,n]=size(reource_p);  
[m0,n0]=size(reource_p_sub);  
result=zeros(m-m0+1,n-n0+1);  
vec_sub = double( reource_p_sub(:) );  
norm_sub = norm( vec_sub );  
for i=1:m-m0+1  
    for j=1:n-n0+1  
        subMatr=reource_p(i:i+m0-1,j:j+n0-1);  
        vec=double( subMatr(:) );  
        result(i,j)=vec'*vec_sub / (norm(vec)*norm_sub+eps);  
    end  
end  
%找到最大相关位置  
[iMaxPos,jMaxPos]=find( result==max( result(:)));  
figure,  
subplot(121);imshow(reource_p_sub),title('匹配模板子图像');  
subplot(122); 
imshow(reource_p);  
title('标记出匹配区域的原图'),  
hold on  
plot(jMaxPos,iMaxPos,'*');%绘制最大相关点  
 %用矩形框标记出匹配区域  
plot([jMaxPos,jMaxPos+n0-1],[iMaxPos,iMaxPos]);  
plot([jMaxPos+n0-1,jMaxPos+n0-1],[iMaxPos,iMaxPos+m0-1]);  
plot([jMaxPos,jMaxPos+n0-1],[iMaxPos+m0-1,iMaxPos+m0-1]);  
plot([jMaxPos,jMaxPos],[iMaxPos,iMaxPos+m0-1]);  

end
