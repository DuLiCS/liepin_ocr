Rearrangement_img = imread('b_9.png');

Rearrangement_img = rgb2gray(Rearrangement_img);

level = graythresh(Rearrangement_img)

Rearrangement_img = im2bw(Rearrangement_img,0.1);

%Rearrangement_img = double(Rearrangement_img)


%Rearrangement_img(Rearrangement_img == 1) = 255

imwrite(Rearrangement_img,'b9.png');


