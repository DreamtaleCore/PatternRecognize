I = double(imread('/home/ros/ws/algorithm/PatternRecognize/data/ORL_face_dataset/ORL92112/bmp/s1/1.BMP'));
X = reshape(I, size(I,1)*size(I,2), 1);
coeff = pca(X);
Itransformed = X*coeff;
Ipc1 = reshape(Itransformed(:,1),size(I,1),size(I,2));
figure, imshow(Ipc1,[]);