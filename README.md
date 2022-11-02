# ACROBAT-2022


IHC images provided with this challenge had on average more than 25k x 30k x 3 pixels, which were very large to process on our limited memory computers with GPUs. Thus, due to computation and time limitations, we reduced the dimension of every image to 0.05 of the original dimensions in each direction. 
Then, we selected one H&E-stained tissue image as the reference color, and applied the Macenko stain normalizer method[1] on all of the IHC images to make them more similar to the HE-stained tissue image coloring.

For image registration between two IHC and HE images, We used the ORB detector[2] to find 10000 points of interest for each IHC and HE images, and we registered two images with the ORB descriptors using the BFMatcher(Brute-force matcher using Hamming distance) [3]. Using BFMatcher, we sorted the distance of these matched descriptions and selected the top 30% of best matched points. 

Then, we found the homography matrix H, which is a nonsingular projective matrix that maps the points from a specific IHC image to the corresponding points in the H&E image. This homography matrix is found for each mapping of IHC to H&E images using the matched descriptions in the projective geometry.
In some cases, the estimation of the homography matrix is not accurate, meaning that it maps a point to another point that is not valid (negative position value or bigger than the image size). This issue happens due to the noise in images and the difficulty of mapping. In this case, we do the whole described method again in a higher resolution; for example, instead of resizing to 0.05, we resize the images to 0.1. We keep finding the homography matrix in higher resolutions until we find a feasible homography matrix. 

We tested our model with different numbers for points of interest(PoI) and various resolutions for image registration. Increasing the number of PoI helped to achieve more accurate image registration. We know that doing image registration in higher resolution improves accuracy, but we decreased the image resolution due to computation and time limitations. Among different stain normalizer methods, we picked Macenko because it produced better results on the validation dataset.

Our method is simple and can solve the problem at different resolutions, and it scales well with different computation resources. Moreover, our image registration is completely automated and works in real time.

ORB detector is less robust on image transformations(rotation, translation and scaling). However, other methods, such as SIFT and SURF, can be used because they are more robust but require more computational time. We think with more time and computation resources, this model can be improved.



Citations
[1] Macenko, Marc, et al. "A method for normalizing histology slides for quantitative analysis." 2009 IEEE international symposium on biomedical imaging: from nano to macro. IEEE, 2009.  

[2] Rublee, Ethan, et al. "ORB: An efficient alternative to SIFT or SURF." 2011 International conference on computer vision. Ieee, 2011.

[3] Jakubović, Amila, and Jasmin Velagić. "Image feature matching and object detection using brute-force matchers." 2018 International Symposium ELMAR. IEEE, 2018.
