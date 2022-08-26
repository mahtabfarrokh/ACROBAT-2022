# ACROBAT-2022


The method that is used here is to first make IHC files similar to HE by using a HE stained tissue image as reference color and then change the IHC color setting. Then, using a orb detector, I find area(points) of interest. Using the found set of points, I find the homography matrix H. 
It happens that the found H matrix is wrong, and gives negetive values, in this case, I keep generating matrix H in different resolutions till I get a valid matrix H. 
