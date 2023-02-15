# MorphologicalOperations
Solve 3 part of Morphological and Other Set Operations problems

# Part A: Morphological and Other Set Operations
At first threshold the image

Dilation	
Then create a padded image with zeros
Then pixel by pixel iterate kernel over image and every where they match we place white pixel there with that kernel type and I use both cross and diamond kernel 3x3

Erodation 
Then create a padded image with zeros
Then pixel by pixel iterate kernel over image and every where they match we place black pixel there with that kernel type and I use both cross and diamond kernel 3x3

![image](https://user-images.githubusercontent.com/24508376/219142004-24c1bc02-b570-4e2e-a5ab-5ffa861ab45e.png)
![image](https://user-images.githubusercontent.com/24508376/219142020-cebe91c1-7509-4608-94bf-a8ae286186ac.png)


# Part B: Boundary Extraction

At first I erode the image
And then subtract the original image from erdoed image

![image](https://user-images.githubusercontent.com/24508376/219142269-5893aecb-7d31-4215-ba52-7f0324802be6.png)


# Part C: Connected Components
Befor I start extracting component I erode image once with a huge kernel (11,11)

Then I pad the image then search over the whole of image and find every where is not zero then I create a moving kernel that goes through  all image and where it conflict with bunch of pixels that valued 1, save the locations and patch label to them
Then goes through that way and when it meet a pixel again, patch last patched label to it again and there we go through all image and labeling and at the end I show all of the images it patched label and it is 18

![image](https://user-images.githubusercontent.com/24508376/219142676-5fac4802-77aa-40e5-9dbc-b43f53f1232c.png)
