## Experiments



- Have a separate ID for a scene-target pair 
- Each of the 10 target images for an object are paired separately with the same scene image
- Currently keeping `0048` video as train and `0050` as test 

Testing
- Given a new object and a new scene
- Provided only one target image


#### Primary expts. 


1.	Ablations (w/ 1 scene-target pair)
	  - Feature loss
	  - Bounding box regression loss
	  - Network architecture (using multi-scale features from FPN)

#### Seconday expts.

1. 	single pair
	  - Train : 1 scene image - 1 target image
	  - Test  : Same as train

2.	multi-pair
	  - Train : 1 scene - 10 target images (averaged features)
	  - Test  : 1 scene - 1 target image
3. 	zero shot w/ common objects in scene
	  - Test with `0049` video as it has common objects with `0048`
