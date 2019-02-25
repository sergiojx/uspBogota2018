# Urban Safety Perception Bogota 2018
## Image set [imgset1-11.zip]
5505 street images of the Chapinero locality
## Indexed actual vote image pair annotations
descriptorIndexer_Jul_0518.txt <br/>
Indexing starts at 1 not at 0 !! <br/>
18959 annotations 
## Visual survey published image name list
imgnames.txt
## Visual survey published image feature vector files
### cielabA, gist and hog features.zip
1. cielabA2.txt
2. cielabB2.txt
3. cielabL2.txt
4. gistSet.txt
5. hogSet.txt
### VGG19_features.zip
1. VGG19Chapinero_Ftrs.csv
## notebooks
### transfer4uspVGG16FtrExtr.ipynb
Used for image VGG19 based feature extraction
### randomVoteSchemeIII.ipynb and randomVoteSchemeIV.ipynb
Used to generate synthetic vote image pairs.
### transfer4uspVGG16SoftMaxNonEQU.ipynb
Training notebook

### transfer4uspVGG16NonEQUVerify_Jul_0518.zip
Tensor Flow model
### trueskillImgScoreShmIII.ipynb
Top 40 image rating visualization
### True Skill based predictors
#### VGG19ChapineroTSkillPredictor.py
#### VGG19MartiresTSkillPredictor.py
#### VGG19UsaquenTSkillPredictor.py

## Papers
* Acosta, S., Camargo, J. *City safety perception model based on visual
content of street images*. IEEE IV International Smart Cities Conference
(ISC2), 2018.

* Acosta, S., Camargo, J. *Predicting city safety perception based on visual image content*. 
23rd Iberoamerican Congress on Pattern Recognition
(CIARP), 2018.
```
@InProceedings{Acosta:2018:PSP,
  author      = "Sergio Acosta and Jorge Camargo",
  title       = "Predicting city safety perception based on visual image content",
  booktitle   = "CIARP 2018, LNCS 11401",
  year        = "2019",
}
```
