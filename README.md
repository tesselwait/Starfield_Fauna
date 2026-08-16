# Starfield_Fauna

<img width="1920" height="540" alt="Starfield_Fauna" src="https://github.com/user-attachments/assets/0b9a86bc-d8cc-4806-997b-616d65fd772e" />

Image classification dataset:
20,000 (854 x 480) images from 50 fauna species in the video game Starfield.
Images were extracted from video capture.  About 2 minutes of footage was shot in all or most of the species biomes.
One minute of daytime and nighttime footage respectively, usually in two 30-second takes to vary the background.
A PowerShell script is used to establish a frame extract rate and extract the 400 frames plus some extra to replace
 images that were obstructed/blurry or contained other fauna species ignoring birds/critters.
 The shots are for the most part close-up and centered to keep the task focused on discerning between 50 species
 rather than finding the creature in the image.  The images are initially randomized however some normalization
 was done if the ratio of images from some biomes was heavily skewed between the training, validation, and test sets.

480p dataset: (854 x 480) 12gb
https://drive.google.com/drive/u/0/folders/1BZiRsVgp4KEscB0xR_CYZkNkHaOCr1L0

240p dataset: (426 x 240) 3.3gb
https://drive.google.com/drive/u/0/folders/1xQqYR1VHP3TSzQ7bXP3OKtWqcF34q8yz

144p dataset: (256 x 144) 1.3gb
https://drive.google.com/drive/u/0/folders/1VmvCZEBPf_V4Rk5_scXcBNzdrdazi5uT

Temp note: 282-bighorn, 271-bighorn, and 448-bighorn were replaced at 8:20pm for other species in background.
188-lacraia was replaced at 9:35pm, 244-lacraia replaced at 9:55pm, 137-blistercrab replaced at 11:50pm
463-blistercrab replaced at 12:05am
155-cagebrain, 186-cagebrain, 230-cagebrain, 337-cagebrain, 466-cagebrain replaced at 1:10am
