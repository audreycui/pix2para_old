# pix2para
From Pixel to Paragraph: A Deep Artwork Analysis Paragraph Generator

### Introduction
Art influences society by shaping our perspective and influencing our sense of self. By portraying subject matter with deliberate composition, color scheme, and other stylistic choices, the artist forms emotional connections between artwork and viewer and communicates their vision with the world. The goal of my project is to develop an artificial neural network system that interprets input artwork and generates a passage that describes the objects and other features (ex. color palette) present in the artwork as well as the ideas and emotions the artwork conveys.

### Approach
I am developing a GAN (generative adversarial network) that conditions on the artwork object features and stylistic features to generate the analysis paragraphs. 

I have used and modified code from the following repositories: 
* [SeqGAN-Tensorflow](https://github.com/audreycui/SeqGAN-Tensorflow)
* [Show_and_Tell](https://github.com/audreycui/Show_and_Tell)
* [img2poem](https://github.com/audreycui/img2poem)
* [Conditional-GAN](https://github.com/zhangqianhui/Conditional-GAN)

I used Show_and_Tell, a seq2seq image captioning model, as a starting point for my model framework. I have modified Show_and_Tell's decoder to be modeled after img2poem's generator so that it can be trained on a reward function. I added the discriminator from SeqGAN, which is an unconditional text generation GAN. To make the discriminator a conditional model, I referenced code from Conditional-GAN, which is a conditional image generation GAN. 

I wrote scripts for scraping artwork images and their corresponding analysis paragraphs from [theartstory](theartstory.com) and [Smithsonian American Art Museum](https://americanart.si.edu/). 

More details about my modifications to existing code can be found as comments in the files.  

### TODO
* Finish debugging current image to text GAN model 
* Add data loading and processing methods specific for artwork and artwork analysis paragraphs (current model uses Microsoft COCO image-caption dataset; current data loading/processing methods are borrowed from Show_and_Tell)
* Add discriminators that distinguish between artwork analyis vs. normal image caption, relevant to artwork vs irrelevant to artwork 
