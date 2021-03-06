# GAN-Balancing-Datasets
COSC 490 - Towson University 

Insurehub page   
Using KDD99 Dataset http://kdd.ics.uci.edu/databases/kddcup99/

Members:
- Tim Merino
- Matt Stillwell
- Max Coplan
- Mark Steele
- Ksenia Tepliakova
- Alex Stoyanov
- Jon Patton
- Long Chen

## Description
Our goal is to build a GAN network with discriminator function usinginformation from the previous research in this field and earlier work at ArgonneNational Laboratory using suitable datasets. We have also found publicly availablecyber attack data sets that can be used to supplement our GAN training. Currentcyber attack data is unbalanced, outdated, and narrowly scoped. This can lead tocyber attacks not being accurately identified, leading to security breaches.Bytraining a GAN using the limited data sets available online, our Generator neuralnetwork model would be able to replicate and expand existing datasets and teach other discriminator algorithms using our data. Our Discriminator algorithm couldalso recognize cyber attacks based on incoming traffic and reduce false positivesthat plague current detection systems.


## Specific Aims
Our primary goal is to use a Generative Adversarial Network in order to balancedatasets that will ultimately be used to better train classifiers. First, we will train ourclassifier on a subset of our dataset. Since we are using a supervised learning approach witha labeled dataset, we will be able to accurately measure the success of our discriminator.Next, we will train our generator in an adversarial relationship with the discriminator. ​Wewill have accomplished our goal when the generator can produce data that the classifiercannot reliably distinguish from real data. ​Finally, once our generator has a high successrate, we will use it to generate attack data in order to balance our original dataset. Our finalmeasure of success is how well the classifier performs with the newly balanced dataset. Wewill explicitly compare both the accuracy of the classifier trained on the original datasetand the classifier trained on the balanced dataset, as well as the false positive rate of thetwo networks. ​We can also test our generated data against other classification approachesto see if it is classified as cyber attack data in non-GAN classifications. By comparingperformance of algorithms trained with unbalanced datasets versus trained with databalanced by our generated data, we can see real world applications of our data. From theseresults, we will be able to better understand cyber attack data and have a viable method forbalancing large datasets.
