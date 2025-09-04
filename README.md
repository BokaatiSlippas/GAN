# GAN

Notes: 

Generator -> produces fake data
Discriminator -> produces proability of realness

Input noise into generator which produce fake sample (same dimensions as real data). This is called G(z), real data is called x

Discriminator takes in data and outputs probability of how real the data is. Therefore D(G(z)) should be close to 0 if discriminator is amazing and generator is bad. D(x) should in this case be close to 1

Cost function : V(D,G) = E[log(D(x))] + E[log(1-D(G(z)))]
Discriminator perspective: D(x) should be close to 1 therefore first E close to 0 and second E also close to 0 (THEREFORE : discriminator wants to maximise cost towards 0)
Generator perspective: don't care about D(x), D(G(z)) should be close to 1 therefore second E close to negative infinity (THEREFORE : generator wants to minimise cost to negative infinity)

Overall conclusion minG maxD of V(D,G) = E[logD(x)] + E[log(1-D(G(Z)))]

for i in range(epochs)
  for j in range(k)
    sample m noise samples to produce m Generator samples, i.e. G(z0), G(z1), ..., G(zm) label each as 0
    sample m real samples, i.e. x1, x2, ..., xm label each as 1
    Calculate gradient of V with respect to Discriminator parameters and update Discriminator parameters in direction that maximises V
  sample m noise samples to produce m Generator samples (DONT NEED REAL SAMPLES BECAUSE AFTER GRADIENT X DISAPPEARS)
  Calculate gradient of V with respect to Generator parameters and update Generator parameters in direction that minimises V
