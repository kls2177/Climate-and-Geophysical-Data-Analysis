Signal-to-Noise
=======================

You are probably wondering how the $z$-score applies to statistical hypothesis testing. So, let's first take a look at what we mean by *hypothesis testing*.

## Hypothesis Testing

Thus far, we have covered how to determine the probability of getting a value $x_i$ within a range [$a$, $b$].

For instance, using the ENSO example, we have looked at how we use the $z$-score to address questions like,

> "What is the probability that the ENSO index in the winter of 1997-1998 was its value or greater?"

However, in climate and geophysical sciences (and most other fields of science for that matter), we tend to be more interested in establishing the probability of the difference between a sample mean and an underlying population or the difference between two samples. In other words, we are interested in finding a signal in our data that is different from the noise.

More generally, the [*frequentist*][freq] approach to hypothesis testing involves testing a *null* hypothesis by comparing the sample data you observe in your laboratory experiment, field measurements, computer simulation, etc. with the predictions of a null hypothesis. You estimate what the probability would be of obtaining the observed results, or something more extreme, if the null hypothesis were true. If this estimated probability (the $p$-value) is small enough (below the significance value), then you conclude that it is unlikely that the null hypothesis is true; you *reject* the null hypothesis and *accept* an alternative hypothesis.

Returning to our ENSO example, we instead might ask,

> “Was the ENSO index from 2003-2013 consistent with climatological behaviour?”

The framing of the above question implies that the null hypothesis is:
- the ENSO index from 2003-2013 (the sample mean) is consistent with "climatological behaviour" (the population mean).

The alternative hypothesis is thus:
- the ENSO index from 2003-2013 (the sample mean) is not consistent with "climatolgoical behaviour" (the population mean).

### The Null Hypothesis

Proper construction of the null hypothesis and its alternative is critical to the meaning of statistical hypothesis testing.

The null hypothesis is a statement of the conventional wisdom or the "status quo", i.e. the null hypothesis is that things are the same as each other, or the same as a theoretical expectation. Its alternative is an interesting conclusion that follows directly from the rejection of the null hypothesis. The alternative hypothesis is that things are different from each other, or different from a theoretical expectation.

Simply put, the null hypothesis states that there is no signal in the sample data, just noise, while the alternative hypothesis is that there is a signal.

Here are some examples of null ($H_0$) and alternative ($H_1$) hypotheses:

> $H_0$: The means of two samples are equal.
> $H_1$: The means of two samples are not equal.

> $H_0$: The anomaly is zero.
> $H_1$: The anomaly is not zero.

> $H_0$: The correlation coefficient is zero.
> $H_1$: The correlation coefficient is not zero.

Now, that we are familiar with the concept of hypothesis testing, we can now examine how we use the $z$-score to reject or accept a null hypothesis.




[freq]: https://365datascience.com/bayesian-vs-frequentist-approach/
