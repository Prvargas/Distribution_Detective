# Distribution_Detective
A project that can classify the distribution of data.


# Motivation

- After working on several dozen data science projects and case studies I noticed there is **NO** tool that explicitly classifies probability distributions. 
- I am an enthusiastic problem solver, so I set out to fix this.



# Approach

- I wanted to create a tool that can classify the probability distribution of any dataset feature.
- Human beings typically look at a histogram images to determine distributions, so I decided my model should too.
- I am a data scientist, therefore I decided a mathematical confirmation for the classification is also needed.
- The Chi Square Test is used to mathematically confirm ditribution classifications.


<!-- #region -->
# Probability Distributions

![image.png](attachment:image.png)


I chose to use 5 distributions but there are many more.


**5 Distributions:**
1. Bernoulli
2. Uniform
3. Poisson
4. Exponential
5. Normal (Gaussian)


<!-- #endregion -->

<!-- #region -->
# Algorithm


![image.png](attachment:image.png)


<!-- #endregion -->

## Discrete Vs Continous
- The first part of the algorithm will determine if the data is discrete on continous
- **Discrete**: Integers and Strings
- **Continous:** Floats


## Goodness Of Fit Test:  Chi – Square Test

![image.png](attachment:image.png)



<!-- #region -->
**ADDITIONAL INFO:**
- Used to compare if 2 sample distributions come from the same population distribution.
- If frequencies are too small the test will be invalid.
- Typical rule frequencies should be at least 5.


**Null Hypothesis:** H0 The distributions are the same
- Low Pvalue REJECT NULL
- Smaller the Chi-Stat the better

**Alternate Hypothesis:** H1 The distributions are different
- High Pvalue DO NOT REJECT NULL



I created a custom algorithm that performs the chi square test on the dataset feature and all of the 5 distribution types. The distribution with the smallest Chi Square stat is chosen as the MOST LIKELY distribution type for dataset.
<!-- #endregion -->

<!-- #region -->
## Classify Image: Covolutional Neural Network
![image.png](attachment:image.png)

I needed quality **LABELED** histogram images. The internet was not very much help so I decided to create a **“Random Histogram Generator Tool”**. This tool randomly generates the 5 distributions at random bin sizes.



**TRAINING DATA:** I randomly generated 5K images per class for a total of 25k images.


**VALIDATION DATA:** I randomly generated 100 images per class for a total of 500 images.


<!-- #endregion -->

![image.png](attachment:image.png)


# Algorithm Results
![image.png](attachment:image.png)

<!-- #region -->
# Research Findings
**Weibull Distribution:**
- When the Weibull lambda is small it behaves like the Exponential dist
- When the Weibull lambda is large it behaves like the Normal dist


**Gamma Distribution:**
- When the Gamma lambda is small it behaves like the Exponential dist
- When the Gamma lambda is large it behaves like the Poisson or Normal dist


**Beta Distribution:**
- When the Beta dist alpha is small and beta is large it behaves like the Exponential dist
- When the Beta dist alpha is large and beta is large it behaves like the Uniform dist
- When the Beta dist alpha is small and beta is small it behaves like the Bernoulli dist

<!-- #endregion -->

```python

```
