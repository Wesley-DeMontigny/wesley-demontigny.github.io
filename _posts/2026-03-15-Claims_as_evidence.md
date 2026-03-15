---
title: "When Are Claims Evidence?"
layout: post
date: 2026-03-15
categories: [Bayesian, Probability]
tags: [probability]
---
<script type="text/javascript"
  id="MathJax-script"
  async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

I recently watched a back and forth between two YouTubers who generally engage in discussions on philosophy, epistemology, and debate. The topic was about whether or not claims are evidence. One of the YouTubers (**A**) adamantly claims that claims are not evidence, while the other (**B**) says that this is clearly not the case. In one moment, **B** brought up an example. He considered the case of a friend claiming that he has bought a soccer ball. To **B**, this is clearly evidence of his friend having bought a soccer ball. **A** points out that it is not that simple and that when he treats this as evidence he is using a LOT of background knowledge and circumstantial evidence to come to that conclusion, even if he doesn't realize it.

I think **A** is on the right track here and that **B** is being a little simplistic, but the language of this discussion is not well suited for the topic. I think some of the language of probability will help us here, because although logical syllogisms of the form  

$$
C \implies A; \quad C \therefore A
$$

are useful in philosophy, real life deals in degrees of uncertainty. Claims can be expressed as a kind of measurement in a latent variable problem.

Let's say we are interested in the truth of some claim $$z$$. Without loss of generality, let us consider $$z$$ as a binary variable, either taking on values $$z=0$$ or $$z=1$$. We can observe some data that (hopefully) indirectly measures $$z$$, which we will denote as the vector $$\mathbf{x}$$, which contains potentially multiple kinds of measurements $$x_i$$ for $$i \in 1,\dots,n$$. We want to understand $$\mathbf{x}$$ as arising from some kind of emission process from $$z$$.

Let us consider the soccer ball example, with $$z=0$$ denoting that the friend has not bought the soccer ball and $$z=1$$ denoting that our friend has bought the ball. We will consider our measurement of this latent variable $$x$$ to be one-dimensional, with $$x=0$$ meaning that our friend claims he has not bought a soccer ball and $$x=1$$ meaning he claims he has bought a soccer ball.

If our friend is perfectly truthful, then our result is clear and $$x=z$$. However, there are a number of things that could complicate this. The easiest thing is to question our friend's truthfulness. Suppose that our friend is embarrassed about his soccer interests: he will never falsely claim to buy a ball, but may lie about not buying a ball. Let us consider that he lies about his soccer interests with frequency $$1-p$$. Suddenly, our observation $$x$$ does not perfectly correspond to $$z$$. Instead, $$x$$ is emitted by $$z$$ according to the process

$$
x \sim \text{Bernoulli}(p z).
$$

That is, when $$z=1$$ we get $$x=1$$ with probability $$p$$, and when $$z=0$$ we get $$x=0$$ with probability $$1$$.

If we are interested in the probability that $$z$$ is true, we need to consider the conditional probability of $$z$$ given $$x$$, denoted $$P(z \mid x)$$. In this case, if we are interested in the probability of $$z$$ given some observation $$x$$, we have

$$
\begin{aligned}
P(z=0 \mid x) &\propto P(z=0)P(x \mid z=0)
= 
\begin{cases}
(1-\psi) & \text{if } x=0 \\
0 & \text{if } x=1
\end{cases} \\\\
P(z=1 \mid x) &\propto P(z=1)P(x \mid z=1)
=
\begin{cases}
\psi (1-p) & \text{if } x=0 \\
\psi p & \text{if } x=1
\end{cases}
\end{aligned}
$$

Notice that an extra term has appeared here, $$\psi$$. This is the prior probability that our friend (or maybe anyone, depending on how you'd like to set it up) would buy a soccer ball.

Here I have not written out the normalization constants (hence the symbol $$\propto$$). If we consider the case where $$x=0$$, that is, he has claimed that he has not bought a soccer ball, we get a probability of truth equal to

$$
P(z=1 \mid x=0) = 
\frac{\psi (1-p)}{(1-\psi)+\psi (1-p)}
=
\frac{\psi - \psi p}{1 - \psi p}
$$

By this very simple introduction of untruthfulness, we have injected some very serious assumptions. If we do not have perfect correspondence of $$x$$ to $$z$$, we now need to rely not only on the truthfulness of our friend, $$p$$, but also on how reasonable it is that he would buy a soccer ball in the first place, $$\psi$$. 

Notice that if $$\psi$$ is extremely low, then our friend saying he has bought a soccer ball is essentially worthless in convincing us. That is, even if our friend is the most truthful person we've ever met, if soccer balls are extremely rare then we would probably conclude he's lying. If we change the soccer ball example to our friend saying he bought a purple alien on the black market, all of a sudden not only do we not change our belief very much about $$z$$, but we may update our beliefs about $$p$$, the truthfulness of our friend.

Things get even more complicated when our friend could be either untruthful or just gullible (or both). Let us consider the case where we are interested in whether $$z=1$$, but there is some distractor $$\tilde z$$ that, when $$\tilde z = 1$$, can lead our friend into thinking $$z=1$$. Suppose in this case our friend believes he has won some sweepstakes from a random email he received, but we know that plenty of these fake emails circulate. In this case we will say that our friend may lie about winning, resulting in $$x=1$$ when $$z=0$$ with probability $$(1-p)$$, or that he may have fallen for a false sweepstakes $$\tilde z = 1$$ with probability $$q$$. We will assume, for simplicity, that $$x=1$$ when $$z=1$$; that is, our friend would always tell us if he had actually won a sweepstakes (maybe he is down on his luck and would be too excited not to share). This relatively simple real-life problem introduces a lot of different variables into our system. We now have unnormalized probabilities

$$
\begin{aligned}
P(z=0 \mid x) &\propto P(z=0)P(x \mid z=0) \\
&= \alpha P(z=0)P(x \mid z=0,\tilde z=1)
+ (1-\alpha)P(z=0)P(x \mid z=0,\tilde z=0) \\
&=
\begin{cases}
\alpha (1-q)(1-\psi)+(1-\alpha)(1-\psi) & \text{if } x=0 \\
\alpha q(1-\psi)+(1-\alpha)(1-p)(1-\psi) & \text{if } x=1
\end{cases}
\\\\
P(z=1 \mid x) &\propto P(z=1)P(x \mid z=1)
=
\begin{cases}
0 & \text{if } x=0 \\
\psi & \text{if } x=1
\end{cases}
\end{aligned}
$$

With our normalized probability of observing our friend telling the truth and making the claim being

$$
P(z=1 \mid x=1) =
\frac{\psi}{\psi + \alpha q(1-\psi) + (1-\alpha)(1-p)(1-\psi)}.
$$

Notice that we now have a mixture of processes happening here. We have a friend that is exposed to fraudulent sweepstakes at a frequency $$\alpha$$. Think about how simple this example was. The situation is not particularly complex, and yet we suddenly have a lot of factors to consider. If we also had to assess uncertainty in each of the parameters mentioned above (say we aren't certain about $$\alpha$$ or $$p$$), or if we had uncertainty in the generative model itself, then the situation becomes much closer to claims contributing essentially no evidence. Here, I also made the basic assumption that the lying was one-way! This is also generally not the case and would significantly change how we update our beliefs given a claim.

YouTuber **A** was right: the evidential value of claims is incredibly dependent on our prior knowledge of the subject. But I do not know if I would strictly phrase it as "claims not being evidence." It is more like this: without strong background and domain knowledge on a given subject, claims themselves should not have much sway on our beliefs, as they are extremely weak measurements of the truth.
