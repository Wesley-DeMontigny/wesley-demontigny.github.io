---
title: "Variational Bayes from a Generalized Bayesian Inference Perspective"
layout: post
date: 2025-04-09
categories: [Bayesian, Variational Inference]
tags: [generalized bayesian, VI, inference]
---
<script type="text/javascript"
  id="MathJax-script"
  async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
Markov Chain Monte Carlo (MCMC) is an amazing tool for Bayesian inference, but it's unfortunately quite computationally intensive. Often, that's fine, and we're willing to wait for an asymptotically exact answer. But that isn't always the case. Certain models are so complicated that the Markov chain will fail to converge until we are all dead. And depending on the application, even a day may be far too long to wait for a model to be fit! In these situations, we need an alternative to MCMC.

Variational inference (VI) is one such alternative. It aims to approximate the posterior distribution by using a tractable "variational" distribution to represent the true posterior. This turns Bayesian inference into an optimization problem: instead of sampling from the posterior, we optimize the parameters of the variational distribution to make it as close to the true posterior as possible. For example, suppose we believe that our posterior is roughly Gaussian-shaped. In that case, we may use a Gaussian as our variational distribution for our parameter \(\theta\), and optimize the mean \(\mu\) and variance \(\sigma^2\) accordingly. While this approach has its downsides, it is generally much faster than MCMC.

When I first encountered these methods, I thought it was interesting that the derivation for the objective function- the **E**vidence **L**ower **Bo**und or ELBO - wasn't immediately intuitive. Sure, it all makes sense once you know it, but I wouldn't have derived it myself on my first try. Lately, I've been reading a lot of material on Generalized Bayesian Inference (GBI), which I will review later in this post. Once you really understand that framework, I think it offers a much stronger motivation for the ELBO than the standard derivation.
### Traditional Derivations for Variational Inference
Suppose we have some variational distribution \(q(\theta | \psi)\) that we want to use as an approximation for the posterior distribution \(p(\theta | x)\). The most sensible thing to do would be to pick the parameter \(\psi\) such that the KL divergence between the variational distribution and posterior is minimized:
\[KL(q(\theta|\psi) ||p(\theta|x)) = E_{\theta \sim q}(\text{log }\frac{q(\theta|\psi)}{p(\theta|x)})\]
We can expand \(p(\theta | x) = \frac{p(\theta, x)}{p(x)}\), so that we get:
\[=E_{\theta \sim q}(\text{log }\frac{q(\theta|\psi)}{p(\theta,x)} + \text{log }p(x))\]
\[=E_{\theta \sim q}(\text{log }\frac{q(\theta|\psi)}{p(\theta,x)}) + \text{log }p(x) = \text{log } p(x) - \text{ELBO}\]
Since \(p(x)\), the marginal likelihood, is a constant, we ignore it for our optimization and focus on the ELBO. By maximizing the ELBO, we will minimize the KL divergence. We can rewrite the ELBO like so:
\[\text{ELBO} = E_{\theta\sim q}(\text{log } p(\theta,x)) - E_{\theta\sim q}(\text{log } q(\theta|\psi))\]
\[= E_{\theta \sim q}(\text{log }p(x|\theta)) + E_{\theta\sim q}(\text{log } p(\theta)) - E_{\theta\sim q}(\text{log } q(\theta|\psi))\]
\[= E_{\theta \sim q}(\text{log }p(x|\theta)) - KL(q(\theta|\psi) || p(\theta))\]
So, to do variational inference, we pick a \(\psi\) to maximize the expected likelihood and minimize the divergence of the variational distribution and the prior. This all makes sense, but I don't know that it would have occurred to me if I was the one tackling the problem for the first time. Let's review GBI and see what insights it has to offer.
### The Generalized Bayesian Inference Approach
This is just going to be a quick overview of GBI, but it is SO cool. For a comprehensive review, I really recommend Bissiri, Holmes & Walker, 2016 ([A General Framework for Updating Belief Distributions](https://arxiv.org/abs/1306.6430)). In brief, GBI aims to extend Bayesian updating to models that do not necessarily have a likelihood but do have some loss function \(\ell_\theta(x)\) that indicates the agreement between some parameter \(\theta\) and observed data \(x\). In such a scenario, how does one update some prior belief \(p(\theta)\)? We can think about all hypothetical posterior distributions \(\pi\), and we will aim to choose the updating scheme resulting in:
\[\pi^* = argmin_\pi R(\pi, p(\theta), x)=f(\pi,p(\theta)) + g(\pi,x)\]
where \(R\) is a risk function composed of a loss function \(f\) that measures the posterior's agreement with the prior and a loss function \(g\) that measures the posterior's agreement with the data. This looks like something that would be insanely difficult to solve, but it actually is pretty trivial if we pick some clever choices for \(f\) and \(g\). We will say:
\[f(\pi,p(\theta))=E_{\theta\sim\pi}(\ell_\theta(x))\]
\[g(\pi,x)=KL(\pi(\theta)||p(\theta))\]
We can therefore represent the risk function \(R\) as:
\[R(\pi,p(\theta),x)=\int\pi(\theta)(\ell_\theta(x)+\text{log }\frac{\pi(\theta)}{p(\theta)})d\theta\]
We can move \(\ell_\theta(x)\) inside the log so we get:
\[R(\pi,p(\theta),x)=\int\pi(\theta)\text{log }\frac{\pi(\theta)}{exp(-\ell_\theta(x))p(\theta)}d\theta = KL(\pi(\theta) || exp(-\ell_\theta(x))p(\theta))\]
The posterior that minimizes this KL divergence is just \(\pi^*(\theta) \propto exp(-\ell_\theta(x))p(\theta)\)! We call this the "Gibbs posterior" because it looks like a Gibbs distribution with an energy function \(\ell_\theta\) multiplied by some prior. If we choose a loss function \(\ell_\theta(x) = - \text{log } p(x | \theta)\), we get the classical posterior, \(\pi(\theta) \propto p(x|\theta)p(\theta)\)! However, something really amazing here is that we could have chosen another loss function, and we would still be able to do Bayesian updating. This isn't exactly a trivial thing to do, but it is totally legitimate under GBI.

Now, how does this connect to variational inference? Instead of considering all possible posteriors, let's assume the posterior takes a form \(q(\theta | \psi)\) for our parameters of interest \(\theta\) and tunable parameters \(\psi\). We will also choose the classical loss function \(\ell_\theta(x) = -\text{log } p(x|\theta)\). In that case, our risk minimization looks like
\[q^*=argmin_\psi(\psi,p(\theta),x) = - E_{\theta\sim q}(\text{log } p(x|\theta)) + KL(q(\theta|\psi)||p(\theta))\]
We can rewrite this as:
\[q^*=argmax_\psi(\psi, p(\theta), x)=E_{\theta\sim q}(\text{log } p(x|\theta)) - KL(q(\theta|\psi)||p(\theta))\]
This is the ELBO! So, under GBI, VI is just a special case where we assume that the distribution takes a particular form.
### Concluding Remarks
I hope that all made sense and was useful! I have been thinking about GBI a lot, so this felt more intuitive. Under the GBI perspective, VI isn't a cheap method for faster inference but a natural method for those willing to constrain their updated belief to a family of distributions. In this sense, VI can be viewed as a fully Bayesian update procedure.
