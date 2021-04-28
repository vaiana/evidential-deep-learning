r"""
Loss functions from 
    Evidential Deep Learning to Quantify Classification Uncertainty
    Sensoy et al

See: 
    https://arxiv.org/pdf/1806.01768.pdf


This module defines the expected mean squared error loss under a Dirichlet 
prior (eq 5 from the paper) and the KL Divergence from a uniform Dirichlet as 
the regularization term. The total loss we would like to learn from is

.. math::
    mse + \lambda * kl_regularization

where the factor :math:`\lambda` is annealed from 0 to 1 (see discussion on 
page 6 of the paper).

The main components defined herien:

    1. The dirichlet expected mse loss
    2. The uniform dirichlet kl regularization

The term *evidence* refers to :math:`g(\text{model(x)})` where 
:math:`g` is any non-negative transformation.  Note the final layer of 
the model should have no activation and simply return raw logits.

References:
    1. https://arxiv.org/pdf/1806.01768.pdf
    2. https://muratsensoy.github.io/uncertainty.html
    3. http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/

"""
import torch


class EDLLoss(torch.nn.Module):
    r"""Evidential Deep Learning Loss

    Creates a criterion that measures the expected mean squared error (MSE) 
    of the true class label :math:`y` and the probability mass function 
    :math:`p` with the expectation taken over :math:`p\sim\text{Dir}(\alpha)`
    where :math:`\text{Dir}(\alpha)` is the Dirichlet distribution.

    Args:
        reg_factor: multiplier for the KL-uniform divergence regularization 
            term, optionally it can have a `step` method which takes no 
            parameters and updates the regularization factor

    """

    def __init__(self, reg_factor=None):
        super().__init__() 
        self.reg_factor = reg_factor

    def forward(self, input, target):
        loss  = dirichlet_mse_loss(input, target)

        if self.reg_factor and self.reg_factor!=0:
            reg = self.reg_factor * uniform_dirichlet_kl(input, target)
            loss = loss + reg 
       
        # allow regularization factor to change over time
        if hasattr(self.reg_factor, 'step'):
            self.reg_factor.step()
        
        return loss



def dirichlet_mse_loss(evidence, target):
    r"""Expected MSE Loss with dirichlet prior

    Args:
        evidence: evidence tensor of the form :math:`g(model(x))` where 
            :math:`g` is any non-negative transformation.
        target: the true class labels in sparse or one-hot format
  
    Returns:
        mse scalar loss

    """
    num_classes = evidence.shape[1] 
    alpha = evidence + 1  # (batch_size, num_clases)
    s_alpha = torch.sum(alpha, dim=1, keepdim=True)  # (batch_size,1)
    p_hat = alpha / s_alpha  # (batch_size, num_clases), \hat{p_ij} from paper


    # Sparse targets
    if target.dim()==1:
        y_index = target.long().view(-1,1) 
        src = p_hat.gather(dim=1, index=y_index) -1 
        diff = p_hat.scatter(dim=1, index=y_index, src=src)
        error_term = torch.square(diff)
    # One-hot targets
    elif target.dim()==2:
        error_term = torch.square(target-p_hat)
    else:
        raise ValueError(f"Targets must be of shape (N,) or (N,C) "
                         f"but given shape {targets.size()}")


    var_term = p_hat * (1 - p_hat) / (s_alpha + 1)  # (batch_size, num_classes)
    mse = torch.sum(error_term + var_term, dim=1)
    return torch.mean(mse)




def uniform_dirichlet_kl(evidence, target):
    r"""Kullback-Leibler divergence from uniform Dirichlet


    Args:
        evidence: evidence tensor of the form :math:`g(model(x))` where 
            :math:`g` is any non-negative transformation.
        target: the true class labels in sparse or one-hot format
    
    Returns:
        kl scalar loss

    Note:
        In general the beta term below should be 
        `beta_term = torch.sum(lgamma(beta),dim=1, keepdim=True) - lgamma(beta)`
        but the first term (the sum) is 0 since beta is a vector of ones so we 
        omit it.


    See Also
        http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/

    """
    lgamma = torch.lgamma
    digamma = torch.digamma
    device = evidence.device
    num_classes = evidence.shape[1]

    alpha = evidence + 1  #( batch_size, num_classes)
    beta = torch.ones((1, num_classes), device=device)  # (1, num_classes)

    # Sparse targets
    if target.dim()==1:
        target = target.long().view(-1, 1)  # (batch_size,1) for alpha.scatter

        # \hat{alpha} from paper
        # 1 in correct class index, alpha in wrong class indicies
        src = torch.ones(target.shape, dtype=alpha.dtype, device=device)
        alpha = alpha.scatter(dim=1, index=target, src=src)

    # One-hot targets
    elif target.dim()==2:
        alpha = target + (1-target) * alpha

    # sums
    s_alpha = torch.sum(alpha, dim=1, keepdim=True)  # (batch_size, 1)
    s_beta = torch.sum(beta)  # scalar == num_classes

    # compute the terms contributing to final sum
    alpha_term = lgamma(s_alpha) - torch.sum(lgamma(alpha), dim=1)
    beta_term = -lgamma(s_beta)
    digamma_term = (alpha - beta) * (digamma(alpha) - digamma(s_alpha))
    digamma_term = torch.sum(digamma_term, dim=1)

    return torch.mean(alpha_term + beta_term + digamma_term)



