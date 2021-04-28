import pytest

import torch
from edl.loss import dirichlet_mse_loss
from edl.loss import uniform_dirichlet_kl


@pytest.fixture
def near_zero_loss_setup():
    NUM_CLASSES = 5
    BATCH_SIZE = 10

    evidence = torch.zeros(BATCH_SIZE, NUM_CLASSES)
    evidence[:,0] = 1e5 # lots of evidence for class 0
    y_true = torch.zeros(BATCH_SIZE) # truth is class 0

    yield evidence, y_true

@pytest.fixture
def near_zero_loss_dense_setup():
    NUM_CLASSES = 5
    BATCH_SIZE = 10

    evidence = torch.zeros(BATCH_SIZE, NUM_CLASSES)
    evidence[:,0] = 1e5 # lots of evidence for class 0
    y_true = torch.zeros(BATCH_SIZE, NUM_CLASSES) 
    y_true[:,0] =1 # truth is class 0

    yield evidence, y_true






def test_mse_loss_is_zero(near_zero_loss_setup):
    # WHEN 
    evidence, y_true = near_zero_loss_setup

    # THEN
    assert dirichlet_mse_loss(evidence, y_true).mean() < 1e-5
    
def test_mse_loss_is_zero_dense(near_zero_loss_dense_setup):
    # WHEN 
    evidence, y_true = near_zero_loss_dense_setup

    # THEN
    assert dirichlet_mse_loss(evidence, y_true).mean() < 1e-5
 

def test_uniform_dirichlet_kl_is_zero(near_zero_loss_setup):
    # WHEN 
    evidence, y_true = near_zero_loss_setup

    # THEN 
    assert uniform_dirichlet_kl(evidence, y_true).mean() < 1e-5


def test_uniform_dirichlet_kl_is_zero_dense(near_zero_loss_dense_setup):
    # WHEN 
    evidence, y_true = near_zero_loss_dense_setup

    # THEN 
    assert uniform_dirichlet_kl(evidence, y_true).mean() < 1e-5




def test_mse_loss_is_scalar(near_zero_loss_setup):
    # When
    evidence, y_true = near_zero_loss_setup

    # Then
    assert dirichlet_mse_loss(evidence, y_true).shape == ()

def test_kl_loss_is_scalar(near_zero_loss_setup):
    # When
    evidence, y_true = near_zero_loss_setup

    # Then
    assert uniform_dirichlet_kl(evidence, y_true).shape == ()

