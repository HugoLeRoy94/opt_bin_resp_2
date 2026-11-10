"""The objective is two independent choices, and none of them maximizes H(Y).

    ESTIMATOR  mi = "shannon" (exact) or "kt" (lower bound, scales)
    GROUPING   grouped = cells with identical abundance rows become one group

Maximizing H(Y) alone is the trap these tests exist to prevent: an array whose
outputs are fair coins regardless of input scores the maximum H(Y) and carries
zero information.
"""
import pytest
import torch

from estimate.entropy import (compute_response_conditional_entropy,
                              compute_shannon_joint_entropy)
from estimate.losses import (ExactMutualInformationLoss,
                             GroupedCellMutualInformationLoss,
                             GroupedKTMutualInformationLoss,
                             KTMutualInformationLoss, ResponseEntropy)


def test_entropy_alone_prefers_a_useless_array():
    """Why no objective maximizes H(Y). A fair-coin array wins on entropy."""
    B, R = 64, 8
    coins = torch.full((B, R), 0.5)               # ignores the input entirely
    torch.manual_seed(0)
    codes = (torch.rand(B, R) > .5).float()
    informative = codes * (1 - 1e-3) + (1 - codes) * 1e-3

    measure = ResponseEntropy(entropy_type="kt", collision_chunk_size=256)
    assert measure.compute_entropy(coins) > measure.compute_entropy(informative)

    # The objectives get it the right way round.
    loss = KTMutualInformationLoss(collision_chunk_size=256)
    assert loss(coins) > loss(informative)        # loss is negated MI


def test_the_measurement_base_cannot_be_trained():
    """ResponseEntropy has no forward, so H(Y) cannot be selected as a target."""
    assert not hasattr(ResponseEntropy, "forward") or \
        ResponseEntropy.forward is torch.nn.Module.forward


def test_grouping_is_a_no_op_when_every_cell_differs():
    """Grouping loses nothing: the count vector is a sufficient statistic."""
    torch.manual_seed(0)
    C = 6
    activity = torch.rand(200, C)
    grouped = GroupedCellMutualInformationLoss(torch.eye(C))
    exact = ExactMutualInformationLoss()
    assert grouped.compute_metrics(activity)["mutual_information_grouped"].item() == \
        pytest.approx(-exact(activity).item(), abs=1e-5)


def test_exact_mi_is_entropy_minus_noise():
    """The new ungrouped cell of the 2x2: I(Y;X) = H(Y) - H(Y|X), both exact."""
    torch.manual_seed(1)
    activity = torch.rand(128, 7)
    soft = torch.stack((1 - activity, activity), -1)
    expected = (compute_shannon_joint_entropy(soft)
                - compute_response_conditional_entropy(activity))
    assert -ExactMutualInformationLoss()(activity).item() == pytest.approx(
        expected.item(), abs=1e-6)


@pytest.mark.parametrize("mi,grouped,expected", [
    ("shannon", False, ExactMutualInformationLoss),
    ("shannon", True, GroupedCellMutualInformationLoss),
    ("kt", False, KTMutualInformationLoss),
    ("kt", True, GroupedKTMutualInformationLoss)])
def test_every_combination_builds(mi, grouped, expected):
    """All four cells exist; neither axis is a special case of the other."""
    from configs import CELL, RECEPTOR

    from estimate.grouping import CellGrouping
    from train.config import Config
    from train.runner import build_loss

    fields = dict(CELL if grouped else RECEPTOR, mi=mi, grouped=grouped)
    config = Config(**fields)
    grouping = CellGrouping(torch.eye(len(config.cell_gene_sets))) if grouped else None
    assert isinstance(build_loss(config, grouping), expected)


def test_grouping_without_cells_is_refused():
    """Receptors have nothing to group, so asking for it is a design error."""
    from configs import RECEPTOR

    from train.config import Config
    with pytest.raises(ValueError, match="Grouping requires cell mode"):
        Config(**dict(RECEPTOR, grouped=True))
