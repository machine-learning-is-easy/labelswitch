import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Optional
from torch.nn.functional import nll_loss, kl_div
class LabelWiseSignificanceCrossEntropy(nn.CrossEntropyLoss):
    r"""
    Examples::

        >>> # Example of target with class indices
        >>> loss = nn.CrossEntropyLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
        >>>
        >>> # Example of target with class probabilities
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5).softmax(dim=1)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0, alpha: float = 0.0,
                 device=None, num_class: int = 10, input_type="prob") -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing=label_smoothing)
        P0 = alpha/num_class
        self.zero = torch.tensor(0.0)
        self.expectation = torch.tensor(2 * P0)
        self.input_type = input_type

        if device:
            self.expectation = self.expectation.to(device)
            self.zero = self.zero.to(device)


    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # get right input and wrong input following
        if len(input.size()) > len(target.size()):
            class_num = input.size()[-1]
            target_digit = torch.nn.functional.one_hot(target, num_classes=class_num)
        else:
            target_digit = target

        if self.input_type != 'prob':
            input = torch.softmax(input, dim=-1)

        target_mask_0 = torch.logical_not(torch.logical_and(target_digit < 0.5, input.le(self.expectation)))
        input_new = torch.where(target_mask_0, input, self.zero)
        return super().forward(input_new, target)


class AdaptiveMaskedCrossEntropyLoss(nn.CrossEntropyLoss):
    """

    Examples::

        >>> # Example of target with class indices
        >>> loss = nn.CrossEntropyLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
        >>>
        >>> # Example of target with class probabilities
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5).softmax(dim=1)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    # get the top n differences

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0, alpha: float = 0.0, device=None,
                 num_class: int = 10) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction)
        P0 = alpha / num_class
        self.expectation = torch.tensor(P0 * 2)
        self.zero = torch.tensor(0.0)
        if device:
            self.expectation = self.expectation.to(device)
            self.zero = self.zero.to(device)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # get right input and wrong input following
        if len(input.size()) > len(target.size()):
            class_num = input.size()[-1]
            target_digit = torch.nn.functional.one_hot(target, num_classes=class_num)
        else:
            target_digit = target

        # get maximum of input with 0 target
        mask_index = target_digit.le(0.5)  # get the value of the 0 target
        maximum_value = torch.max(torch.masked_select(input, mask_index)) * 0.8
        expect = torch.minimum(maximum_value, self.expectation)
        target_mask_0 = torch.logical_and(target_digit < 0.5, input.le(expect))
        input_new = torch.where(torch.logical_not(target_mask_0), input, self.zero)
        return super().forward(input_new, target)