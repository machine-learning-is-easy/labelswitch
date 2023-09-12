import torch

class LabelSwitch(torch.nn.Module):
    def __init__(self, num_labels, device=None):
        super().__init__()
        self.encode_transfer = torch.eye(num_labels, requires_grad=False)
        self.unswitched_label = set([_ for _ in range(num_labels)])
        self.unswitched_output = set([_ for _ in range(num_labels)])
        self.index_selection = torch.tensor([_ for _ in range(num_labels)])
        self.num_labels = num_labels

        if device and device.type != "cpu":
            self.encode_transfer = self.encode_transfer.to(device)
            self.index_selection = self.index_selection.to(device)
        self.device = device
    def forward(self, outputs, labels=None):
        if labels is not None:
            if len(outputs.shape) != len(labels.shape) + 1:
                raise Exception("Expected the dimension of output equal dimension of label + 1")

            if outputs.shape[-1] != self.num_labels:
                raise Exception("Expect the last dimension of output equal number of label of this object")

            outputs = outputs.view(-1, self.num_labels)
            labels = labels.flatten()
            if self.unswitched_label:  # still has some label need to be switched
                for y, label in zip(outputs, labels):  # remove batch dimension
                    switched_y_row = torch.argmax(torch.index_select(y, -1, self.index_selection))
                    if switched_y_row.item() != label.item():
                        # change label to y
                        if label.item() in self.unswitched_label:
                            # swap the columns of matrix
                            _unit = torch.eye(self.num_labels, requires_grad=False)
                            select_col = [ind for ind in range(self.num_labels)]
                            select_col[switched_y_row.item()], select_col[label.item()] = select_col[label.item()], select_col[switched_y_row.item()]
                            # switch row of unit
                            _unit = torch.index_select(_unit, 0, torch.tensor(select_col))
                            _unit = _unit.to(self.device)
                            self.encode_transfer = torch.inner(self.encode_transfer, _unit)
                            self.index_selection = torch.argmax(self.encode_transfer, 1)
                            self.unswitched_label.remove(label.item())
                    else:
                        if label.item() in self.unswitched_label:
                            self.unswitched_label.remove(label.item())

        return torch.index_select(outputs, -1, self.index_selection)