import torch

class SwitchEncoding(torch.nn.Module):
    def __init__(self, num_labels, device=None):
        super().__init__()
        self.encode_transfer = torch.eye(num_labels, requires_grad=False)
        self.unswitched = set([_ for _ in range(num_labels)])
        self.num_labels = num_labels

        if device and device.type != "cpu":
            self.encode_transfer = self.encode_transfer.to(device)
        self.device = device
    def forward(self, outputs, labels=None):
        if labels is None:
            if hasattr(self, "index_selection"):
                return torch.index_select(outputs, 1, self.index_selection)
            else:
                return torch.inner(outputs, self.encode_transfer)
        else:
            if self.unswitched:  # still has some label need to be switched
                for y, label in zip(outputs, labels):
                    switched_y_row = torch.argmax(torch.inner(y, self.encode_transfer))
                    if switched_y_row.item() != label.item():
                        # change label to y
                        if label.item() in self.unswitched:
                            # swap the columns of matrix
                            _unit = torch.eye(self.num_labels, requires_grad=False)
                            select_col = [ind for ind in range(self.num_labels)]
                            select_col[switched_y_row.item()], select_col[label.item()] = select_col[label.item()], select_col[switched_y_row.item()]
                            # switch row of unit
                            _unit = torch.index_select(_unit, 0, torch.tensor(select_col))
                            _unit = _unit.to(self.device)
                            self.encode_transfer = torch.inner(self.encode_transfer, _unit)
                            self.unswitched.remove(label.item())
                    else:
                        if label.item() in self.unswitched:
                            self.unswitched.remove(label.item())
                # checking if the labels has been switched
                if self.unswitched:
                    return torch.inner(outputs, self.encode_transfer)
                else:
                    # convert all the encode_transfer matrix to index_selection
                    self.index_selection = torch.argmax(self.encode_transfer, 1)
                    return torch.index_select(outputs, 1, self.index_selection)
            else:
                return torch.index_select(outputs, 1, self.index_selection)