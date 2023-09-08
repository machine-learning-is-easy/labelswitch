import torch

class SwitchEncoding(torch.nn.Module):
    def __int__(self, num_labels):
        super().__init__()
        self.encode_transfer = torch.eye(num_labels, requires_grad=False)
        self.unswitched = set([_ for _ in range(num_labels)])
        self.num_labels = num_labels
    def forward(self, outputs, labels=None):
        if labels is None:
            return outputs * self.encode_transfer
        else:
            if self.num_labels:  # still has some label need to be switched
                for y, label in zip(outputs, labels):
                    switched_y_row = torch.argmax(y * self.encode_transfer)
                    if switched_y_row != label:
                        # change label to y
                        if label in self.unswitched and switched_y_row in self.unswitched:
                            # swap the columns of matrix
                            _unit = torch.eye(self.num_labels, requires_grad=False)
                            _unit[switched_y_row], _unit[label] = _unit[label], _unit[switched_y_row]
                            self.encode_transfer = self.encode_transfer * _unit
                            self.unswitched.pop(label)
                            self.unswitched.pop(switched_y_row)

                return outputs * self.encode_transfer
            else:
                return outputs * self.encode_transfer