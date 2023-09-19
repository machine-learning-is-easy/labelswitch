import torch
from collections import defaultdict
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
        self.switched_pair = []
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
                            self.switched_pair.append([label.item(), switched_y_row.item()])
                    else:
                        if label.item() in self.unswitched_label:
                            self.unswitched_label.remove(label.item())
            else:
                a = 1

        return torch.index_select(outputs, -1, self.index_selection)


class Resonance(torch.nn.Module):
    def __init__(self, num_labels, device=None):
        super().__init__()
        self.st = dict()
        self.index_selection = torch.tensor([_ for _ in range(num_labels)])
        if device and device.type != "cpu":
            self.index_selection = self.index_selection.to(device)
        self.device = device
        self.switched_pair = []
        self.state = 'buffering'
        self.num_label = num_labels
        self.map_complete = False
    def buffer(self, outputs, labels=None):
        if labels is not None:
            if len(outputs.shape) != len(labels.shape) + 1:
                raise Exception("Expected the dimension of output equal dimension of label + 1")

            if outputs.shape[-1] != self.num_label:
                raise Exception("Expect the last dimension of output equal number of label of this object")

            outputs = outputs.view(-1, self.num_label)
            labels = labels.flatten()
            output_digits = torch.argmax(outputs, -1)
            for y, label in zip(output_digits, labels):  # remove batch dimension
                if label.item() not in self.st:
                    self.st[label.item()] = dict()
                if y.item() not in self.st[label.item()]:
                    self.st[label.item()][y.item()] = 0
                self.st[label.item()][y.item()] += 1

    def create_map(self):
        switched_label = []
        switched_output = []
        switch_pair = []
        index_select = [None] * self.num_label

        # create pairs frequency
        pairs = []
        for label in self.st.keys():
            for output, fre in self.st[label].items():
                pairs.append((label, output, fre))
        pairs.sort(key=lambda x: x[2], reverse=True)

        for label, output, f in pairs:
            if label not in switched_label:
                if output not in switched_output:
                    index_select[label] = output
                    switched_label.append(label)
                    switched_output.append(output)
                    switch_pair.append((label, output))
                    if len(switched_label) == self.num_label:
                        break
        # does not find index, incase the training set does not include some label dataset
        unfound_label = set([_ for _ in range(self.num_label)]).difference(set(switched_label))
        unfound_output = set([_ for _ in range(self.num_label)]).difference(set(switched_output))

        if unfound_label:
            for ind in range(len(index_select)):
                if index_select[ind] is None:
                    index_select[ind] = unfound_output.pop()

        self.index_selection = torch.index_select(self.index_selection, -1, torch.tensor(index_select).to(self.device))
        self.map_complete = True
    def forward(self, outputs):
        return torch.index_select(outputs, -1, self.index_selection)